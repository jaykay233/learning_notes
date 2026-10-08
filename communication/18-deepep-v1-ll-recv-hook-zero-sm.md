# DeepEP V1：Low-latency `return_recv_hook` 与「0 SM」重叠

> 承接 [12](./12-deepep-v1-low-latency-dispatch-kernel.md)–[14](./14-deepep-v1-low-latency-combine-and-pack.md)、[17](./17-deepep-v1-internode-deep-dive-qa.md)。LL 路径已讲清 SEND/RECV、staging、门铃；本文钉死 **hook 机制**：如何把 kernel 拆成两枪，以及文档里「without any SM occupation」到底指哪一段。
>
> 代码：`deep_ep/buffers/legacy.py`；`csrc/legacy/buffer.hpp`（`low_latency_dispatch/combine`）；`csrc/kernels/legacy/internode_ll.cu`、`compiled.cuh`；说明见 `docs/legacy.md`。传输原语：NVSHMEM IBGDA `put_nbi` / AMO。

```text
本次讲解位置
章节：communication / DeepEP V1
小节：LL recv hook / 通信计算重叠
知识点：phases=SEND|RECV；hook 只发再收；nbi 后 NIC DMA；「0 SM」= 等网窗口不占 SM
上次：跨机深挖问答（17）
下次：V2 ElasticBuffer / hybrid dispatch，或 SourceMeta 位图
PTX / 原语：nvshmemi_ibgda_put_nbi_warp、amo_nonfetch_add、compute_stream vs comm_stream
```

## 为什么现在讲这个

读完 LL dispatch/combine 后，最容易把三句话拧在一起：

1. 「hook = 整次通信 0 SM」——不对，SEND/RECV kernel 仍占 SM。
2. 「hook 只是 Python 回调」——不对，本质是 **再 launch 一次只含 RECV 的 kernel**。
3. 「和 `async_finish` 一样」——不对，二者互斥；hook 打在 compute stream 上。

弄清「谁 post、谁 DMA、谁 poll」，才能理解双 microbatch 重叠图里中间那段为什么能和 Attention/GEMM 叠上。

---

## 1. API：`return_recv_hook=True` 时发生什么

Python 文档（legacy.py）：

```text
若设置 return_recv_hook：kernel 只做 RDMA request issues，
并不真正「收齐」数据；必须再调 hook() 才能保证数据可用。
```

C++ 实现（`buffer.hpp`）：

```text
launcher(return_recv_hook
    ? LEGACY_LOW_LATENCY_SEND_PHASE                          // 只发
    : (LEGACY_LOW_LATENCY_SEND_PHASE | LEGACY_LOW_LATENCY_RECV_PHASE));  // 一次做完

if (return_recv_hook)
    recv_hook = [=]() { launcher(LEGACY_LOW_LATENCY_RECV_PHASE); };
```

| `return_recv_hook` | 第一次 launch | 返回的 hook |
|---|---|---|
| `False` | `SEND \| RECV` | 无 |
| `True` | **只 `SEND`** | 调用时再 launch **只 `RECV`** |

phase 常量（`compiled.cuh`）：

```text
LEGACY_LOW_LATENCY_SEND_PHASE = 1
LEGACY_LOW_LATENCY_RECV_PHASE = 2
```

Kernel 入口按 bit 跳过另一半（`internode_ll.cu` dispatch；combine 同理）：

```text
if ((phases & SEND) == 0) goto RECV;
// ... SEND 体 ...
RECV:
if ((phases & RECV) == 0) return;
```

Combine 用同一套 phase / hook 约定（`buffer.hpp` `low_latency_combine`）。

---

## 2. 「0 SM」到底省哪一段

文档（`docs/legacy.md`）：hook 用于 double-batch overlapping，**RDMA network traffic 在后台发生，不从计算部分占用 GPU SM**。

```text
普通模式（无 hook）
  [======== SEND+RECV kernel 占 SM ========] → 才能用 recv_x / GEMM
         └─ 大量时间在 poll count/flag + 打包

hook 模式
  [== SEND kernel 占 SM ==]  put_nbi + 写门铃
           │
           ├─ NIC DMA：数据飞向对端 staging（不占 GPU SM） ──┐
           │                                                  │
           └─ 本卡 SM 去做 Attention / 另一 batch GEMM ───────┘ 重叠
           │
           ▼
  hook() → [== RECV kernel 占 SM ==] 等门铃、打包 / reduce
```

| 阶段 | 占不占 SM | 谁在干活 |
|---|---|---|
| SEND kernel | **占** | GPU：装箱、`put_nbi`、AMO 门铃 |
| SEND 之后～hook 之前 | **相对「等收」不占** | **NIC** 做 RDMA DMA |
| hook → RECV kernel | **占** | GPU：等 count/flag、搬 `packed_recv_x` / 加权 reduce |

**不是「整次 LL 通信 0 SM」**，而是：**等网的那段不空转 SM**。

测试用法（`tests/legacy/test_low_latency.py`）：

```python
def large_gemm_with_hook(hook):
    mat_0 @ mat_1   # 占 SM 做计算
    hook()          # 再启 RECV
```

---

## 3. 为何能「后台飞」：NVSHMEM IBGDA `put_nbi`

SEND 跨节点（`internode_ll.cu`）：

```text
dst_p2p_ptr = nvshmemi_get_p2p_ptr(...)
if dst_p2p_ptr == 0:
    nvshmemi_ibgda_put_nbi_warp(...)   # 非阻塞：GPU post WQE 后返回
else:
    warp 直接 st 写对端（同节点 NVLink P2P）
```

门铃同样走 NVSHMEM：

| 路径 | 门铃 |
|---|---|
| LL dispatch | `rdma_recv_count = -n-1`（AMO） |
| LL combine | `rdma_recv_flag = 1`（AMO / P2P store） |

`nbi` = non-blocking immediate：GPU 只负责 **post**；PCIe/IB 上的数据搬运由 **NIC DMA** 完成，不需要一颗 SM 跟着拷。有 hook 时 post 完就让出 SM；无 hook 时同一 kernel 立刻进 RECV 轮询，SM 被占住。

---

## 4. Host 流与约束（`buffer.hpp`）

```text
launch_stream = return_recv_hook ? compute_stream : comm_stream
断言：not (async and return_recv_hook)
无 hook 时：先 stream_wait(comm, compute)，结束再 wait 回来
有 hook 时：不在这里等 RECV；由用户稍后 hook()
```

| 点 | 含义 |
|---|---|
| Hook 打在 **compute stream** | 与后续 GEMM 同流，方便 `SEND → GEMM → hook` |
| 普通模式用 **comm_stream** | 流间 overlap（另一套重叠方式） |
| 与 `async_finish` **互斥** | hook 模式自己管「何时收齐」 |

---

## 5. 双 microbatch 重叠在干什么

```text
Batch A:  Attn ── SEND_A ──(RDMA 飞)── hook_A ── MoE ── SEND_comb ── hook_comb
Batch B:            Attn ── SEND_B ──(RDMA 飞)── ...
                      ↑                ↑
                   占 SM            不占 SM（NIC）
```

中间「RDMA 飞」与另一批的 Attention/GEMM 重叠；各段时长不一定对齐，需按 workload 调阶段切分（见 `docs/legacy.md` 配图）。

---

## 6. 心智模型 vs 易混点

```text
Mental model
  SEND = post（GPU）
  中间 = NIC DMA（0 SM 等待）
  hook/RECV = poll + pack（GPU）
```

| 误读 | 正确 |
|---|---|
| 整条 LL 0 SM | 仅 SEND↔RECV **之间** 等网不占 SM |
| hook 是空回调 / CPU 收包 | 再 launch 一次 `RECV_PHASE` kernel |
| hook ≈ `async_finish` | 互斥；hook 用 compute stream |
| 同节点也走 IBGDA | 有 P2P 指针时直接写对端显存 |
| V2「0 SM Engram/PP」= 本机制 | 另一套；README 称 **0 SM RDMA LL EP 已不再支持** |

---

## 7. 可对照的最小流程（伪代码）

```python
# 概念流程，非独立可跑脚本；完整见 tests/legacy/test_low_latency.py
recv_x, count, handle, event, hook = buffer.low_latency_dispatch(
    x, topk_idx, num_tokens, num_experts,
    async_finish=False, return_recv_hook=True,
)
# 此处 RDMA 已在飞；SM 可做别的
do_independent_compute()
hook()  # launch RECV：等 count、打包 expert-major

combined, event, hook2 = buffer.low_latency_combine(
    expert_out, topk_idx, topk_weights, handle,
    return_recv_hook=True,
)
do_independent_compute()
hook2()  # launch combine RECV：等 flag、加权 reduce
```

无 hook 时：`async_finish=True` + `event.current_stream_wait()`，或默认同流等 SEND|RECV 一次做完。

---

## 8. 自测题与答案

1. **`return_recv_hook=True` 第一次 launch 的 phases 是什么？**
   答：只有 `LEGACY_LOW_LATENCY_SEND_PHASE`（=1）。

2. **hook() 内部做了什么？**
   答：再次调用同一 launcher，phases=`RECV_PHASE`（=2）。

3. **「0 SM」指哪一段？**
   答：SEND 发完到 hook 之前，NIC DMA 传数、GPU 不做等收；不是整次通信 0 SM。

4. **跨节点 SEND 用什么原语才能后台飞？**
   答：`nvshmemi_ibgda_put_nbi_warp`（非阻塞 post）。

5. **为何不能同时开 `async_finish` 和 `return_recv_hook`？**
   答：host 断言互斥；hook 模式用 compute stream 自己安排收齐时机。

## 9. 学习进度

- [x] LL dispatch / combine / 打包（12–14）
- [x] 跨机 normal 与深挖（16–17）
- [x] LL recv hook 与「0 SM」重叠语义
- [ ] V2 ElasticBuffer / hybrid（可选下一课）

### 下一知识点

DeepEP V2：`ElasticBuffer`、scaleup/scaleout、`dispatch.cuh` vs `hybrid_dispatch.cuh`；或回头补 `SourceMeta` 位图。
