# DeepEP V2 Hybrid：Scaleout + Scaleup 完整数据流

> 承接 [20](./20-deepep-v2-buffer-coordinates-and-handle-arrays.md)（直连坐标）、[23](./23-deepep-rank-major-vs-expert-major-dataflow.md)（rank/expert-major 对照）。直连路径读完后，跨机走 `hybrid_dispatch.cuh` / `hybrid_combine.cuh`：在 scaleup（节点内）外包一层 scaleout（跨组 / Rail）。本文把 **Notify → Scaleout → Forward → Epilogue → GEMM → Scaleup combine → Forward combine → reduce epilogue** 串成一张可背诵的完整流。
>
> 源码：`deep_ep/include/deep_ep/impls/{hybrid_dispatch,hybrid_combine,dispatch_copy_epilogue}.cuh`，分叉在 `csrc/kernels/elastic/{dispatch,combine}.hpp`（`num_scaleout_ranks > 1`）。

```text
本次讲解位置
章节：communication / DeepEP V2 Hybrid
小节：Scaleout + Scaleup 端到端数据流
知识点：两级 staging、三张钥匙表、dispatch/combine 极性对偶
上次：rank-major vs expert-major 数据流对照卡（23）
下次：engram_fetch 旁路（非 EP 主干）
原语：ncclTeamTagRail / Lsa、channel_linked_list、token_metadata_at_forward、g
```

## 为什么现在讲这个

直连（`num_scaleout_ranks == 1`）只有一层 all-to-all：`dispatch.cuh` 直接写 `[src][slot]`，`combine.cuh` 直接写回 `[贡献者][t]`。
Hybrid 在中间插入 **Scaleout（跨组）+ Forward（组内拆/聚）**，并多出 `token_metadata_at_forward`、`channel_linked_list`。若只记「多了一跳 RDMA」，会分不清：

- epilogue 读的是哪块 buffer（仍是本节点 `scaleup_buffer`）
- combine 时 linked list 和 forward metadata 各服务哪一档 warp
- 去程 Scaleout/Forward 与回程 Scaleup/Forward 如何极性对偶

没有这张完整流，读 `hybrid_combine.cuh` 时很容易把两档 warp 的职责弄反。

---

## 1. 拓扑与符号

```text
全局 rank = scaleout_rank × num_scaleup + scaleup_rank
E_per_scaleout = num_experts / num_scaleout
E_per_rank     = num_experts / (num_scaleout × num_scaleup)

expert e →
  dst_scaleout = e / E_per_scaleout
  dst_scaleup  = (e % E_per_scaleout) / E_per_rank

g = src_rank × MaxTok + t
  t        = g % MaxTok
  src_rank = g / MaxTok
  src_scaleout = src_rank / num_scaleup
```

JIT 分叉（`dispatch.hpp` / `combine.hpp`）：

```text
num_scaleout_ranks == 1 → dispatch.cuh / combine.cuh（直连）
num_scaleout_ranks >  1 → hybrid_dispatch.cuh / hybrid_combine.cuh
epilogue / reduce_epilogue 两边共用（读的 staging 视图不同）
```

---

## 2. 完整数据流（Scaleout + Scaleup 一起）

```text
════════════════════════════════════════════════════════════
配置：全局 rank = scaleout × scaleup + scaleup
      expert → dst_scaleout = e / E_per_scaleout
               dst_scaleup  = (e % E_per_scaleout) / E_per_rank
════════════════════════════════════════════════════════════

【0. Notify（只在 dispatch）】
源卡扫 topk → 本地 count(rank/expert)
  → Rail：把「发往各 scaleout 组」的 count 推给各 scaleout peer
  → 各 scaleout 节点：recv_and_reduce 汇总所有 scaleout 的 count
  → LSA：写到本节点各 scaleup peer
  → 本卡 psum：psum_per_scaleup_rank / psum_per_expert
     （只描述最终 recv_x 在本节点怎么排）

════════════════════════════════════════════════════════════
【1. hybrid_dispatch · Scaleout】跨组发 token
════════════════════════════════════════════════════════════
源卡 token t（channel 切谁发）
  │ topk → 按 dst_scaleout 去重，channel 内累加 slot
  │ 打包：hidden + SF + topk + g(=src_rank*MaxTok+t)
  ▼
若 dst_scaleout == 本节点：
  直接写入 本节点 scaleout_recv[本scaleout][ch][slot]
否则：
  TMA → scaleout_send[t]
  gin.put(Rail) → 对端 scaleout_recv[src_scaleout][ch][slot]
  │
  └─ 周期性 update_scaleout_tail：(finish, tail) → 各 scaleout peer
     （告诉对端 Forward：这条 channel 可以捞到哪了）

════════════════════════════════════════════════════════════
【2. hybrid_dispatch · Forward】组内拆进 scaleup_buffer
════════════════════════════════════════════════════════════
本节点 Forward(ch)：
  round-robin 等各 src_scaleout 的 signaled_tail
  → 从 scaleout_recv[src_so][ch][slot] 捞 token
  → topk 减掉本 scaleout 的 expert 基址 → dst_scaleup
  → 按 scaleup 去重，atomic 抢 slot
  → LSA/TMA 写入 目的 scaleup 的
        scaleup_buffer[src_scaleup_视角][slot]   ≈ 直连的 [src][slot]
  │
  ├─ 记 token_metadata_at_forward[ch][j] =
  │     [ g, is_last_in_chunk, scaleup×K, slot×K ]
  ├─ token 内带 linked_list_idx（链上第几个）
  └─ dst_buffer_slot_idx[ch][src_so][slot][k] = scaleup slot（cached）

scaleup barrier → Trigger epilogue

════════════════════════════════════════════════════════════
【3. dispatch_copy_epilogue】（共用，只读 scaleup_buffer）
════════════════════════════════════════════════════════════
按 psum_per_scaleup_rank 扫：
  scaleup_buffer[src_su][段内slot] → recv_x[i]（或 expand 行）
  │
  ├─ channel_linked_list[linked_list_idx] = i
  │     （链节点 → 接收行号；尾 = -1）
  └─ recv_src_metadata[i] =
        [ g, 段内slot*K+master,  expand行×K ]

（不读 token_metadata_at_forward；那张表留给 combine）

════════════════════════════════════════════════════════════
【4. 用户】
════════════════════════════════════════════════════════════
recv_x / expand → grouped GEMM → y   （与 recv 同行序）

════════════════════════════════════════════════════════════
【5. hybrid_combine · Scaleup】组内寄回（对偶 dispatch Forward）
════════════════════════════════════════════════════════════
专家卡 Scaleup(ch)：
  沿 channel_linked_list[ch][*][dst_su] 依次取 token_idx = i
  → 读 y[i]（+ expand 三分支，同直连）
  → 用 recv_src_metadata[i] 算写址：
        g → t / src_scaleout
        metadata[1] → 段内 slot 或 master lane
  → LSA/TMA 写入 目的 scaleup 的 scaleup_buffer[贡献者条带][…]
  → 更新 channel_scaleup_tail   （告诉本节点 Forward：可捞）

════════════════════════════════════════════════════════════
【6. hybrid_combine · Forward】跨组寄回源（对偶 dispatch Scaleout）
════════════════════════════════════════════════════════════
本节点 Forward(ch)：
  for j = 0,1,… until metadata[j][0] == -1:   # 重放去程顺序
    读 token_metadata_at_forward[ch][j]
      = [ g, is_last, scaleup×K, slot×K ]
    → 等所需 scaleup 的 channel_scaleup_tail
    → 从 scaleup_buffer 取各贡献（允许多次 reduce 则本节点先加）
    → 目标：源卡的 scaleout_recv[贡献者][t]
         本 scaleout：本地 bypass 写入
         跨 scaleout：scaleout_send → gin.put(Rail) → 源 scaleout_recv
    → is_last 控制 RDMA aggregate 打断

全部重放完：
  Rail 互发 finish 信令并互等、清 tail
  （内核末无全局 barrier）

════════════════════════════════════════════════════════════
【7. combine_reduce_epilogue】
════════════════════════════════════════════════════════════
源卡读 scaleout_recv[贡献者][t]
  → 用自己的 topk 推出要收哪几条
  → reduce → combined_x[t]

════════════════════════════════════════════════════════════
缓冲与极性（记这张表就够）
════════════════════════════════════════════════════════════

          Dispatch 方向                Combine 方向
Scaleout  源 → 各 scaleout_recv        各节点 → 源 scaleout_recv
Forward   scaleout_recv → scaleup_buf  scaleup_buf → scaleout_recv
Epilogue  scaleup_buf → recv_x / y     scaleout_recv → combined_x

钥匙：
  linked_list              ：Scaleup combine 找 y[i]
  recv_src_metadata        ：Scaleup combine 写 scaleup_buffer 哪格
  token_metadata_at_forward：Forward combine 重放顺序 + 从哪格取
  g = src_rank*MaxTok + t  ：全程身份
```

---

## 3. 三张钥匙表

| 表 | 形状（概念） | 谁写 | 谁读 | 干什么 |
|---|---|---|---|---|
| `dst_buffer_slot_idx` | `[ch][src_so][slot][k]` | dispatch Forward | cached dispatch | 复用 scaleup slot |
| `token_metadata_at_forward` | `[ch][j][2+2K]` | dispatch Forward | **combine Forward** | 重放去程顺序；`[0]=g`，`[1]=chunk末`，后半 scaleup/slot |
| `channel_linked_list` | `[ch][节点][scaleup]` | **epilogue** 填值；dispatch Forward 写链下标进 token | **combine Scaleup** | 链节点 → recv 行号 `i`，尾 `-1` |

`recv_src_metadata[i]`（epilogue 产出，直连也有）：

| 下标 | 直连 | Hybrid |
|---|---|---|
| `[0]` | `g` | `g` |
| `[1]` | `src_rank*K + master` | **`段内 slot*K + master`** |
| `[2+k]` | expand 展开行 | 同左 |

Hybrid 的 `[1]` 存段内 slot，是为了 combine Scaleup 在本节点 `scaleup_buffer` 上寻址；跨组身份靠 `g` 与 forward metadata。

---

## 4. Warp 角色对偶

| 阶段 | Dispatch | Combine |
|---|---|---|
| 跨组 | **Scaleout warp**：发 token + scaleout_tail | **Forward warp**：按 forward metadata 寄回源 scaleout |
| 组内 | **Forward warp**：捞进 scaleup_buffer + 记 metadata/链下标 | **Scaleup warp**：按 linked list 取 `y[i]` 写 scaleup_buffer |
| 收尾 | epilogue → `recv_x` | reduce_epilogue → `combined_x` |
| Notify | 有（两级 count） | 无 |

---

## 5. 具体数字走读

设定：`num_scaleout=2`，`num_scaleup=4` → 8 卡；每卡 2 expert → 16 expert；`MaxTok=8`；topk=2。
源在 scaleout0 / scaleup1 → `src_rank = 0*4+1 = 1`，token `t=5`，`topk=[3, 12]`。

```text
expert 3  → scaleout 0, scaleup 1  （本节点）
expert 12 → scaleout 1, scaleup 2  （跨组）

g = 1*8 + 5 = 13
```

**Dispatch**

1. Scaleout：对本节点 scaleout0 写本地 `scaleout_recv`；对 scaleout1 `put` 到对端 `scaleout_recv`。
2. Forward（两端各自）：把 token 拆进目标 scaleup 的 `scaleup_buffer`；记下 forward metadata / linked_list_idx。
3. Epilogue：专家卡得到 `recv_x` 行，`recv_src_metadata[i][0]=13`，linked list 指向该行。

**Combine**

1. Scaleup：沿 linked list 找到行 `i`，把 `y[i]` 写回各 scaleup 的 `scaleup_buffer`。
2. Forward：按 forward metadata 重放；本节点 reduce 后，一份 bypass 留在 scaleout0，一份 Rail 回源 scaleout0 的 `scaleout_recv[贡献者][5]`。
3. 源卡 reduce_epilogue：读贡献者条带 → `combined_x[5]`。

---

## 6. 可运行小验证：拓扑下标与 `g`

文件：`communication/24_hybrid_index_sim.py`。只验证编号空间（不模拟通信）。

```python
#!/usr/bin/env python3
"""Encode/decode hybrid rank and global token index g."""

from __future__ import annotations


def main() -> None:
    num_scaleout, num_scaleup = 2, 4
    num_experts, max_tok = 16, 8
    e_per_scaleout = num_experts // num_scaleout
    e_per_rank = num_experts // (num_scaleout * num_scaleup)

    def split_expert(e: int):
        so = e // e_per_scaleout
        su = (e % e_per_scaleout) // e_per_rank
        return so, su

    src_so, src_su, t = 0, 1, 5
    src_rank = src_so * num_scaleup + src_su
    g = src_rank * max_tok + t
    assert g // max_tok == src_rank and g % max_tok == t
    assert g // (max_tok * num_scaleup) == src_so

    topk = [3, 12]
    print(f"src_rank={src_rank} (so={src_so}, su={src_su}) t={t} g={g}")
    for e in topk:
        so, su = split_expert(e)
        print(f"  expert {e:2d} -> scaleout={so} scaleup={su}  "
              f"{'local-so' if so == src_so else 'cross-so'}")

    # Round-trip g
    g2 = 13
    print(f"decode g={g2}: rank={g2 // max_tok} t={g2 % max_tok} "
          f"so={g2 // (max_tok * num_scaleup)}")


if __name__ == "__main__":
    main()
```

运行：

```bash
python3 communication/24_hybrid_index_sim.py
```

期望输出：

```text
src_rank=1 (so=0, su=1) t=5 g=13
  expert  3 -> scaleout=0 scaleup=1  local-so
  expert 12 -> scaleout=1 scaleup=2  cross-so
decode g=13: rank=1 t=5 so=0
```

---

## 7. 源码落点（速查）

| 步骤 | 文件与位置 |
|---|---|
| JIT 分叉 | `csrc/kernels/elastic/dispatch.hpp` L53–L78；`combine.hpp` L47–L66 |
| Notify 两级 | `hybrid_dispatch.cuh` L107–L327 |
| Scaleout 发 token | `hybrid_dispatch.cuh` L329–L463 |
| Forward 进 scaleup + metadata | `hybrid_dispatch.cuh` L464–L658 |
| Epilogue + linked list | `dispatch_copy_epilogue.cuh` L49–L228 |
| Combine Scaleup + linked list | `hybrid_combine.cuh` L106–L350 |
| Combine Forward 重放 | `hybrid_combine.cuh` L351–L621 |
| 表形状注释 | `csrc/elastic/buffer.hpp` L887–L948 |

---

## 8. 常见误判

| 误判 | 实际 |
|---|---|
| Epilogue 读 `token_metadata_at_forward` | 否；只读 `scaleup_buffer`；forward metadata 给 combine Forward |
| Linked list 在 dispatch Forward 里填成行号 | Forward 只写入 token 内的链下标；**行号由 epilogue 填** |
| Hybrid 的 `recv_src_metadata[1]` 仍是 src_rank | 是 **段内 slot×K+master** |
| Combine 两档 warp 都扫 linked list | 只有 **Scaleup** 扫 linked list；**Forward** 扫 forward metadata |
| 回程最终在 scaleup_buffer 上 reduce | 源卡最终读的是 **scaleout_recv[贡献者][t]** |

---

## 9. 自测题

1. **Dispatch 的 Scaleout 与 Combine 的哪一档对偶？**
   答：与 Combine 的 **Forward**（都是跨组 Rail；方向相反）。

2. **`channel_linked_list` 谁写谁读？**
   答：Epilogue 写 `= i`；Combine Scaleup 读出 `i` 去取 `y[i]`。

3. **`g=13, MaxTok=8, scaleup=4` 时源 scaleout / t？**
   答：`t=5`，`src_rank=1`，`src_scaleout=0`。

4. **Epilogue 之后用户看到的 `recv_x` 按什么分段？**
   答：本节点 `psum_num_recv_tokens_per_scaleup_rank`（scaleup 源段），不是按全局 rank 段。

5. **直连和 Hybrid 的最终 combine 布局差在哪？**
   答：语义都是「贡献者 × 原 token」；直连贡献者条带在复用的 `buffer` 上，Hybrid 源卡读的是 **scaleout_recv** 视图。

---

## 10. 学习进度

- [x] V2 直连坐标与 handle（20）
- [x] Rank-major / Expert-major 数据流卡（23）
- [x] V2 Hybrid Scaleout+Scaleup 完整数据流
- [x] Engram fetch 旁路（见 [25](./25-deepep-v2-engram-fetch-rdma-get.md)）

### 下一知识点

已写 [25](./25-deepep-v2-engram-fetch-rdma-get.md)；后续可接 PP send/recv。
