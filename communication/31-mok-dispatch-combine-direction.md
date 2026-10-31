# MoK 的 dispatch / combine 方向：代码里怎么一眼看出来

> [02-mok-scheduling-and-buffers.md](./02-mok-scheduling-and-buffers.md) §4 从**动机**角度讲了为什么 dispatch 用 pull、combine 用 push（schedule 所有权、就绪信号扇出、双向链路利用率）。这篇是它的**代码侧对照**：怎么从两个 kernel 的签名、四条 TMA 语句、以及一个 buffer 维度上，直接读出谁发起、往哪个方向、为什么不会撞车。
>
> 代码基准：`mixture-of-kittens` 本地仓库。
>
> - `csrc/megakernel/dispatch_combine.cuh`：`dispatch_kernel` / `combine_kernel`
> - `csrc/megakernel/types.cuh`：`x_routed_send_buffer` / `y_routed_recv_buffer` 的形状差异
> - `csrc/megakernel/forward.cuh`：前向的四个调用点与任务数
> - `csrc/megakernel/backward.cuh`：反向如何复用同一对 kernel
> - `csrc/utils.cuh`：`fwd_epilogue` 里的加权求和
> - 复算脚本：[31_mok_dispatch_combine_direction_sim.py](./31_mok_dispatch_combine_direction_sim.py)
>
> 前置：本文默认你已经读过 [30-mok-count-pad-schedule-dataflow.md](./30-mok-count-pad-schedule-dataflow.md)，知道 `schedule_peer_rank[i]` 的主语是"持有专家的那个 rank"。

---

```text
本次讲解位置
章节：communication
小节：MoK dispatch / combine 的数据方向
知识点：签名帧规则、*topk 维度如何消掉写冲突、名字与机制不一致的反向复用
上次：30 笔记 —— count/pad/schedule 三阶段与行语义
下次：dispatch_kernel 的 MXFP8 转置量化与 e8m0 scale 布局
```

## 0. 一句话

**两个 kernel 都由专家端发起，token owner 从头到尾被动。dispatch 是拉（peer → local），combine 是推（local → peer）。** 差别只在第一个参数指向谁、TMA 的 load 和 store 哪个落在 `peer_buf` 上。

---

## 1. 帧规则：两个签名同构

```cpp
// dispatch_combine.cuh:2-4
static __device__ __forceinline__ void dispatch_kernel(
    const activation_bf16_pgl &peer_buf,       // ← 第 1 个：对端（pgl = 每 rank 一个裸指针）
    const routed_activation_gl &x_gmem,        // ← 第 2 个：本地

// dispatch_combine.cuh:154-155
static __device__ __forceinline__ void combine_kernel(
    const activation_bf16_pgl &peer_buf,       // ← 第 1 个：对端
    const epi_bf16_gl &local_buf,              // ← 第 2 个：本地
```

**参数位置固定：第 1 个永远是对端，第 2 个永远是本地。** 于是方向可以纯从帧规则推出来：

| kernel | 读哪里 | 写哪里 | 方向 |
|---|---|---|---|
| `dispatch_kernel` | `peer_buf[rank]` | `x_gmem` | **拉** |
| `combine_kernel` | `local_buf` | `peer_buf[rank]` | **推** |

`activation_bf16_pgl` 是 `std::array<bf16*, NUM_DEVICES>`（`types.cuh:71`），**它本身就是"别人的显存地址数组"**。看到这个类型作参数，就知道这个 kernel 会跨卡读写。

> 也正因为签名同构，反向才能用同一对函数换名字复用（见 §6）。

---

## 2. 四条 TMA 语句的实际方向

dispatch 一侧（`dispatch_combine.cuh:74-76`、`:109`）：

```cpp
if (peer_rank >= 0)
    tma::load_async(token_chunks[tid],
        &peer_buf[peer_rank][(size_t)(peer_token_idx / row_divisor) * cols + col_block_idx * DISPATCH_Nb],
        chunk_bytes, inputs_arrived);          // ← 从对端读
...
tma::store_async(x_gmem, x_fp8_tiles[out], {row_block, col_block});   // ← 写本地
```

combine 一侧（`dispatch_combine.cuh:225-228`、`:236-238`）：

```cpp
if (peer_rank[stage] >= 0)
    tma::load_async(token_chunks[stage][tid],
        &local_buf.raw_ptr[(size_t)row_idx[stage] * cols + col_block_idx[stage] * COMBINE_Nb],
        chunk_bytes, inputs_arrived[stage]);   // ← 从本地读
...
tma::store_async(&peer_buf[peer_rank[stage]][(size_t)peer_token_idx[stage] * cols + ...],
                 token_chunks[stage][tid], chunk_bytes);   // ← 写对端
```

**四条语句里，`peer_buf` 出现在 load 还是 store，就是方向的全部信息。**

注意 combine 的两处下标细节：

- **读本地**用 `row_idx`：本地 `y_routed` 是槽位行，直接就是行号。
- **写对端**用 `peer_token_idx`，**且不除 topk**。原因见 §4。

---

## 3. 为什么主语一定是专家端：表在它手里

```text
schedule_peer_rank[i]      = 3         "我的第 i 行 ← rank3"
schedule_peer_token_idx[i] = 4812
```

主语是"**我**"，也就是持有专家权重的 rank。同一个三元组 `(i, r, idx)` 有两种用法：

```text
正向（gather）：i 是目的地，从 r 的 idx 读   →  拉
反向（scatter）：i 是源，写到 r 的 idx      →  推
```

两种用法**都由专家端发起**。如果 token owner 想自己干，它需要一张逆表：

```text
"我的 token 601 的 k=4，落在 rank3 的 y_routed 第 4812 行"   ← 它没有
```

这张逆表不是免费的：要么 expert 端反向扫 schedule 建一张，要么 token 端存 `topk` 份 `(rank, row)` 指针。MoK 的选择是**干脆不建**——专家端有正向表，就只让专家端发指令。

这是 02 §4.2 那个"就绪信号扇出"论据的代码根源：**谁持有索引，谁发起。**

---

## 4. push 为什么不需要原子：`*topk` 这一维

这是本文最值得记的一点。

两个对称 buffer 的形状并不对称：

```cpp
// types.cuh:108-109
activation_bf16_pgl x_routed_send_buffer; // (num_local_tokens, H)
activation_bf16_pgl y_routed_recv_buffer; // (num_local_tokens * topk, H)
```

```text
                    行数        行语义
dispatch 的读端      7168       token
combine  的写端     57344       (token, k)
                     ↑ 差 topk 倍
```

于是两边的下标处理也不同：

```cpp
// dispatch：源是 token-major，所以要折回 token
peer_token_idx / row_divisor        // row_divisor = topk  (forward.cuh:107)

// combine：目标是 (token,k)-major，原样用
peer_token_idx[stage]
```

**为什么这个维度必须存在**：每个 `(token, k)` 恰好只有一个来源 rank（k 指向某个专家，那个专家只住在一个 rank 上），所以 57,344 个 scatter 目标**两两不交**。

```text
如果 recv buffer 是 [7168, H]：
   token 601 的 8 个 k 会写同一个位置
   → 需要 atomicAdd 或串行化或额外归约阶段

实际是 [57344, H]：
   8 个位置互不相同 → 零原子、零锁、零覆盖检测
```

**用 8 倍的空间换掉了原子，并把加权求和推迟到本地最后一步。** 而那个"最后一步"恰好是全层唯一需要 token owner 出面的时候：

```cpp
// utils.cuh:85-90  fwd_epilogue
compute_group::load(accumulator, stage_vecs[0]);        // y_shared 那份（tid==0 载入）
for (int k = 0; k < topk; ++k) {
    compute_group::load(term, stage_vecs[1 + k]);       // 8 份 routed（tid=1..topk 载入）
    compute_group::mul(term, term, weights[stage * topk + k]);
    compute_group::add(accumulator, accumulator, term);
}
```

顺带一个佐证：`d_router_weights` 也是**推**回去的，随 combine 一起走：

```cpp
// dispatch_combine.cuh:243
(*d_router_weight_buffer)[peer_rank[stage]][peer_token_idx[stage]] = d_router_weight;
```

因为它也是按 `(token, k)` 索引的，可以搭同一趟便车。**push 的复用性来自"目标的自然 owner 是 token 端"。**

---

## 5. 读可复制，写需排他

方向选择还有一个更底层的非对称性：

```text
dispatch：8 个专家 rank 可能同时读 rank5 的 x_buffer 同一行
          → 合法。读是非破坏性的、可复制的

combine ：8 个专家 rank 同时写 rank5 的 combine_buffer（不同地址）
          → 合法，但必须地址不交；一旦同址就需要排他
```

**pull 天然支持 fan-out（一个源被多方读），push 天然要求 fan-in 且地址唯一。**

还有个具体后果：dispatch 是**复制式 gather**。一个 token 的 topk 中若有 3 个落在同一 rank，它的 hidden 会被 DMA **3 次**，读同一源行、写进 3 个不同槽位行。这不是浪费——是槽位语义的必然结果（见 30 笔记 §6 的行语义表）。

如果反过来设计（token 端 push），那么"一个 token 要发 8 份到 8 个不同 rank"就要发 8 次单向写，且每次都要处理目标端容量与 fence——这就是 02 §4.2 列的那五条负担。

---

## 6. 反向：名字和机制不一致

backward 没有新写 kernel，而是把一对函数**换名字调用**：

```cpp
// backward.cuh:132-139
auto reverse_combine = [&](int macrobatch_idx, int task_idx) {
    dispatch_kernel<true>(g.d_y_buffer,          // pgl：对端
                          g.d_y_fp8_routed,      // 本地
                          ...);
};

// backward.cuh:140-146
auto reverse_dispatch = [&](int macrobatch_idx, int task_idx) {
    combine_kernel(g.d_x_routed_buffer,          // pgl：对端
                   g.d_x_routed,                 // 本地
                   ...);
};
```

四个通信操作的对账：

| 语义 | 实际 kernel | 数据 | 方向 |
|---|---|---|---|
| 前向 dispatch | `dispatch_kernel` | `x` | 拉 |
| 前向 combine | `combine_kernel` | `y` | 推 |
| 反向 reverse-combine | **`dispatch_kernel<true>`** | `d_y` | 拉 |
| 反向 reverse-dispatch | **`combine_kernel`** | `d_x` | 推 |

**`reverse_combine` 叫 combine，用的却是 `dispatch_kernel`。** 因为它的语义是"把 token 端的 d_y 拉回专家端算梯度"，机制上是 gather。

> 结论：**名字由语义定，机制由数据方向定。** 读代码时以 `peer_buf` 出现在 load 还是 store 为准，不要被 lambda 名字带偏。

`dispatch_kernel<true>` 的那个 `true` 是 `SCALE_ROWS`——额外把 `router_weights` 一起搬（`:70-71`），因为反向的 reduce 要它。

---

## 7. 任务数不是相等，而是同源

两个方向的 tile 形状不同：

```python
# forward.cuh:92-101
dispatch: (macrobatch_tokens / DISPATCH_Mb) * (H / DISPATCH_Nb)         # 128 × 512
combine : ((macrobatch_tokens / COMBINE_Mb) * (H / COMBINE_Nb) + ...) / COMBINE_PIPE_DEPTH   # 16 × 1024，pipe 7
```

Kimi 形状（`num_tokens = 57344`, H = 7168）：

```text
dispatch : 448 chunks × 14 col blocks  =  6,272 tasks（每块 128×512）
combine  : 3584 tiles  ×  7 col blocks = 25,088 tiles ÷ 7 pipe = 3,584 tasks（每任务 7 tile）
```

**不相等，但同源**：两边都从同一个 `num_tokens`、同一批 macrobatch 推导，所以同一组 `num_comm_sms` 能同时覆盖两者——`forward.cuh:120-128` 那个循环就是靠这一点把 dispatch 和 combine 塞进同一段通信 SM：

```cpp
for (int task_idx = comm_cta_idx; task_idx < max(combine_tasks, dispatch_tasks); task_idx += g.num_comm_sms) {
    if (task_idx < combine_tasks)    combine(macrobatch_idx, task_idx);
    if (task_idx < dispatch_tasks)   dispatch(macrobatch_idx - 1, task_idx);   // 顺手预取下一窗
}
```

`combine` 的 tile 更小（16 行）是为了配合 7 级流水——小 tile 让"读回来 → 立刻写出去"的重叠更细。

---

## 8. 误解清单

| 误解 | 纠正 | 证据 |
|---|---|---|
| "dispatch 由 token 端 push" | 专家端 pull；token 端不发起任何通信 | `:74-76` 读 `peer_buf`，`:109` 写 `x_gmem` |
| "combine 是 token 端拉" | 专家端 push | `:225-228` 读 `local_buf`，`:236-238` 写 `peer_buf` |
| "combine 也拉数据（因为用了 load_async）" | 它 load 的是**本地** `y_routed`，load 指令不等于远程读 | `:226` 的 `local_buf.raw_ptr` |
| "combine 的 scatter 需要原子" | 目标 buffer 有 `*topk` 维，地址两两不交 | `types.cuh:109` |
| "recv buffer 和 send buffer 形状应该对称" | 差 topk 倍，这正是消掉写冲突的手段 | 同上 |
| "reverse_combine 用的是 combine_kernel" | 用的是 `dispatch_kernel<true>` | `backward.cuh:132` |
| "两个方向任务数应该相等" | 同源不等：6272 vs 3584 | §7 |
| "token 端在 combine 阶段仍要参与" | 只在最后 `fwd_epilogue` 本地加权 | `utils.cuh:85-90` |

---

## 9. 一页速记

```text
帧规则（一眼判方向）：
  第 1 参数 = peer_buf（对端），第 2 参数 = 本地
  peer_buf 出现在 load  → 拉（dispatch_kernel）
  peer_buf 出现在 store → 推（combine_kernel）

四个通信操作：
  前向 dispatch        dispatch_kernel        远端 x   → 本地  fp8    拉
  前向 combine         combine_kernel         本地 y   → 远端  bf16   推
  反向 reverse-combine dispatch_kernel<true>  远端 d_y → 本地          拉
  反向 reverse-dispatch combine_kernel        本地 d_x → 远端          推

主语：永远是专家端（因为 schedule 表在它手里，token 端没有逆表）

无冲突的两个前提：
  ① 读可复制：一个 token 被多个专家 rank 读同一行，无冲突
  ② 写要排他：recv buffer 是 (token, k) 排（*topk），每个 (token,k) 唯一来源

token owner 只在最后一处出现：
  output[t] = y_shared[t] + Σ_k w[t][k] · combine_buffer[t·topk + k]   (utils.cuh:85-90)
```

### 源码阅读顺序

```text
1. csrc/megakernel/dispatch_combine.cuh:2-30    两个签名并排看（本节帧规则）
2. csrc/megakernel/types.cuh:105-110            send vs recv buffer 的形状差
3. csrc/megakernel/dispatch_combine.cuh:44-80   dispatch 的 load / store
4. csrc/megakernel/dispatch_combine.cuh:180-246 combine 的 load（本地）/ store（对端）
5. csrc/megakernel/forward.cuh:92-128           两个 lambda 与交错循环
6. csrc/megakernel/backward.cuh:132-146         换名字复用（名字 vs 机制）
7. csrc/utils.cuh:43-91                         fwd_epilogue 的加权求和
```
