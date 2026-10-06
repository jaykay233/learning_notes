# DeepEP V1 Normal 与 Low-latency：完整端到端数据流对照

> 汇总 [04 · Normal 数据流](./04-deepep-v1-normal-dataflow.md) 和 [05 · Low-latency 数据流](./05-deepep-v1-low-latency-dataflow.md)，从 router 输入、布局准备、rank 间发送、接收布局、本地 expert 计算，一直追到 combine 输出。本文讨论仓库 V1 `legacy` API，不要与 V2 接口混淆。
>
> 对照源码：`/Users/saboxu/Downloads/communication/DeepEP`，重点为 `deep_ep/buffers/legacy.py`、`csrc/legacy/buffer.hpp`、`csrc/kernels/legacy/intranode.cu`、`csrc/kernels/legacy/internode.cu`、`csrc/kernels/legacy/internode_ll.cu`、`docs/legacy.md` 和 `tests/legacy/`。

```text
本次讲解位置
章节：communication / DeepEP V1
小节：Normal 与 Low-latency 的端到端 dispatch → expert → combine 对照
知识点：rank-major 与 expert-major、layout 元数据、handle、权重归并与同步边界
上次：low-latency 的 expert-major buffer、handle 和 hook
下次：结合具体 workload 选择模式并估算 buffer/延迟/overlap 取舍
PTX：不适用（本文聚焦 MoE EP 通信 API 和布局）
```

## 为什么现在做完整对照

分别看过两条路径后，最容易混淆的是把“通信目的地”“接收 tensor 的行顺序”“expert 计算分桶”“router 权重归并”当成同一件事。实际上，normal 与 low-latency 的最大差别是**如何组织 token 路由与接收 buffer**；它们的 combine 权重语义、容量协议、布局 metadata 和同步方式也不完全相同。把数据流放在同一张图里，能避免把 normal 的 rank/channel handle 传给 low-latency combine、把 low-latency capacity 当成有效行数，或漏掉 normal 侧的 gate multiply。

## 1. 先给全局图

### Normal（高吞吐路径）

```text
Router on every source rank
  x[N,H] + topk_idx[N,K] + topk_weights[N,K]
              │
              ├─ get_dispatch_layout(topk_idx)
              │    ├─ num_tokens_per_rank[P]
              │    ├─ num_tokens_per_rdma_rank[...]
              │    ├─ num_tokens_per_expert[E]
              │    └─ is_token_in_rank[N,P]
              ▼
  dispatch：按目的 rank/channel 安排通信；同一 token 发往同一 rank 通常只需一份
              ▼
  recv_x[M,H] + recv_topk_idx[M,K] + recv_topk_weights[M,K]
  + per-local-expert counts + normal handle
              │
              ├─ 根据 recv_topk_idx 组织/路由 local expert 计算
              ├─ 按模型的 MoE 公式处理 gate 权重
              ▼
  combine(expert_outputs, handle)：逆向通信 + 对来源 token 的贡献求和
              ▼
  combined_x[N,H]
```

### Low-latency（低延迟路径）

```text
Router on every source rank
  x[N,H] + topk_idx[N,K]
  topk_weights[N,K] 留到 combine 阶段使用
              │
              ├─ 不需要先调用 normal get_dispatch_layout
              └─ 调用方给 capacity=C（所有 rank 相同，N<C）
              ▼
  low_latency_dispatch：内部计算路由，写入目标 rank 的 local-expert 分区
              ▼
  recv_x[L,C*P,H] (+ optional FP8 scales)
  + recv_count[L] + low-latency handle
              │
              ├─ 每个 local expert 处理自己的有效输入
              └─ expert 输出保留与接收 buffer 对应的 expert/slot 布局
              ▼
  low_latency_combine(expert_out, topk_idx, topk_weights, handle)
  按 topk_weights 加权 + 逆向通信 + reduce
              ▼
  combined_x[N,H]
```

两条路径都把专家放在 EP ranks 上，并都由通信 API 搬运 token/expert 结果；区别不是“normal 才有 all-to-all、low-latency 没有”，而是路由粒度、接收布局、容量与 combine 协议不同。

## 2. 统一符号与 shape

| 符号 | 含义 |
|---|---|
| `P` | EP group 中的 rank 数 |
| `E` | 全局 expert 数 |
| `L = E/P` | 均匀切分时每 rank 的 local expert 数 |
| `N` | 当前 source rank 的输入 token 数 |
| `H` | hidden size |
| `K` | 每 token 的 top-k 路由槽位数 |
| `C` | low-latency 的 `num_max_dispatch_tokens_per_rank` |
| `M_r` | normal 下某目标 rank 收到的去重 token 行数 |

Router 输出通常是：

```text
x             [N,H]       BF16，或 normal 路径支持的 FP8 payload + scales
topk_idx      [N,K]       全局 expert ID，-1 表示该槽位无路由
topk_weights  [N,K]       与 topk_idx 同槽位对齐的 gate 权重
```

expert ID 常按 rank 均匀连续切分时：

```text
owner_rank(global_expert) = global_expert // L
local_expert_id           = global_expert % L
```
具体 expert 到 rank 的映射由模型/EP group 约定；以上公式用于均匀连续分片的直观例子。

## 3. Normal：layout → rank/channel dispatch → expert 计算 → combine

### 3.1 布局准备

标准新路由调用先执行：

```python
num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, \
    is_token_in_rank, layout_event = buffer.get_dispatch_layout(topk_idx, num_experts)
```

- `num_tokens_per_rank[P]`：当前 source rank 发给每个 destination rank 的**去重 token 行数**，对应 all-to-all 视角下的发送 splits。
- `num_tokens_per_rdma_rank[...]`：多节点下相同 GPU/NVLink index 的 RDMA peer 聚合计数；单节点场景为 `None`。
- `num_tokens_per_expert[E]`：全局 expert 路由项计数，不是每 rank 的去重 token 数。
- `is_token_in_rank[N,P]`：token 是否至少路由到该目标 rank。

因此，如果一个 token 的两个 top-k experts 在同一 rank，`num_tokens_per_rank` 对该 token-rank 只计一次，但 `num_tokens_per_expert` 会在两个 expert 上各计一次。

### 3.2 请求体和 rank 粒度的数据搬运

逻辑上，normal dispatch 的输入包括 activation `x`、路由索引/权重，以及前一步算出的 layout 数组。`dispatch` 按 token→destination rank 做发送计划；多节点场景使用 V1 normal 的 NVLink/RDMA forwarding 组织。接收 tensor 的行按通信 rank/channel 布局组织，**不是按 expert 排好的一组连续 bucket**。

Dispatch 返回的关键对象：

```text
recv_x                    [M,H] 或 FP8 payload/scales
recv_topk_idx             [M,K]，目标 rank 上的 local expert IDs；不属于该 rank 的槽位为 -1
recv_topk_weights         [M,K]，与接收的路由槽位对应
num_recv_tokens_per_expert_list [L]，expert 计数（可能按 expert_alignment 对齐）
handle                    normal combine/反向 dispatch 使用的布局上下文
```

**如何知道 normal `recv_x` 第 `m` 行要给哪个专家？看同一行的 `recv_topk_idx[m,:]`。** 它不是“这一行唯一所属的 expert”。如果两个所选 experts 都在当前 rank，索引中会有两个 local expert ID；rank-major 接收行可供多个专家复用/展开，后续 MoE/GEMM 路径据此组织计算。计数列表只报每个 expert 的数量，不提供逐行对应关系。

`handle` 是 opaque 元数据。单节点实现包含 rank/channel prefix、`recv_src_idx`、`is_token_in_rank`、`send_head` 等；多节点 handle 结构不同，还涉及 RDMA/NVLink 层次元数据。不要假设它是跨路径统一的 tuple，也不需要业务代码手工解析。要定位一行逻辑身份，概念上需要 source rank + source-local token index；normal 单节点的 `recv_src_idx` 本身只给 token index，source-rank 段由 rank prefix 布局提供。channel 是物理传输分段，不是 token 的语义身份。

### 3.3 Normal combine 的权重边界

专家计算产出的 `expert_outputs` 行序必须对应 dispatch 的 `recv_x` 行。`buffer.combine(expert_outputs, handle)` 做反向通信，把来自各目标 rank 的结果送回源 rank并归并；它**不会自动把 `expert_outputs` 乘上 top-k gate 权重**。可选 `topk_weights=` 参数是对权重张量本身做归并，不等于把权重乘入 `x`。

若目标 MoE 结果应为：

```text
y_i = Σ_k w[i,k] * Expert_{topk_idx[i,k]}(x_i)
```
则模型侧需确保送入 normal combine 的每条 expert 贡献已经符合这条 gate 公式（或在其计算路径中完成同等权重处理）；combine 再把各 rank 的贡献求和。输出首维恢复为源 rank 原始 `N`：`combined_x[N,H]`。

## 4. Low-latency：内部路由 → expert-major 固定容量区 → weighted combine

### 4.1 无独立 layout API，容量由调用方给定

```python
recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
    x, topk_idx,
    num_max_dispatch_tokens_per_rank=C,
    num_experts=E,
    use_fp8=True,
    async_finish=False,
    return_recv_hook=False,
)
```

Low-latency dispatch 自己读取 `topk_idx` 并准备路由/接收布局，不要求先调用 normal 的 `get_dispatch_layout`。调用方提供所有 rank 共用的 `C`；当前实际 dispatch token 数必须小于 `C`。`C` 是容量边界，不是收到 token 数。

### 4.2 Expert-major 接收 buffer

令 `L=E/P`。返回的 payload：

```text
use_fp8=False: recv_x [L, C*P, H] BF16
use_fp8=True:  (fp8_data [L,C*P,H], scales [L,C*P,H/128] 常见格式)
recv_count:    [L] int32
```

第一维直接区分当前 rank 的 local experts；第二维为跨 source ranks 的容量槽位。只有部分槽位有效：`recv_count[l]` 是 expert `l` 收到的有效 token 总数；handle 的 per-source layout range 标出每个 source rank 在该 expert buffer 中的区间。不能对全部 `C*P` 行无条件执行 expert GEMM。

这一路径接收布局是 expert-major，且每个路由 expert assignment 都进入对应专家分区。同一个 token 若选中同一 destination rank 上的两个 local experts，数据会出现在两个 expert 输入区；normal 则通常在该 rank 收一行 token，再用一行中的多个 `recv_topk_idx` 槽位描述多个 local experts。

### 4.3 Low-latency handle 和逆向映射

当前 V1 Python 封装返回的 handle 为：

```text
(
  packed_recv_src_info,        # [L, P*C] int32；每槽的 source-local token index
  packed_recv_layout_range,    # [L, P] int64；每个 local expert × source rank 的范围
  C,
  H,
  E,
)
```

对 `packed_recv_layout_range[l,r]`，当前实现编码为高 32-bit begin、低 32-bit count：

```python
packed = packed_recv_layout_range[l, r].item()
begin = packed >> 32
count = packed & 0xFFFFFFFF
# recv_x[l, begin:begin + count] 是来自 source rank r 的有效区间
# packed_recv_src_info[l, begin:begin + count] 是对应源 rank 的本地 token 下标
```

rank 由 layout_range 的第二维索引 `r` 得到；source token index 由 `src_info` 得到。后三个标量 `C,H,E` 是 combine 重建参数。handle 不包含 activation、topk_idx 或 topk_weights；不要修改它，直接原样传给 `low_latency_combine`。

### 4.4 Expert compute 与 Low-latency combine

专家侧按照 `recv_count` 和上述 packed layout 对有效 token 执行 GEMM/MLP，expert 输出保持 `[L,C*P,H]` 的专家/槽位对应关系。然后：

```python
combined_x, event, hook = buffer.low_latency_combine(
    expert_out, topk_idx, topk_weights, handle,
    async_finish=False,
    return_recv_hook=False,
)
```

`low_latency_combine` 使用 handle 找回 source rank/token，并按 `topk_weights` 加权归并，输出 `[N,H]`。因此在这一点上它和 normal combine 不同：**Low-latency combine 明确做 weighted reduce；normal combine 不会自动乘 gate。**

## 5. 两种路径的同一组例子

设：

```text
P=4, E=16, L=4, N=64, H=4096, K=2, low-latency capacity C=128
rank 1 管理 global experts 4–7（local IDs 0–3）
```

某个 token `t` 选择 global experts `[5,6]`，两者都在 rank 1：

```text
Normal:
  rank 1 收到一份 x[t]
  对应 recv_topk_idx 行为 [1,2]
  后续本地 expert 1 和 expert 2 都计算该 token 的贡献

Low-latency:
  recv_x[1] 的有效区域有一份 x[t]
  recv_x[2] 的有效区域也有一份 x[t]
  recv_count[1]、recv_count[2] 各自计入该路由
```

这时 normal 的去重 token-rank 计数增加 1；expert assignment 计数增加 2。Low-latency 的两个 local expert 接收区各增加一个有效路由项。

如果 token 选择 `[5,10]`，global expert 10 在 rank 2：两条路径都要把 activation 发到 rank 1 和 rank 2。Normal 每个 destination rank 各收到一份 rank-level token row；low-latency 分别写进 rank 1 上 expert 1 与 rank 2 上对应 expert 的 packed 区。两种路径 combine 最终都把贡献归回源 rank 上 token `t` 的位置，只是 normal gate 在 MoE 路径中处理，low-latency combine 用传入的权重处理。

该例的张量容量为：

```text
normal recv_x: [M_r, 4096]，M_r 是该 rank 收到的去重 token 行数
normal recv_topk_idx: [M_r, 2]
low-latency recv_x: [4, 128*4, 4096] = [4,512,4096]
low-latency recv_count: [4]，逐 expert 的实际有效数，不是 [512,512,512,512]
最终两者 combined_x: [64,4096]
```

## 6. 请求内容与 API 返回的对照表

| 阶段 | Normal | Low-latency |
|---|---|---|
| Dispatch 前 | 通常 `get_dispatch_layout(topk_idx,E)`；产生 rank/expert counts 和 token-rank mask | 不调用 normal layout API；调用时传 `x, topk_idx, C, E` |
| Dispatch 的路由粒度 | token → destination rank；同行可描述多个本 rank experts | token-expert assignment → destination local-expert buffer |
| Dispatch 携带/返回的路由数据 | dispatch 输入 `topk_idx`、`topk_weights`；输出 local `recv_topk_idx`、`recv_topk_weights` | dispatch 输入 `topk_idx`；topk weights 在 combine 阶段另传 |
| 接收 activation | `[M_r,H]` rank/channel 布局 | `[L,C*P,H]` expert-major 固定容量布局 |
| 实际有效计数 | `recv_x.shape[0]` 是接收行数；expert counts 另给 | `recv_count[L]`；容量中可能有无效槽位 |
| 回程元数据 | normal handle；intranode/internode 结构不同 | `(src_info, layout_range, C, H, E)` |
| Combine 权重 | combine 归并 x，不自动乘 top-k gate | low-latency combine 用 gate 权重加权并归并 |
| 同步/重叠 | `EventOverlap`、`previous_event`、stream 依赖 | event + 可选 receive hook；消费数据前需调用 hook |

## 7. 两条路径的同步边界

### Normal

`get_dispatch_layout(..., async_finish=True)` 返回的 event 可通过 `previous_event` 串给 `dispatch`。`dispatch/combine` 的 `async_finish=True` 表示当前 stream 不等待通信 kernel 完成；下游消费数据前必须建立正确的 `EventOverlap`/stream 依赖。不要把“函数返回”理解为异步数据已 ready。

### Low-latency

设置 `return_recv_hook=True` 时 kernel 可先 issue 接收请求，再由返回 hook 确认数据到达；必须在计算读取 dispatch 输出前调用 hook，combine 输出也同样处理。该机制让网络在后台传输且计算阶段不占用 SM。V1 只有两个 low-latency buffers，返回 tensor 会复用 buffer，不应同时持有超过两个 low-latency kernel 的结果。

## 8. 传输路径不要和 tensor layout 混为一谈

- Normal 单节点通信主要走 NVLink；多节点路径按 V1 normal 的 NVLink/RDMA forwarding 组织。
- Low-latency API 依赖 IBGDA/RDMA 可见性；V1 同节点 low-latency 也支持 NVLink P2P，但 `allow_nvlink_for_low_latency_mode` 是可配置项（当前 API 默认 `True`）。源码提醒 NVLink traffic 与 hook overlap 可能不兼容，并警告 PCIe 上的 P2P memory ordering 风险。
- “rank-major/expert-major”说的是 tensor 组织布局；“NVLink/RDMA”说的是物理通信路径，两组概念是正交的。

## 9. 选模式前先问的几个问题

| 负载/约束 | 通常优先检查 |
|---|---|
| token 多、训练或 prefill 更在意总体吞吐 | normal：动态接收量、rank/channel 布局、normal config 与 SM 数 |
| decoding 小 batch、单步延迟敏感 | low-latency：capacity、RDMA buffer 用量、hook 与 micro-batch overlap |
| batch 峰值或 token 数变化大 | normal 的动态分布 vs low-latency 的固定容量浪费/溢出边界 |
| 需要按 expert 分桶输入 | normal 下游从 `recv_topk_idx` 组织；low-latency 已提供 expert-major buffer，但要按有效区处理 |
| 需要 gate 加权语义简单明确 | 明确是 normal 在 MoE 路径先乘权重，还是 low-latency combine 负责 weighted reduce |

这不是“normal 一定用于训练、low-latency 一定用于推理”的硬规则；最终要按 batch、hidden、top-k、网络拓扑、延迟/吞吐目标与 buffer 成本评估。

## 10. 常见错误与可观察症状

| 错误 | 会发生什么 |
|---|---|
| 把 normal `recv_x[m]` 当成唯一 expert 的输入 | token 可能有多个 local expert；遗漏/错派专家计算，模型输出缺路由贡献 |
| 把 normal combine 当成自动乘 gate | combine 输出幅度/数值和 MoE reference 不符 |
| 把 low-latency `C*P` 全部当有效行 | 无效槽位被送进 GEMM，带来垃圾结果或不稳定误差 |
| 用 normal `get_dispatch_layout` 给 low-latency dispatch 准备元数据 | 多余或把不匹配的布局概念套到另一套接口 |
| 混用两种 handle 或手工改写 handle | combine 回错 rank/token 或访问不匹配的缓冲布局 |
| 认为 `return_recv_hook=True` 后数据自动到齐 | 在 hook 前消费数据可能读到未完成接收内容 |
| 把 expert-major 当作“网络直接按全局 expert 做 collective” | 混淆本地 buffer 排列和 rank 间的物理传输 |

## 11. 速记卡

```text
Normal:
  get_dispatch_layout(topk_idx)
  -> token-rank 去重统计 + rank/channel dispatch
  -> recv_x[M,H] + recv_topk_idx[M,K] + weights + normal handle
  -> MoE path 按 local IDs 算，并确保 gate 语义正确
  -> combine 逆向传输 + sum（不自动乘 gate）

Low-latency:
  low_latency_dispatch(x, topk_idx, C, E)，不先调 normal layout
  -> recv_x[L,C*P,H] + recv_count[L] + LL handle
  -> local experts 算有效 packed rows
  -> low_latency_combine(..., topk_idx, weights, handle)
  -> weighted reverse combine，输出 [N,H]

永远分开想：
  1. 路由目的地：rank 还是 local expert？
  2. tensor layout：rank/channel 还是 expert-major？
  3. gate 权重：在哪个阶段乘？
  4. readiness：event/stream 还是 hook？
```

## 12. 自测题与答案

1. 一个 token 的两个 top-k experts 在同一 rank，normal 的 `num_tokens_per_rank` 增几？low-latency 有几个 expert assignment？
   - **Normal 增 1 个 token-rank 行；low-latency 对两个 local expert 接收区各有一个路由输入。**
2. normal 如何知道 `recv_x` 一行对应哪些 local experts？
   - 看同一行 `recv_topk_idx`；一行可含多个 local expert IDs，`-1` 表示该槽位不在当前 rank。
3. low-latency 的 `C*P` 是有效 token 数吗？
   - **不是容量就是有效数。** `recv_count` 是每 local expert 的有效总数，handle 的 layout range 给出 source-rank 分段。
4. normal 和 low-latency 的 combine 对 gate 权重有什么区别？
   - Normal combine 不自动乘 gate；low-latency combine 按 `topk_weights` weighted reduce。
5. 两个 API 的 handle 能互换吗？
   - **不能。** normal handle 记 rank/channel 通信布局，low-latency handle 记 expert/source-range 与容量参数；都应视为 opaque metadata。

## 13. 源码映射与验证边界

- Normal 的 layout/dispatch/combine Python 契约：`deep_ep/buffers/legacy.py`、`docs/legacy.md`。
- Normal handle 和 rank/channel 数据布局：`csrc/legacy/buffer.hpp`、`csrc/kernels/legacy/intranode.cu`、`csrc/kernels/legacy/internode.cu`、`tests/legacy/test_intranode.py`、`tests/legacy/test_internode.py`。
- Low-latency API/handle shape：`deep_ep/buffers/legacy.py`、`csrc/legacy/buffer.hpp`。
- Low-latency source info、per-rank ranges、packing 与 weighted combine：`csrc/kernels/legacy/internode_ll.cu`、`tests/legacy/test_low_latency.py`。

本笔记是源码静态梳理；实际性能、NVLink/RDMA 路径选择、hook overlap、CUDA graph 行为与数值正确性仍需在匹配的 GPU/NVSHMEM/IBGDA 环境中运行相应 V1 测试验证。
