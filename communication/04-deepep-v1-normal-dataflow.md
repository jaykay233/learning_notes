# DeepEP V1 Normal：从路由布局到 dispatch / expert / combine

> 承接 [03-low-latency-collectives-and-synchronization.md](./03-low-latency-collectives-and-synchronization.md)。本文沿着 DeepEP V1 `legacy` API，把 normal 模式的数据从 router 输出追到 combine，并区分逻辑路由、rank/channel 通信布局和 expert-major 计算布局。
>
> 代码对照：本地 DeepEP checkout `/Users/saboxu/Downloads/communication/DeepEP`，重点文件是 `deep_ep/buffers/legacy.py`、`csrc/legacy/buffer.hpp`、`csrc/kernels/legacy/intranode.cu`、`csrc/kernels/legacy/internode.cu`。这些是 V1 legacy 实现；不要和仓库当前 V2 API 混读。

```text
本次讲解位置
章节：communication / DeepEP V1
小节：Normal dispatch / combine 完整数据流
知识点：layout、rank/channel 通信元数据、handle 与 combine 的关系
上次：小消息 collective 的同步、Sentinel、credit、fabric reduce
下次：DeepEP V1 low-latency 路径的定容 buffer、打包与 hook
PTX：不适用（本文聚焦 DeepEP API 与 CUDA 通信 kernel）
```

## 为什么现在讲这个

前面的小消息 collective 笔记解释了数据如何跨 GPU 到达、同步和归约；MoE EP 还多了一个决定正确性的映射问题：一个 token 可能送到多个 rank，每个 rank 又有多个 local experts。若把“按 rank 发数据”“按 expert 排列输入”“按 router 权重合并”混成同一步，就容易误以为 `recv_x` 已经 expert-major，或误以为 `combine` 会自动乘 gate。本文用完整数据流把这些边界分开。

## 0. 一句话心智模型

```text
router 给出全局 expert ID 与权重
  -> layout 算每个 token 要去哪些 rank
  -> dispatch 按 rank 传 token，并把 expert ID 转成本地编号
  -> MoE/GEMM 路径按路由运行本地 expert（必要时自行重排）
  -> combine 用 dispatch 的 handle 反向传输并跨 rank 求和
```

**DeepEP normal dispatch 的通信目的地是 rank，不是 expert。**`recv_topk_idx` 告诉目标 rank 上的后续计算：收到的 token 对应哪些 local experts。`recv_x` 的接收行不因此变成 expert-major。

## 1. Normal 和 low-latency 的定位

| 路径 | 主要取向 | 数据/容量风格 | 通信层次概览 |
|---|---|---|---|
| V1 normal（high-throughput） | 较高吞吐，适合较大 token 数和训练/预填充式负载 | 先计算动态 token 分布；dispatch 返回实际接收行数 | 单节点主要走 NVLink；多节点组合节点内 NVLink 与跨节点 RDMA |
| V1 low-latency | 降低小批量/解码式 MoE 的通信延迟 | 调用方给每 rank 的最大 dispatch 容量，使用专门的打包与缓冲 API | 专门的 low-latency 路径；不能当成 normal API 的简单开关 |

本文深入讲 normal。低延迟路径虽同属 V1 `Buffer`，但其 `low_latency_dispatch` / `low_latency_combine` 输入输出和容量约束不同，不要把下文 normal 的 tuple/handle 结构套用到它。

## 2. 张量与维度约定

以下先讨论当前 rank 的本地输入；每个 rank 都有一份自己的输入。

| 符号 | 含义 |
|---|---|
| `P` | EP group 的 rank 数 |
| `E` | 全局 expert 数；常见均匀切分下每 rank 持有 `E/P` 个 expert |
| `N` | 当前 source rank 原始输入 token 数 |
| `H` | hidden size |
| `K` | 每 token 的 top-k 路由数 |
| `M_r` | 目标 rank `r` dispatch 后收到的 token 行数 |
| `C` | 每次 normal 通信配置使用的 channel 数 |

典型输入：

```text
x             [N, H]       BF16，或 FP8 payload + scales
 topk_idx     [N, K]       全局 expert ID；-1 表示无路由
 topk_weights [N, K]       与 topk_idx 槽位对齐的 router 权重
```

`topk_idx[i, j]` 和 `topk_weights[i, j]` 描述同一个 token `i` 的第 `j` 个路由槽位。

## 3. 第一步：`get_dispatch_layout` 算什么

```python
num_tokens_per_rank, \
num_tokens_per_rdma_rank, \
num_tokens_per_expert, \
is_token_in_rank, \
layout_event = buffer.get_dispatch_layout(topk_idx, num_experts)
```

| 返回值 | Shape / 类型 | 用途 |
|---|---|---|
| `num_tokens_per_rank` | `[P]`, int32 | 当前 rank 要向每个目标 rank 发送多少行 token；是 token-rank 的去重计数 |
| `num_tokens_per_rdma_rank` | `[num_rdma_ranks]` 或单节点为 `None` | 多节点时，同 GPU index 的 RDMA peer 聚合计数 |
| `num_tokens_per_expert` | `[E]`, int32 | 每个全局 expert 的 token 路由计数 |
| `is_token_in_rank` | `[N, P]`, bool | token `i` 是否需要发送到目标 rank `r` |
| `layout_event` | EventOverlap | 异步布局计算的依赖事件 |

当一个 token 的两个 top-k expert 都在同一 rank，该 rank 通常只接收一份 `x`，但这行的 top-k 槽位可以有两个 local expert ID。因此 `num_tokens_per_rank` 和 `num_tokens_per_expert` 统计的对象不同：前者是 token 发往 rank 的次数，后者是 token-expert 路由数。

布局 API 文档位于本地 checkout 的 `deep_ep/buffers/legacy.py` `get_dispatch_layout` 定义处（约 L293）。

## 4. 第二步：dispatch 的“请求体”逻辑上带什么

调用形式：

```python
recv_x, recv_topk_idx, recv_topk_weights, \
recv_tokens_per_expert, handle, dispatch_event = buffer.dispatch(
    x,
    topk_idx=topk_idx,
    topk_weights=topk_weights,
    num_tokens_per_rank=num_tokens_per_rank,
    num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
    is_token_in_rank=is_token_in_rank,
    num_tokens_per_expert=num_tokens_per_expert,
    previous_event=layout_event,
    async_finish=True,
)
```

这里的“请求体”只是概念上的 payload，不是 DeepEP 定义的网络 JSON 包。通信逻辑上要送的是：

1. token hidden state `x`（FP8 输入还带 scale）；
2. 与 token 对齐的 top-k expert ID；
3. 与 top-k 槽位对齐的权重；
4. DeepEP 内部用来组织 rank/channel 通信的计数、offset、buffer slot 和来源路由元数据。

布局张量用于决定往哪里发；它们不是每行 token 的 expert 输出。

### 4.1 按目标 rank 复制并本地化 top-k

假设 rank 0 持有全局 experts `[0, 1]`，rank 1 持有 `[2, 3, 4]`，某 token 的路由是：

```text
global topk_idx     = [1, 4]
topk_weights        = [0.3, 0.7]
```

这个 token 会去 rank 0 和 rank 1。两个 rank 收到的路由槽位分别是：

```text
rank 0: recv_topk_idx=[1, -1], recv_topk_weights=[0.3, 0.0]
rank 1: recv_topk_idx=[-1, 2], recv_topk_weights=[0.0, 0.7]
```

全局 expert ID 被转换为目标 rank 上的 local expert ID；不属于目标 rank 的槽位是 `-1`，相应无效权重置 `0`。单节点转换可见 `csrc/kernels/legacy/intranode.cu` 的 top-k copy 段；多节点转换可见 `csrc/kernels/legacy/internode.cu` 的 top-k transform 段。

### 4.2 dispatch 返回 shape 与语义

设当前 rank 收到 `M` 行（常见情况下 `M = num_tokens_per_rank_global[current_rank]`）：

| 返回值 | Shape / 类型 | 语义 |
|---|---|---|
| `recv_x` | `[M, H]`，或 FP8 tuple | 收到的 token hidden states |
| `recv_topk_idx` | `[M, K]` | 每个接收 token 的 local expert ID；其它 rank 的槽位是 `-1` |
| `recv_topk_weights` | `[M, K]` | 与路由槽位对应的权重；无效槽位为 0 |
| `recv_tokens_per_expert` | Python list `[num_local_experts]` | 每个本地 expert 的 token 数，按 `expert_alignment` 对齐；使用 `num_worst_tokens` 时此 list 为空 |
| `handle` | opaque tuple | combine 所需的反向通信元数据 |
| `dispatch_event` | EventOverlap | `async_finish=True` 时等待/串接异步通信 |

`recv_x` 行按照 DeepEP 的 rank/channel 通信布局存放，不是保证的 expert-major 排列。需要按 expert 连续聚集时，由后续 MoE/GEMM 路径按 `recv_topk_idx` / counts 处理、重排或融合处理。

## 5. all-to-all splits 的类比：哪些量对应 input/output splits

传统 all-to-all 的 `input_splits` / `output_splits` 可以帮助理解，但 V1 API 不一定以这两个同名数组暴露：

| 传统说法 | DeepEP normal 中相近的信息 |
|---|---|
| 本 rank 发给各 peer 的行数（input splits） | `num_tokens_per_rank`；单个目的地的具体 token 还由 `is_token_in_rank` 标识 |
| 本 rank 从各 source peer 收多少行（output splits） | 可从所有 peer 的发送计数/接收前缀布局得到；本地 `rank_prefix_matrix` 含有接收分段信息 |
| 具体行来自哪个 source token | intranode `recv_src_idx` 给源 rank 本地 token index；source rank 由 rank 分段确定 |
| expert 分桶顺序 | 不由这些 splits 自动产生；看 `recv_topk_idx`，是否 permute 由后续计算路径决定 |

因此 DeepEP normal 是 rank 粒度的 dispatch，不是“按 expert 把 payload 排好再做 all-to-all”的同义词。

## 6. handle：逻辑身份和物理通信位置

`handle` 是 dispatch 生成的 opaque 通信上下文，应用层应原样传回 `combine`。它不包含 `recv_x` 本身，也不等于 `recv_topk_idx` / `recv_topk_weights`。

### 6.1 单节点 handle

当前 V1 Python 实现的 intranode handle 是：

```text
(
  rank_prefix_matrix,
  channel_prefix_matrix,
  recv_channel_prefix_matrix,
  recv_src_idx,
  is_token_in_rank,
  send_head,
)
```

| 字段 | 典型 shape | 含义 |
|---|---:|---|
| `rank_prefix_matrix` | `[P, P]` | rank-to-rank 接收计数前缀；在当前 receiver 的目标列上，按 source rank 累计接收行数 |
| `channel_prefix_matrix` | `[P, C]` | 当前 rank 的发送侧布局：按目的 rank/channel 记录累计偏移 |
| `recv_channel_prefix_matrix` | `[P, C]` | 接收侧来源 rank/channel 在对应 rank 段中的偏移 |
| `recv_src_idx` | `[M]` | 每个接收行的源 rank 本地 token index；它自身不含 source-rank ID |
| `is_token_in_rank` | `[N, P]` | 源 token 到目标 rank 的布尔映射 |
| `send_head` | `[N, P]` | 内部发送缓冲 slot/head 元数据，用于反向 combine 路径 |

这些是内部布局；不承诺其 tuple 顺序/具体布局是跨版本的公共协议。

### 6.2 多节点 handle

internode handle 的 tuple 不同，包含 RDMA/NVLink 两级的 prefix、rank sum、`recv_src_meta`、`send_rdma_head`、`send_nvl_head` 等字段。源码中的 `SourceMeta` 保存源 RDMA rank 和 NVLink rank bitmap；它不是通用的 `(source_rank, source_token_idx)` 数组。单节点字段解释不能直接套给 internode tuple。

## 7. 具体追踪：确定 recv 行来自哪个 rank、哪段、哪个源 token

设当前接收 rank 是 rank 2，EP group 有 3 个 rank。发给 rank 2 的计数如下：

| 来源 rank | 总数 | channel 0 | channel 1 |
|---|---:|---:|---:|
| rank 0 | 2 | 1 | 1 |
| rank 1 | 3 | 2 | 1 |
| rank 2 | 1 | 0 | 1 |

### 7.1 rank 前缀确定大段

rank 2 对应列上的累计接收数是：

```text
rank_prefix_matrix[:, 2] = [2, 5, 6]
```

用半开区间 `[start, end)` 表示各 source-rank 段：

| source rank | 起点 | 终点 | `recv_x` 行范围 |
|---|---:|---:|---:|
| 0 | 0 | 2 | `[0, 2)` |
| 1 | 2 | 5 | `[2, 5)` |
| 2 | 5 | 6 | `[5, 6)` |

通用公式（当前 receiver 为 `r`，来源 rank 为 `s`）：

```text
rank_start(s) = 0                         if s == 0
                rank_prefix_matrix[s-1,r] otherwise
rank_end(s)   = rank_prefix_matrix[s,r]
```

### 7.2 接收 channel 偏移确定 rank 段内的小段

三组 channel 数量对应的 channel 起点为：

```text
source 0: channel counts [1, 1] -> starts [0, 1]
source 1: channel counts [2, 1] -> starts [0, 2]
source 2: channel counts [0, 1] -> starts [0, 0]
```

这些起点相对于各自 source-rank 段。加上 rank 段起点后：

| source rank | channel | `recv_x` 范围 |
|---|---:|---:|
| 0 | 0 | `[0, 1)` |
| 0 | 1 | `[1, 2)` |
| 1 | 0 | `[2, 4)` |
| 1 | 1 | `[4, 5)` |
| 2 | 0 | `[5, 5)`（空段） |
| 2 | 1 | `[5, 6)` |

### 7.3 `recv_src_idx` 给源 rank 内 token 编号

假设 kernel 收到的 source-local indices 为：

```text
recv_src_idx = [3, 8, 0, 4, 7, 2]
```

那么：

| `recv_x` 行 | 来源 rank | channel | 源 rank 本地 token index |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 3 |
| 1 | 0 | 1 | 8 |
| 2 | 1 | 0 | 0 |
| 3 | 1 | 0 | 4 |
| 4 | 1 | 1 | 7 |
| 5 | 2 | 1 | 2 |

**逻辑上识别 token 的身份只需 source rank + source-local token index。**channel 是 DeepEP 内部并行传输的物理分段信息；它帮助通信 kernel 定位队列/offset 和并行工作，不是 MoE 路由语义，应用层不需要解析它。`recv_src_idx` 单独只能给出 token index，source rank 需结合 rank 分段判断。

## 8. channel 数量从哪里来

V1 normal 配置中 `num_channels = config.num_sms / 2`，且 `num_sms` 必须为偶数。每个 channel 使用两个 block：偶数编号 block 做发送、奇数编号 block 做接收。

```text
默认 Buffer.num_sms = 20
默认 normal channel 数 = 20 / 2 = 10
```

可以通过 `Buffer.set_num_sms(...)` 或显式 `Config` 改变 `num_sms`，所以 10 是默认配置下的数量，不是固定协议常量。rank 数、expert 数不会直接决定 channel 数。

## 9. 第三步：expert 计算由 MoE 层完成

dispatch 只搬数据和路由元信息，不执行 expert MLP/GEMM。下游逻辑按 `recv_topk_idx` 找到需要的 local expert，并按模型约定应用 router 权重。若本地一个 token 对应多个 local experts，下游需要正确组织这些 expert 贡献，再提供 combine 所需的、与 dispatch 接收行顺序对应的结果。

`recv_tokens_per_expert` 是每个 local expert 的计数提示，不表示 `recv_x` 已经被切成 expert-major 的连续块。

## 10. 第四步：combine 做什么，不做什么

调用：

```python
combined_x, combined_topk_weights, combine_event = buffer.combine(
    x=expert_outputs,
    handle=handle,
    # topk_weights=...  # 可选的 top-k 权重归并输入
)
```

`expert_outputs` 的行必须对应 dispatch 的 `recv_x` 行。DeepEP 使用 `handle` 将这些结果反向发回来源 rank，并把来自多个 destination ranks 的对应贡献相加。

| 行为 | combine 是否负责 |
|---|---|
| 按 handle 反向通信 | 是 |
| 将多个目标 rank 对同一源 token 的 `x` 结果求和 | 是 |
| 执行 expert GEMM | 否 |
| 自动将 `x` 乘上 `topk_weights` | 否 |
| 可选地单独归并 `topk_weights` | 是，作为独立张量处理 |

因此 `combined_x.shape` 的首维是来源 rank 原来的本地输入 token 数 `N`，而不是 dispatch 接收行数 `M`。它是否是 gate 加权和，取决于调用方在送入 combine 前是否已按 MoE 公式加权：

```text
y_i = Σ_e gate[i,e] * expert_e(x_i)
```

如果 `expert_outputs` 已包含每个 expert 的加权贡献，那么 combine 后就是这些加权贡献跨 rank 的和；如果没乘 gate，combine 只会得到未加权贡献之和。单独传 `topk_weights=` 不会把它乘进 `x`。

## 11. 完整调用示意

以下是主流程的完整 API 顺序（不是独立脚本；Buffer 初始化和分布式环境由应用负责）：

```python
# 1) router 已产生 x, topk_idx, topk_weights
layout = buffer.get_dispatch_layout(topk_idx, num_experts)
num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, layout_event = layout

# 2) rank-level dispatch
recv_x, recv_topk_idx, recv_topk_weights, recv_tokens_per_expert, handle, dispatch_event = buffer.dispatch(
    x,
    topk_idx=topk_idx,
    topk_weights=topk_weights,
    num_tokens_per_rank=num_tokens_per_rank,
    num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
    is_token_in_rank=is_token_in_rank,
    num_tokens_per_expert=num_tokens_per_expert,
    previous_event=layout_event,
    async_finish=True,
)

# 3) 应用按 recv_topk_idx 执行本地 experts，并按模型公式应用 gate 权重
expert_outputs = moe_expert_path(
    recv_x,
    recv_topk_idx,
    recv_topk_weights,
    recv_tokens_per_expert,
)

# 4) reverse communication + sum；expert_outputs 行序必须仍对应 recv_x
combined_x, _, combine_event = buffer.combine(
    expert_outputs,
    handle=handle,
    previous_event=dispatch_event,
    async_finish=True,
)
```

`moe_expert_path` 是模型侧代码，不是 DeepEP API。对于 `async_finish=True`，消费输出前必须通过 event/stream 依赖正确同步；上面把 dispatch event 传给后续阶段只是说明依赖关系，实际 event 所属 stream 与调度方式应按应用的 `EventOverlap` 用法处理。

### 可运行的语义验证

DeepEP V1 需要支持的 CUDA/NVLink 环境。下面完整脚本（例如保存为 `normal_roundtrip.py`）用 `recv_x` 原样作为 combine 输入，验证通信 handle 会把每个 token 的一份副本送到每个目的 rank，再把这些副本加回 source rank。它**不是 MoE 计算**，也故意不乘 router 权重。仅支持 `get_*_config` 提供配置的 EP group size；示例用两个同节点 rank。

```python
import os

import torch
import torch.distributed as dist
import deep_ep
from deep_ep import Buffer


def main() -> None:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    group = dist.group.WORLD
    world_size = dist.get_world_size()
    assert world_size == 2, "This smoke test expects two same-node ranks"

    n_tokens = 8
    hidden = 64
    topk = 2
    local_experts = 2
    num_experts = world_size * local_experts

    x = torch.randn((n_tokens, hidden), device=device, dtype=torch.bfloat16)
    # topk without replacement: distinct global expert IDs per token
    topk_idx = torch.rand((n_tokens, num_experts), device=device).topk(topk, dim=1).indices
    topk_weights = torch.softmax(torch.randn((n_tokens, topk), device=device), dim=1)

    dispatch_config = Buffer.get_dispatch_config(world_size)
    combine_config = Buffer.get_combine_config(world_size)
    hidden_bytes = hidden * x.element_size()
    configs = (dispatch_config, combine_config)
    nvl_bytes = max(c.get_nvl_buffer_size_hint(hidden_bytes, world_size) for c in configs)
    rdma_bytes = max(c.get_rdma_buffer_size_hint(hidden_bytes, world_size) for c in configs)

    buffer = Buffer(
        group,
        num_nvl_bytes=nvl_bytes,
        num_rdma_bytes=rdma_bytes,
        explicitly_destroy=True,
    )
    try:
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            layout_event,
        ) = buffer.get_dispatch_layout(topk_idx, num_experts)

        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            recv_tokens_per_expert,
            handle,
            dispatch_event,
        ) = buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=layout_event,
            config=dispatch_config,
            async_finish=False,
        )

        # Identity payload: one copy returns from each destination rank.
        combined_x, _, _ = buffer.combine(
            recv_x,
            handle=handle,
            config=combine_config,
            async_finish=False,
        )

        destination_count = is_token_in_rank.sum(dim=1, keepdim=True).to(torch.float32)
        expected = x.float() * destination_count
        torch.testing.assert_close(combined_x.float(), expected, rtol=0.0, atol=0.2)

        if dist.get_rank() == 0:
            print("PASS: dispatch/combine round trip; shape =", tuple(combined_x.shape))
            print("recv rows on rank 0 =", recv_x.shape[0])
            print("destination ranks per local token =", destination_count.flatten().tolist())
    finally:
        buffer.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

运行命令：

```bash
torchrun --standalone --nproc_per_node=2 normal_roundtrip.py
```

预期可观察到 rank 0 输出 `PASS`，且 `combined_x.shape == (8, 64)`。每个 token 的结果近似等于原 `x` 乘以它被发送到的**不同目的 rank 数**；同一目的 rank 上命中两个 local experts 仍只算一个 token-rank 副本。需在有兼容 GPU、NVLink 和正确构建的 DeepEP V1 环境运行；当前 Markdown 的编辑/静态核对不等于已在 GPU 上运行验证。

## 12. 常见误解与症状

| 误解 | 正确理解 | 可能症状 |
|---|---|---|
| `recv_x` 已按 expert 排序 | 它按通信接收布局组织；路由在 `recv_topk_idx` | 用连续切片当 expert bucket，expert 输出会错配 token |
| `recv_src_idx[i]` 单独能知道 source rank | 它是源 rank 本地 token index；rank 要由 rank 段元数据确定 | 不同 rank 的 token index 相同，单独映射会冲突 |
| handle 是稳定的公开 source ID 列表 | handle 是 opaque 的通信元数据 | 依赖内部 tuple 顺序会在版本/路径变化时失效 |
| 一个 token 的每个 top-k expert 都会导致对同 rank 复制一份 `x` | 复制粒度是 token-rank；多个 local experts 可共享一份 token 行 | 把 `recv_x` 行数错误地当成 top-k assignment 总数 |
| combine 自动执行 router 加权 | combine 对 `x` 做反向传输与求和；不自动乘 gate | 输出尺度与预期 MoE 公式不符 |
| `combine(topk_weights=...)` 就会给 `x` 加权 | 权重张量是单独归并，不与 `x` 相乘 | 以为传了权重就能省略模型侧 gate multiply |
| channel 是 token 的逻辑身份 | channel 是并行传输的物理分段元数据 | 应用代码不必要地解析 handle 并绑定内部实现 |

## 13. 速记卡

```text
layout:
  num_tokens_per_rank [P]
  num_tokens_per_expert [E]
  is_token_in_rank [N, P]
  多节点额外 num_tokens_per_rdma_rank

dispatch payload:
  x (+ optional scales) + topk_idx + topk_weights

接收端:
  recv_x [M, H]
  recv_topk_idx [M, K]     # local expert IDs, other-rank slots = -1
  recv_topk_weights [M, K] # other-rank slots = 0
  per-local-expert counts + opaque handle

布局:
  rank-level communication, not expert-major permutation
  rank + source-local token index identify logical source token
  channel offsets are internal transport metadata

combine:
  reverse route via handle + sum contributions across destination ranks
  combined_x [N, H], N = this source rank's original local token count
  no implicit router-weight multiply; gate must be applied by MoE path

normal channels:
  C = config.num_sms / 2
  default num_sms=20 -> 10 channels; configurable, not a fixed protocol constant
```

## 14. 自测题与答案

1. **一个 token 的 top-2 experts 都在同一个 rank，会把 `x` 发两份给该 rank 吗？**
   答：不会，通常按 token-rank 去重发送一份；两个 top-k 槽位仍会在路由元数据中体现。
2. **`recv_topk_idx` 中的 `2` 是全局 expert 2 吗？**
   答：dispatch 接收后它表示目标 rank 的 local expert index；非本 rank 槽位是 `-1`。
3. **`recv_src_idx[i] = 7` 能否单独确定来源 rank？**
   答：不能。它表示源 rank 本地输入第 7 个 token；rank 由 prefix/rank 分段确定。
4. **`combined_x` 的首维是什么？它保证是加权 MoE 输出吗？**
   答：是当前 source rank 原来的 `N` 个 token；是否 gate 加权取决于传入 combine 前的 MoE 计算，combine 不自动乘权重。
5. **默认 channel 有多少？**
   答：`num_sms=20` 时有 `20/2=10`；配置改变后数量也变。

## 15. 学习进度

- [x] DeepEP V1 normal：layout → rank dispatch → local expert metadata → handle → combine
- [x] 区分 rank-major 通信布局、expert-major 计算布局与 gate 加权
- [x] DeepEP V1 low-latency：定容 dispatch、packed expert 输入、handle 回程元数据、weighted combine 与 hook

### 下一知识点

DeepEP V1 normal 与 low-latency 的性能边界：结合 workload 看容量、buffer 占用、通信延迟和双 micro-batch overlap 的取舍。
