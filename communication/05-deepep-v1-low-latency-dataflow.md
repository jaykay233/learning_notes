# DeepEP V1 Low-latency：expert-major dispatch、handle 与 weighted combine

> 承接 [04-deepep-v1-normal-dataflow.md](./04-deepep-v1-normal-dataflow.md)。本文聚焦 DeepEP V1 `legacy` low-latency 路径，并和 normal 的 rank-major 接收语义对照。代码对照仓库：`/Users/saboxu/Downloads/communication/DeepEP`；重点文件是 `deep_ep/buffers/legacy.py`、`docs/legacy.md`、`csrc/legacy/buffer.hpp`、`csrc/kernels/legacy/internode_ll.cu`。

```text
本次讲解位置
章节：communication / DeepEP V1
小节：Low-latency dispatch / expert / combine 完整数据流
知识点：expert-major 固定容量 buffer、handle 的回程元数据、weighted combine 与 hook
上次：normal 路径中 recv_topk_idx 如何标出每行对应的 local experts
下次：DeepEP V1 normal 与 low-latency 的性能边界及 overlap 调度
PTX：不适用（本文聚焦 CUDA 通信 API、buffer layout 与 RDMA/NVLink 数据流）
```

## 为什么现在讲这个

刚理解 normal 的 `recv_x` 后，一个很自然的问题是：low-latency 是否只换了通信 kernel？并不是。两条路径对 token 的组织粒度不同：normal 以目标 rank 为主要接收分组，再由 `recv_topk_idx` 指出每行要去哪些 local experts；low-latency 返回的 buffer 已按 local expert 分区。若混用二者的布局假设，最直接的后果是把 low-latency 的 capacity 当作有效 token 数，或把 normal 的一行误当成一个专家的独立输入。

## 1. 一句话心智模型

```text
router: x[N,H] + topk_idx[N,K] (+ topk_weights[N,K])
  -> low_latency_dispatch：按选中 expert 路由，目标端写入 local-expert 分区
  -> recv_x[L, C*P, H] + recv_count[L] + handle
  -> 每个 local expert 计算有效行
  -> low_latency_combine：利用 handle 反向定位源 rank/token，并按 topk_weights 加权归并
  -> combined_x[N,H]
```

这里 `L` 是每 rank 的 local expert 数，`P` 是 EP ranks 数，`C` 是每 rank 输入 token 的容量上限，`H` 是 hidden size，`K` 是 top-k。

**expert-major 描述的是 low-latency 的接收/计算布局。** 数据仍然在 rank 之间传输；不是一个脱离 rank 的全局 expert 网络矩阵。normal 与 low-latency 的另一处区别是：normal 通常对每个目的 rank 发送一份 token，再携带 local-expert 路由信息；low-latency 的接收区按 expert 分开，因此同一个 token 若选中同一 rank 上的多个 experts，会进入多个 expert 输入区。

## 2. 输入、输出与 shape

### 输入

```text
x             [N, H]    BF16 输入 token
topk_idx      [N, K]    全局 expert ID；-1 表示无路由
 topk_weights [N, K]    combine 时使用的 router 权重
```

`low_latency_dispatch` 接收 `x`、`topk_idx`、`capacity` 和全局专家数；它**不要求先调用 normal 的 `get_dispatch_layout`**。路由与 low-latency 所需的打包布局在该路径内部计算。调用方要保证每个 rank 的 token 数严格小于给定 capacity，所有 rank 使用相同上限。

### Dispatch 返回值

令 `L = E/P`，其中 `E` 是全局 expert 数：

| 返回项 | 典型 shape / 内容 | 作用 |
|---|---|---|
| `recv_x`（默认 FP8 tuple） | data `[L, C*P, H]`；scale 通常 `[L, C*P, H/128]` | 每个 local expert 一个接收区；也可 `use_fp8=False` 返回 BF16 |
| `recv_count` | `[L]`, int32 | 每个 local expert 的实际有效 token 数 |
| `handle` | 5 元组，见后文 | combine 所需回程路由与布局信息 |
| `event` | `EventOverlap` | 异步路径的执行事件 |
| `hook` | callable | 启用 receive-hook 时确保数据已经到达 |

`C*P` 是每个 local expert 的**容量槽位数**，不是有效行数。有效数由 `recv_count[l]` 给出；来源 rank 对应的具体区间则由 handle 的 layout 元数据给出。FP8 scale 的物理布局为 TMA 兼容的转置/column-major 组织，使用时应尊重接口返回的布局。

## 3. 为什么同一个 token 可能进多个 expert 区

假设 rank 1 拥有全局 expert `4–7`，输入 token `t` 选择了 expert `[4, 7]`。normal 路径可以让 rank 1 接收一次 `x[t]`，再让 `recv_topk_idx` 的这一行说明要送给 local expert `[0, 3]`。low-latency 的接收 buffer 按 local expert 分开，因此 `x[t]` 会分别出现在 rank 1 的 expert 0 与 expert 3 接收区中。

这解释了两个路径的结构差异：

| 维度 | Normal | Low-latency |
|---|---|---|
| 接收组织 | rank/channel 为主要通信布局 | 第一维直接是 local expert |
| 多个选中专家在同一 rank | token 行可共享，`recv_topk_idx` 描述多个 local expert | token 会落入对应的多个 expert 分区 |
| 动态/固定容量 | dispatch 返回实际接收 token 行 | 预先给 `C`，分配 `[L,C*P,H]` 容量 buffer；用 `recv_count` 识别有效数 |
| Combine 权重 | normal combine 默认求和不乘权重；权重处理由相应接口/调用方负责 | `low_latency_combine` 用 `topk_weights` 加权并归并 |

## 4. `handle` 里具体有什么

当前 V1 Python 封装构造的 handle 是：

```python
handle = (
    packed_recv_src_info,
    packed_recv_layout_range,
    capacity,
    hidden,
    num_experts,
)
```

前两个 tensor 是回程元数据，不是 activation：

1. `packed_recv_src_info`：shape `[L, P*C]`，int32。每个接收槽位记录 token 在其**来源 rank 本地**的 token 下标。
2. `packed_recv_layout_range`：shape `[L, P]`，int64。索引 `[local_expert, source_rank]` 给出该 expert 接收 buffer 中来自某个 source rank 的连续区间。当前实现的打包约定为：

   ```python
   packed = packed_recv_layout_range[local_expert, source_rank].item()
   begin = packed >> 32
   count = packed & 0xFFFFFFFF
   slots = slice(begin, begin + count)
   ```

因此，对一个 local expert `l` 和来源 rank `r`：`recv_x[l, begin:begin+count]` 是来自 rank `r` 的有效数据；对应 `packed_recv_src_info[l, begin:begin+count]` 给出这些行在来源 rank 上的原 token 下标。

后三项 `capacity`、`hidden`、`num_experts` 是 combine 重建通信形状/参数所需的信息。**通常把 handle 原样交给 `low_latency_combine`，不应在业务代码里自行修改或重建它。**

注意，handle 本身不装 `topk_weights`。combine 调用还要单独提供 `topk_idx` 和 `topk_weights`；handle 提供的是 dispatch 实际接收布局与来源 token 映射。

## 5. 一个完整的数值布局例子

取：

```text
P = 4 ranks
E = 16 experts   -> L = E/P = 4 local experts/rank
C = 128 tokens/rank capacity
H = 4096
当前 rank 输入 N = 64
```

low-latency dispatch 返回的 BF16 payload shape 是：

```text
recv_x:     [4, 512, 4096]  # 512 = 128 * 4
recv_count: [4]
```

例如 `recv_count = [30, 27, 34, 25]`：四个 local experts 分别有 30、27、34、25 个有效 expert-token 输入，总计 116 个 expert-token 路由项。总数大于 `N=64` 并不矛盾，因为 token 可能 top-k 到多个专家；而 512 是单 expert 的分配容量，不是真实收到 512 行。

对 expert 2，若 layout 元数据表示：

```text
source rank 0 -> begin=0,   count=8
source rank 1 -> begin=8,   count=7
source rank 2 -> begin=15,  count=9
source rank 3 -> begin=24,  count=10
```

则 expert 2 的前 34 行有效，各来源 rank 的区间由对应的 `packed_recv_layout_range[2, r]` 给出。比如区间 `[15,24)` 是来自 source rank 2 的 9 行；这 9 行各自的原 token 下标在 `packed_recv_src_info[2, 15:24]`。区间外的容量槽位不能当有效 token 用。

专家计算之后，expert 输出必须保留与接收输入一致的 expert/slot 组织关系，以便 combine 根据同一个 handle 找回源位置。combine 对每个原 token 的概念结果是：

```text
combined_x[token] = Σ_k topk_weights[token,k] * expert_output_for_that_route
```

输出 shape 回到 `[N,H]`，在例子里就是 `[64,4096]`。

## 6. API 调用与 hook 顺序

同步、不重叠的调用可以直接使用：

```python
recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
    x,
    topk_idx,
    num_max_dispatch_tokens_per_rank=capacity,
    num_experts=num_experts,
    async_finish=False,
    return_recv_hook=False,
)

# 按 recv_count 和该接口约定的 packed layout 做 local experts 计算，得到 expert_out
combined_x, event, hook = buffer.low_latency_combine(
    expert_out,
    topk_idx,
    topk_weights,
    handle,
    async_finish=False,
    return_recv_hook=False,
)
```

上面省略的 expert kernel 由模型的 grouped-GEMM/MoE 实现提供；关键约束是 `expert_out` 必须仍符合 low-latency combine 所期待的 `[L,C*P,H]` expert-major 形式。

重叠通信与计算时可设置 `return_recv_hook=True`：dispatch 会先 issue RDMA 请求，但不能因此立即读取尚未确认到达的数据；在开始依赖接收数据前调用 dispatch 返回的 `hook()`。Combine 同理，在依赖 combine 结果前调用它返回的 hook。hook 允许 RDMA 网络传输在后台进行，不占用计算阶段的 SM；它不是“数据已经到齐”的自动保证。按该 API 的约束，不能同时持有超过两个 low-latency kernel 的返回结果 tensor，因为底层结果会复用两个 buffer。

## 7. 传输路径：RDMA-centric，但节点内 NVLink 是可配置能力

V1 legacy 文档把 low-latency kernel 描述为 “pure RDMA”，并要求所有 ranks 通过 RDMA/IBGDA 可见；这是该低时延路径的核心通信前提。与此同时，V1 roadmap 明确列出 intranode low-latency kernel 的 NVLink protocol 支持，源码也包含 NVLink P2P 地址路径。`Buffer` 构造参数 `allow_nvlink_for_low_latency_mode` 控制是否允许 low-latency 使用 NVLink traffic，当前 Python API 默认值为 `True`；设为 `False` 时会关闭 NVSHMEM P2P 路径。源码特别提示，允许 NVLink traffic 与 hook-based overlap “somehow incompatible”，并警告 PCIe 连接可能有 memory-ordering 问题。因此不要把它简化成“所有物理传输、包括同节点 GPU 间，都一定经过 NIC”，也不要在没有验证拓扑/同步要求时随意打开该选项。更稳妥的描述是：

- Low-latency API 的基础要求仍是所有 ranks RDMA 可见并启用 IBGDA；
- 同节点 NVLink P2P 是可配置的路径能力，不等于跨节点 RDMA 被替代；
- 若使用 hook overlap，需特别评估 `allow_nvlink_for_low_latency_mode` 的兼容性；
- 这不是 normal 路径那套面向高吞吐的 NVLink/RDMA forwarding API 的简单复用；low-latency 仍有自己的 buffer、handle 和 hook 协议。

## 8. 最小双-rank identity-expert smoke test

下面的完整脚本把专家计算设为 identity：每个收到的 expert 输入原样作为 expert 输出。两个 ranks 的每个 token 都路由到全局 experts 0 与 2，权重分别是 0.25、0.75；两路专家输出加权后应恢复原 token。这个测试能验证 dispatch、packed expert buffer、handle 和 weighted combine 的 round trip，但不能替代真实 MoE GEMM 的正确性与性能测试。需在 DeepEP V1 已构建、CUDA/NVSHMEM/IBGDA 已配置的 GPU 环境运行。

```python
# low_latency_identity_roundtrip.py
import os

import torch
import torch.distributed as dist

import deep_ep
from deep_ep import Buffer


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    group = dist.group.WORLD

    capacity = 8
    hidden = 7168
    num_experts = 4
    num_ranks = dist.get_world_size(group)
    assert num_ranks == 2
    num_local_experts = num_experts // num_ranks

    rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
        capacity, hidden, num_ranks, num_experts
    )
    buffer = Buffer(
        group,
        num_nvl_bytes=0,
        num_rdma_bytes=rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=num_local_experts,
        allow_nvlink_for_low_latency_mode=False,
        explicitly_destroy=True,
    )

    try:
        num_tokens = 4
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
        topk_idx = torch.empty((num_tokens, 2), dtype=deep_ep.topk_idx_t, device="cuda")
        topk_idx[:, 0] = 0
        topk_idx[:, 1] = 2
        topk_weights = torch.empty((num_tokens, 2), dtype=torch.float32, device="cuda")
        topk_weights[:, 0] = 0.25
        topk_weights[:, 1] = 0.75

        recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
            x, topk_idx, capacity, num_experts,
            use_fp8=False, async_finish=False, return_recv_hook=False,
        )

        # Identity expert: preserve the required [local_expert, capacity*ranks, hidden] layout.
        expert_out = recv_x.clone()
        combined_x, event, hook = buffer.low_latency_combine(
            expert_out, topk_idx, topk_weights, handle,
            async_finish=False, return_recv_hook=False,
        )

        torch.testing.assert_close(combined_x, x, rtol=0.0, atol=0.1)
        if dist.get_rank(group) == 0:
            print("PASS: low-latency dispatch/combine identity round trip")
            print("recv_x:", tuple(recv_x.shape))
            print("recv_count:", recv_count.tolist())
            print("combined_x:", tuple(combined_x.shape))
    finally:
        buffer.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

运行示例：

```bash
torchrun --standalone --nproc_per_node=2 low_latency_identity_roundtrip.py
```

预期观察到 `PASS`、`recv_x` 的首维为每 rank 的 local expert 数，且 `combined_x` shape 为 `(4, 7168)`。本笔记编辑时只做静态核对，没有在具备 NVSHMEM/IBGDA 的 GPU 环境执行该脚本。

## 9. 容量与配置约束

- `num_max_dispatch_tokens_per_rank` 在所有 ranks 上取相同值，且本轮实际 dispatch token 数必须小于它。
- Low-latency 固定容量 buffer 内存开销大；V1 文档建议 decoding 场景的容量通常小于 256，并用 `Buffer.get_low_latency_rdma_size_hint(...)` 估算所需 RDMA buffer。
- 最佳性能建议 QP 数量等于每 rank 的 local expert 数，即 `num_experts // group.size()`。
- QP depth 至少满足 `(capacity + 1) * 2`。
- `allow_nvlink_for_low_latency_mode` 默认是 `True`；它与 hook-based overlap 的兼容性要依据实际拓扑/路径评估，源码警告 PCIe P2P 可能有 memory-ordering 问题。
- 默认 `use_fp8=True`；如果设为 FP8，expert kernel 要理解 payload 与 scale 格式。若设置 `use_fp8=False`，接收数据为 BF16。
- `-1` expert index 代表该 top-k 槽位不选择专家。

## 10. 常见误解与症状

| 误解 | 正确理解 | 可能症状 |
|---|---|---|
| `[L,C*P,H]` 的每一行都有效 | 只有来源 rank 描述的区间有效，`recv_count` 是每个 expert 的有效总数 | 把未初始化/无效槽位送入 GEMM，输出出现垃圾值或数值不匹配 |
| 必须先调用 normal `get_dispatch_layout` | low-latency API 根据 `topk_idx` 内部计算自己的布局 | 把 normal 的 layout 元数据传错接口，或徒增一次无用计算 |
| handle 是 token 或 expert activation | handle 是回程路由/布局 metadata 加静态参数 | 手动重排/改写 handle 后 combine 回错源 token |
| low-latency combine 只加和 | 它使用传入 `topk_weights` 做 weighted reduce | 与 reference `x * sum(topk_weights)` 不一致 |
| expert-major 代表全局网络上的 expert collective | 它描述目标端 local-expert 接收布局，物理通信仍由 ranks 承载 | 混淆网络目的地粒度与 tensor 存储顺序 |
| 设置 hook 后返回就代表接收已完成 | hook 模式要求显式调用接收 hook 后再消费数据 | 偶发读到未到达 payload、结果不稳定 |

## 11. 自测（含答案）

1. `recv_x` 第二维 `C*P` 是否代表真实有效行数？
   - **否。** 它是每个 local expert 的容量；实际有效数看 `recv_count`，每个来源的行范围看 layout metadata。
2. Low-latency 调用前要不要跑 normal `get_dispatch_layout`？
   - **不要。** `low_latency_dispatch` 根据 `topk_idx` 内部准备 low-latency 布局。
3. `handle` 如何把某个 expert 输出行映射回源 token？
   - 先由 local expert 与 `packed_recv_layout_range` 中的 source-rank 区间确定来源 rank/槽位，再从相应 `packed_recv_src_info` 读该源 rank 的 token 下标；`topk_idx`/权重另由 combine 调用传入。
4. 一个 token 在同一 rank 选中两个 local experts，low-latency 接收布局会怎样？
   - 它会进入两个 local expert 的输入分区；这和 normal rank-major 路径中一行 token 可由 `recv_topk_idx` 描述多个 local experts 不同。
5. `return_recv_hook=True` 后，什么时候能安全读取接收结果？
   - 在依赖该数据前调用 dispatch 返回的接收 hook；combine 结果也在消费前调用 combine hook。

## 12. 对照源码与验证

- Python API / 返回 handle：`deep_ep/buffers/legacy.py` 的 `low_latency_dispatch`、`low_latency_combine`。
- 固定容量 tensor 分配：`csrc/legacy/buffer.hpp` 的 V1 `low_latency_dispatch`。
- per-source-rank 区间、token source index 写入：`csrc/kernels/legacy/internode_ll.cu` 的 receive-and-packing 代码。
- 形状、有效计数、hook 与 combine correctness 检查：`tests/legacy/test_low_latency.py`。
- 公开示例、buffer size hint、QP 配置、double-micro-batch hook overlap：`docs/legacy.md`。

验证时先在已配置 CUDA/NVSHMEM/IBGDA 的环境运行仓库 V1 low-latency 测试；单纯静态阅读不能验证实际 RDMA、NVLink 拓扑、hook overlap 或 kernel correctness。
