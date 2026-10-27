# TensorRT-LLM 的 EP/EPLB：逻辑专家、物理 slot 与 dispatch 布局

> 源码快照：TensorRT-LLM `0372799d1543ef5920c1318b829cf127800dbcb9`（2026-10-27）。
> 本文把 MoE routing、EPLB placement、EP dispatch 和 combine 放在同一条数据路径里。核心不变量是：router 的 top-k 始终表示逻辑 expert；EPLB 只改变这些逻辑 expert 当前落在哪个物理 slot/rank。

```text
本次讲解位置
章节：inference-scheduling
小节：TensorRT-LLM EP/EPLB 的逻辑专家到物理 expert 映射
知识点：logical top-k、global slot、rank-major dispatch、expert-major GEMM、combine
上次：TensorRT-LLM V2 的 KV pressure fallback ladder
下次：ADP router、phase intent 与 dummy request
PTX：不适用
```

## 为什么现在讲这个

此前最容易卡住的问题是：如果专家原本按 rank 推算，EPLB 调整专家顺序后，router 已经输出的 top-k 还能不能关联到原来的专家？

答案取决于把三个身份分开：

```text
logical expert id      router 选择到的专家语义
physical global slot   EPLB 当前把该专家放在哪里
dispatch row layout    通信和 grouped GEMM 如何排列 token
```

top-k 输出的是第一项，不是第二项。EPLB 在 dispatch 前用当前 placement 把 logical id 重新映射成 global slot；只要权重和 placement 来自同一版本，top-k 仍然对应原来的逻辑专家。真正会出错的是把 `expert_id // experts_per_rank` 这种静态 rank 公式继续用于 EPLB 后的专家，或者只更新了路由表却没有把对应权重搬到新位置。

当前 checkout 中没有独立的 `LPLB` 符号或 scheduler 阶段。公开源码能确认的是 EPLB 的 logical expert → replica/global slot placement；不能把未在源码中出现的 LPLB 行为硬套进这条路径。

## 1. 心智模型：先选专家，再决定专家在哪

一次 MoE forward 可以抽象成：

```text
token hidden states
  -> router: 每个 token 选 K 个 logical expert ids
  -> EPLB: logical expert + replica policy -> global physical slot
  -> EP dispatch: (token, top-k row) 发到 global slot 所在 rank
  -> local permute: 按本 rank 的 local expert 聚成 GEMM 分组
  -> grouped GEMM + activation
  -> combine: 按原 token/top-k 位置和权重求和
  -> home rank 的 [T, H]
```

这里有两个不同的“顺序”：

- top-k 内部的顺序表示第 0、第 1 个专家选择。它决定 combine 时使用哪个 router weight。
- 物理专家在 rank 内的顺序由 placement 决定。它只影响 token 被送到哪个 slot/rank，以及 GEMM 分组从哪里开始。

所以“专家顺序换了”不等于“top-k 语义换了”。EPLB 不需要回头改写 router 已经输出的 logical ids，只需要在 dispatch 之前应用新的 placement。

## 2. 从 logical expert 到 rank 和 local slot

设：

```text
E = EP size，也就是 rank 数
S = 每 rank 的 slot 数
G = E * S = global slot 总数
```

一个全局 slot 由 rank 和 local slot 组成：

```text
global_slot = rank * S + local_slot
rank        = global_slot // S
local_slot  = global_slot % S
```

C++ `prepareGpuPlacementInfo()` 正是用这个公式生成 `globalSlotIds`：

```cpp
int globalSlotId = rank * metaInfo.slotCountPerRank + slotId;
globalSlotIds[expertId].push_back(globalSlotId);
```

`globalSlotIds` 按 logical expert 分组，每个 expert 可以有一个或多个 replica。GPU 上的 `moeComputeRouteDevice()` 再读取：

- `expertReplicaCount[expert]`
- `expertReplicaStartOffset[expert]`
- 该 expert 的 replica 对应的 global slot ids

无冗余专家时，kernel 可以直接把 logical expert 映射到唯一 global slot。存在 replica 时，它会在可用 replicas 间做轮转，把不同 token 分散到不同物理副本。

### 为什么旧的 rank 公式失效

没有 EPLB、每个 rank 放相同数量且专家连续编号时，常写成：

```text
rank = logical_expert // experts_per_rank
local_expert = logical_expert % experts_per_rank
```

有 EPLB 后，这个公式不再成立。专家 6 可能从 rank 3 搬到 rank 0；同一个专家还可能有多个副本。正确公式必须经过当前 placement：

```text
logical_expert
  -> replica global_slot
  -> rank/local_slot
```

`rank` 是结果的属性，不是 logical expert id 的固定函数。

## 3. 源码数据路径

PyTorch MoE scheduler 的 `moe_scheduler.py` 按以下顺序工作：

```text
routing_method.apply()
  -> token_selected_experts              # logical expert ids, [T, K]
  -> _load_balancer_update_statistic()
  -> _load_balancer_route()
  -> token_selected_slots                # physical global slot ids, [T, K]
  -> comm.dispatch()
```

关键点：

- `token_selected_experts` 是 pre-EPLB 的 logical top-k。
- `_load_balancer_route()` 调用 C++ `moe_load_balance_routing`，使用当前 placement 和 replica 列表返回 routed slot ids。
- 没开 EPLB 时，`token_selected_slots` 直接等于 `token_selected_experts`。
- `route_capture.py` 明确捕获 pre-EPLB logical top-k，说明 logical id 是跨 placement 版本稳定的语义身份。

`moe_load_balancer.py::route()` 本身不决定专家，它只把已选中的 logical experts 交给 C++ kernel 做当前 placement 下的 slot 选择。

## 4. EPLB 对 scheduler 到底影响什么

EPLB 不改变以下东西：

- 请求的 priority；
- generation-first 的顺序；
- `max_num_tokens` 和本轮 token budget；
- 哪些请求处于 context/generation；
- prefill 还是 decode 的 phase 选择。

EPLB 会改变：

- 某个 token 被发往哪个 EP rank；
- 每个 rank 收到的 token/top-k 行数；
- 哪个副本承担 load；
- EP collective 中最慢 rank 的 grouped GEMM 和通信工作量；
- `recv_expert_count`、dispatch buffer 填充量和 combine 的归约来源。

因此它是“数据面负载和通信映射”的变化，会间接改变 forward 耗时和其他 rank 的等待时间，但不是“scheduler 重新选请求”的机制。EPLB 变好时，各 rank 的 expert token 数更均衡；变差时，某些 rank 更重，其他 rank 在 EP 同步点等它。

### 一个必须保持的正确性契约

placement 与权重迁移必须描述同一版专家布局。只要一次 forward 使用了以下组合之一，就可能把 token 送到“新 rank 的旧专家”：

```text
新 rankExpertIds/globalSlotIds + 旧 slot 权重
旧 placement + 新 slot 权重
```

surface 上仍然能完成 dispatch 和 combine，但计算结果已经错了。EPLB 更新应视为两个部分一起生效：

```text
routing metadata: logical expert -> physical slot
expert weights:   physical slot -> 对应 logical expert 的参数
```

不同请求的 top-k 逻辑 id 不需要跟着改；需要同步的是 placement 和该 placement 指向的内容。

## 5. EP rank-major 与 expert-major 是什么

这里最容易把“通信返回的数组形状”和“GEMM 需要的数据顺序”混为一谈。

### NCCL EP Low-Latency：rank-major receive buffer

`nccl_ep.py` 返回：

```text
recv_x: [ep_size, max_tokens_per_rank, hidden]
```

第一维按来源 rank 分块。对 EP size 为 4、每 rank 最多 3 个 token、hidden 为 4 的例子：

```text
[4, 3, 4] -> flatten -> [12, 4]
```

无效 padding 行用 `-1` slot 标记。这个 rank-major 形状是 all-to-all 的接收布局，不代表 local expert 已经分组。

### DeepEP Low-Latency：expert-major receive buffer

DeepEP LL 直接返回：

```text
[local_experts, ep_size * max_tokens, hidden]
```

以 2 个 local expert 为例：

```text
[2, 12, 4] -> 24 rows
```

同一行前面的维度已经是本 rank 的专家，`recv_expert_count` 告诉每个 local expert 实际有多少行。它更接近 grouped GEMM 需要的“每个 expert 一段连续行”。

### 从 rank-major 到 grouped GEMM

NCCL rank-major 输出先被 flatten 成 token/top-k 行，再由 `moe_permute_op` 结合 `token_selected_slots` 生成本地 expert 的连续分组和 `expert_first_token_offset`。这个 offset 是 grouped GEMM 的边界：

```text
expert 0: rows [0, 1)
expert 1: rows [1, 4)
expert 2: rows [4, 4)   # 零 token 也要保留合法空区间
```

DeepEP 的 direct metadata 路径已经有 expert-major 数据和 `recv_expert_count`，可以减少一次重排，但仍然要满足同一个 grouped-GEMM 布局契约。

## 6. combine 如何回到原 top-k

dispatch 会丢掉或重排行信息，如果不保存映射，combine 无法把结果放回原 token。需要保留的关键信息包括：

- token 的 home rank；
- 原始 `(token, top_k_slot)` 位置；
- 对应的 router weight；
- 通信后的 row 到原 row 的映射。

NCCL EP 的 combine 使用原 top-k weights，把各来源 rank 的贡献加回 home rank。DeepEP LL 的 combine 直接消费 expert-major 输出并应用路由权重，所以它不能接受一个已经把权重提前 finalize 掉的张量。

只要映射正确，EPLB 改变物理路径不应该改变 combine 的数学结果。下面的 trace 故意让 expert 输出只依赖 logical id、token 和 top-k 位置，再用相同结果验证这一点。

## 7. 数值 trace：专家 6 从 rank 3 搬到 rank 0

使用 4 个 EP rank、每 rank 2 个 slot：

```text
before:
rank 0: [0, 1]    global 0, 1
rank 1: [2, 3]    global 2, 3
rank 2: [4, 5]    global 4, 5
rank 3: [6, 7]    global 6, 7

after:
rank 0: [0, 6]    global 0, 1
rank 1: [2, 3]    global 2, 3
rank 2: [4, 5]    global 4, 5
rank 3: [1, 7]    global 6, 7
```

router 输出不变：

```text
token 0: [6, 2]
token 1: [1, 2]
token 2: [6, 0]
```

以 token 0 的 top-k 第 0 行为例：

| 阶段 | logical | 计算 | global slot | rank | local slot |
|---|---:|---|---:|---:|---:|
| before | 6 | `6` 是唯一副本 | 6 | `6//2=3` | `6%2=0` |
| after | 6 | 当前 replica 映射到 global 1 | 1 | `1//2=0` | `1%2=1` |

token 0 的 combine 手算：

```text
before: (0*100 + 6*10 + 0) + (0*100 + 2*10 + 1)
      = 60 + 21
      = 81

after:  仍是 logical 6 和 logical 2，只是 rank/slot 路径改变
      = 81
```

三个 token 的最终 combine 都是：

```text
[81, 231, 461]
```

但每个 rank 收到的 route 数变了：

```text
before: [2, 2, 0, 2]
after:  [3, 2, 0, 1]
```

这就是 EPLB 的真实影响：请求 scheduler 没变，dispatch 目的地和 rank 负载变了。rank 2 在两种布局下都没有收到 route，它仍必须参与 EP collective；否则其他 rank 会等在通信同步点。

## 8. 完整可运行 artifact

文件名：`inference-scheduling/examples/tensorrt_llm_eplb_dispatch.py`

"""Trace logical experts through EPLB placement, EP dispatch, and combine."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Placement:
    slots_by_rank: tuple[tuple[int, ...], ...]
    slots_per_rank: int

    def __post_init__(self) -> None:
        assert all(len(row) == self.slots_per_rank for row in self.slots_by_rank)

    def slots_for(self, expert_id: int) -> list[int]:
        slots: list[int] = []
        for rank, row in enumerate(self.slots_by_rank):
            for local_slot, placed_expert in enumerate(row):
                if placed_expert == expert_id:
                    slots.append(rank * self.slots_per_rank + local_slot)
        if not slots:
            raise KeyError(f"logical expert {expert_id} has no physical slot")
        return slots

    @property
    def ep_size(self) -> int:
        return len(self.slots_by_rank)


@dataclass(frozen=True)
class Route:
    token: int
    top_k_slot: int
    logical_expert: int
    global_slot: int
    destination_rank: int
    local_slot: int


def route_topk(
    topk_logical: list[list[int]],
    placement: Placement,
    *,
    offset_by_ep_rank: bool = False,
    source_rank: int = 0,
) -> list[Route]:
    routes: list[Route] = []
    round_robin_key: dict[int, int] = {}

    for token, experts in enumerate(topk_logical):
        for top_k_slot, expert in enumerate(experts):
            replicas = placement.slots_for(expert)
            key = round_robin_key.get(expert, 0)
            if offset_by_ep_rank:
                key += source_rank
            global_slot = replicas[key % len(replicas)]
            round_robin_key[expert] = key + 1
            routes.append(
                Route(
                    token=token,
                    top_k_slot=top_k_slot,
                    logical_expert=expert,
                    global_slot=global_slot,
                    destination_rank=global_slot // placement.slots_per_rank,
                    local_slot=global_slot % placement.slots_per_rank,
                )
            )
    return routes


def combine(routes: list[Route]) -> list[int]:
    # The toy expert result depends only on the logical identity and token/top-k
    # position. That makes an incorrect physical mapping visible in the sum.
    combined = [0] * (max(route.token for route in routes) + 1)
    for route in routes:
        toy_expert_output = route.token * 100 + route.logical_expert * 10 + route.top_k_slot
        combined[route.token] += toy_expert_output
    return combined


before = Placement(
    slots_per_rank=2,
    slots_by_rank=((0, 1), (2, 3), (4, 5), (6, 7)),
)
after = Placement(
    slots_per_rank=2,
    slots_by_rank=((0, 6), (2, 3), (4, 5), (1, 7)),
)
topk_logical = [[6, 2], [1, 2], [6, 0]]

print("router logical top-k (EPLB does not rewrite these IDs):", topk_logical)
for label, placement in (("before EPLB", before), ("after EPLB", after)):
    routes = route_topk(topk_logical, placement)
    print(f"\n{label}")
    for route in routes:
        print(
            f"  token={route.token} k={route.top_k_slot} logical={route.logical_expert} "
            f"-> global_slot={route.global_slot} rank={route.destination_rank} "
            f"local_slot={route.local_slot}"
        )
    counts = [0] * placement.ep_size
    for route in routes:
        counts[route.destination_rank] += 1
    print("  destination route counts:", counts)
    print("  combine result:", combine(routes))

ep_size = after.ep_size
max_tokens_per_rank = 3
hidden = 4
local_experts = after.slots_per_rank
rank_major_shape = (ep_size, max_tokens_per_rank, hidden)
expert_major_shape = (local_experts, ep_size * max_tokens_per_rank, hidden)
print("\nNCCL EP LL rank-major dispatch shape:", rank_major_shape, "->", rank_major_shape[0] * rank_major_shape[1], "rows")
print("DeepEP LL expert-major rank-0 shape:", expert_major_shape, "->", expert_major_shape[0] * expert_major_shape[1], "rows")
print("rank-0 recv_expert_count for local logical experts (0, 6): [1, 2]")

运行：

```bash
python3 inference-scheduling/examples/tensorrt_llm_eplb_dispatch.py
```

实际输出：

```text
router logical top-k (EPLB does not rewrite these IDs): [[6, 2], [1, 2], [6, 0]]

before EPLB
  token=0 k=0 logical=6 -> global_slot=6 rank=3 local_slot=0
  token=0 k=1 logical=2 -> global_slot=2 rank=1 local_slot=0
  token=1 k=0 logical=1 -> global_slot=1 rank=0 local_slot=1
  token=1 k=1 logical=2 -> global_slot=2 rank=1 local_slot=0
  token=2 k=0 logical=6 -> global_slot=6 rank=3 local_slot=0
  token=2 k=1 logical=0 -> global_slot=0 rank=0 local_slot=0
  destination route counts: [2, 2, 0, 2]
  combine result: [81, 231, 461]

after EPLB
  token=0 k=0 logical=6 -> global_slot=1 rank=0 local_slot=1
  token=0 k=1 logical=2 -> global_slot=2 rank=1 local_slot=0
  token=1 k=0 logical=1 -> global_slot=6 rank=3 local_slot=0
  token=1 k=1 logical=2 -> global_slot=2 rank=1 local_slot=0
  token=2 k=0 logical=6 -> global_slot=1 rank=0 local_slot=1
  token=2 k=1 logical=0 -> global_slot=0 rank=0 local_slot=0
  destination route counts: [3, 2, 0, 1]
  combine result: [81, 231, 461]

NCCL EP LL rank-major dispatch shape: (4, 3, 4) -> 12 rows
DeepEP LL expert-major rank-0 shape: (2, 12, 4) -> 24 rows
rank-0 recv_expert_count for local logical experts (0, 6): [1, 2]
```

这个例子没有启动 GPU、NCCL 或 DeepEP。它验证的是身份和布局映射，不验证真实 collective 的延迟、带宽或数值 kernel。

## 9. 常见错误

**EPLB 后仍用 `expert_id // experts_per_rank` 推 rank。** 症状是新布局下 token 被发到旧 rank，或者访问了错误 local slot。EPLB 后必须查当前 placement。

**把 rank-major 当成 expert-major。** 症状是把 `[ep_size, max_tokens_per_rank, hidden]` 直接当成本地专家连续分组，导致 grouped GEMM 的 expert offset 错误。

**在 dispatch 前改写 top-k 逻辑 id。** 症状是 combine 权重与专家选择不对应。应保留 pre-EPLB logical top-k，另外保存 routed slot。

**只更新 placement，不更新专家权重。** 症状是 shape 和通信都正常，但输出质量异常或与 baseline 不一致。routing metadata 和 packed expert weights 必须是同一版本。

**认为 EPLB 会改变本地 scheduler 的 token budget。** 症状是把负载不均误判成调度器没有调度。scheduler 仍按本地请求和 token 预算走，EPLB 改变的是 rank 间 expert 负载和同步耗时。

**假设空 rank 可以跳过 EP collective。** EP dispatch/combine 是集合通信，零 route 的 rank 也要给出合法 buffer 和计数并参与同步，否则其他 rank 会卡住。

## 10. 自检

**问题 1：EPLB 改变专家位置后，router 的 top-k 需要重写吗？**

答：不需要。top-k 是 logical expert id；dispatch 前用当前 placement 把它映射到 global slot/rank。需要同步的是 placement 和对应专家权重。

**问题 2：为什么不能用 rank 公式从 logical expert 直接推 physical rank？**

答：EPLB 可以搬动专家并创建 replica，logical id 到 global slot 不是固定的连续映射。必须经过 `globalSlotIds` 查询。

**问题 3：NCCL EP 的 rank-major 张量如何进入 grouped GEMM？**

答：rank-major receive buffer 先 flatten，再按 local expert/token slot 重排，生成每个 expert 的连续行和 `expert_first_token_offset`；DeepEP 则可以直接提供 expert-major 输出和 `recv_expert_count`。

**问题 4：EPLB 对 scheduler 最直接的影响是什么？**

答：不改变 priority/token budget/phase 选择；改变 token 被 dispatch 到的 rank、副本选择、每个 rank 的负载和 EP 同步耗时。

**问题 5：这个 checkout 里可以把 LPLB 当成一条确定的调度路径吗？**

答：不能。当前源码中没有独立的 `LPLB` 符号或阶段。能确认的是 EPLB 的 logical → replica/global slot placement。

## 11. 源码路径

- `tensorrt_llm/_torch/moe/fused_moe/moe_scheduler.py:527`：routing 后更新统计并调用 EPLB route。
- 同文件 `:581`：EP dispatch，随后进入 backend/grouped GEMM。
- 同文件 `:692`：combine 回到 home rank。
- `tensorrt_llm/_torch/moe/fused_moe/moe_load_balancer.py:729`：Python `route()` 调用 C++ routing op。
- `cpp/tensorrt_llm/runtime/moeLoadBalancer/moeLoadBalancer.cpp:280`：`prepareGpuPlacementInfo()`。
- 同文件 `:318`：`globalSlotId = rank * slotCountPerRank + slotId`。
- `cpp/tensorrt_llm/kernels/moe/loadBalance/moeLoadBalanceKernels.cu:599`：replica 选择和 physical global slot。
- 同文件 `:622`：`moeComputeRouteDevice()`。
- `tensorrt_llm/_torch/moe/fused_moe/communication/nccl_ep.py:15`：rank-major LL 契约。
- 同文件 `:359`：rank-major buffer flatten。
- `tensorrt_llm/_torch/moe/fused_moe/communication/deep_ep_low_latency.py:441`：expert-major DeepEP 适配。
- `tensorrt_llm/_torch/moe/fused_moe/fused_moe_deepgemm.py:1059`：`moe_permute_op` 与 expert offset。
- `tensorrt_llm/_torch/route_capture.py:178`：捕获 pre-EPLB logical top-k。
