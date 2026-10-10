# DeepEP 与 EPLB：对称物理槽 + 框架侧 logical→physical

> 承接 [27](./27-deepep-ep-train-infer-and-backward-dual.md)。DeepEP 按 `expert_id // experts_per_rank` 定 rank，看起来「专家对称切在各卡」。那推理/训练做 **EPLB 搬专家权重** 会不会把 DeepEP 算错？答案：DeepEP 契约不变；框架必须在进 EP 前把 gate 的**逻辑 id** 改成**物理槽 id**，并保证权重与表一致。本文钉死 DeepEP 假设、错误做法，以及 DeepSeek EPLB / SGLang / vLLM 的共同解法。
>
> DeepEP 源码：`dispatch.cuh` / `hybrid_dispatch.cuh`（整除路由）、`elastic.py`（recv stats）。框架侧：[deepseek-ai/EPLB](https://github.com/deepseek-ai/EPLB)、SGLang `srt/eplb/`、vLLM `distributed/eplb/`。

```text
本次讲解位置
章节：communication / DeepEP × EPLB
小节：物理对称切分；logical↔physical；框架 remap + 搬权重
知识点：DeepEP 只认物理 id；EPLB 在上层；冗余副本仍保持每卡槽数相等
上次：EP 训练/推理与反传对偶（27）
下次：AGRS session，或按需深挖某框架 EPLB 执行路径
原语：dst_rank = expert_id // E；phy2log / log2phy；rebalance_experts
```

## 为什么现在讲这个

只读 DeepEP 会得到「专家必须按 rank 对称」的印象，于是误以为 **EPLB 不能搬权重**，或以为 **DeepEP 会跟着权重自动改路由**。
实际上：对称的是**物理槽位布局**；逻辑专家放哪个槽、热专家复制几份，是框架的表。搞混这两层，会出现「权重在 A 卡、token 飞到 B 卡」的静默错误。

---

## 1. DeepEP 的契约：只认物理专家 ID

直连：

```104:104:deep_ep/include/deep_ep/impls/dispatch.cuh
            const auto dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerRank : -1;
```

Hybrid：先 scaleout 再 scaleup，同样是对 **传入的 expert id** 做整除（`hybrid_dispatch.cuh`）。
构造侧要求 `num_experts % num_ranks == 0`，每卡 `num_experts_per_rank` 相同。

```text
物理专家 ID 空间（DeepEP 眼里）：
  rank0: [0, E)
  rank1: [E, 2E)
  ...
  dst_rank = physical_expert_id // E
```

`cumulative_local_expert_recv_stats`（`elastic.py`）只是**监控**本卡各 local expert 收到多少 token，给线上 load balance **决策**用，**不改**上述公式。

---

## 2. EPLB「搬权重」会不会影响 DeepEP？

| 做法 | 结果 |
|---|---|
| 只搬权重，仍把**逻辑** id 传给 `dispatch` | **错**：token 按旧对称公式飞，与权重所在卡不一致 |
| 框架维护 `logical → physical`，改写 `topk_idx` 再 `dispatch` | **DeepEP 公式不受影响**；对称切的是物理槽 |
| 搬完后每卡专家数不等、或 id 不按块连续 | **当前 DeepEP 不支持** |

```text
gate → 逻辑 expert
         ↓  EPLB remap（框架）
       物理 expert id = topk_idx
         ↓  DeepEP
       rank = physical_id // E
```

一句话：**搬权重本身不改 DeepEP 数学；必须让 `topk_idx` 始终指向权重当前所在的物理槽。**

---

## 3. 推理框架怎么解决：共同套路

框架**不改** DeepEP 整除公式，在上层做三件事：

1. **扩物理槽**：`num_physical = num_logical + num_redundant`（且能被 `ep_size` 整除）
2. **算放置**：`physical_to_logical`（每槽装哪个逻辑专家；热点可多副本）
3. **每次 forward 改写路由**：`topk_logical → topk_physical` 再进 A2A

重平衡时再：

```text
统计逻辑专家负载（滑动窗口）
  → policy.rebalance → 新 phy2log
  → 按旧/新 map 跨 rank 搬 expert 权重
  → 原子切换 routing 表
```

```text
┌─────────────┐     log→phy      ┌─────────────┐     // E      ┌──────────┐
│ gate 逻辑 id │ ───────────────► │ 物理槽 id    │ ─────────────► │ DeepEP   │
└─────────────┘   (框架查表)      └─────────────┘   (对称邮差)   └──────────┘
                                        ▲
                                        │ 权重按 phy2log 放在对应卡的槽上
```

---

## 4. 各家落地（算法同源，集成点不同）

### 4.1 DeepSeek EPLB（算法库）

仓库：[deepseek-ai/EPLB](https://github.com/deepseek-ai/EPLB)。

- 入口：`rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus)`
- 输出：`phy2log`、`log2phy`、`logcnt`（每逻辑专家几个物理副本）
- 策略：冗余复制热点 → heuristic pack 到 GPU；若 `num_groups % num_nodes == 0` 用**分层**（先 node 再 GPU），减轻跨机；否则 **global** pack（decode 大 EP 常见）

算法只管「怎么摆」；负载怎么估（历史滑动平均等）在部署侧。

### 4.2 SGLang

- 开关：`--enable-eplb`
- 元数据：`ExpertLocationMetadata`（`physical_to_logical_map`、`logical_to_all_physical_map` 等）
- Dispatch：`expert_location_dispatch.py` 把逻辑 topk 映射成物理 id（static / dynamic 等）
- A2A：可接 DeepEP（`--moe-a2a-backend deepep` 等）；DeepEP 只吃映射后的物理 id
- 文档： [Expert Parallelism](https://docs.sglang.io/docs/advanced_features/expert_parallelism)

### 4.3 vLLM

- 开关：`--enable-eplb`，`--eplb-config`（`num_redundant_experts`、`step_interval`、`window_size` 等）
- 状态：`EplbState` / `physical_to_logical_map` / `logical_to_physical_map`
- 执行：`rearrange_expert_weights_inplace` 按旧/新 map 搬权重（NCCL / nixl 等 communicator）
- 文档：[Expert Parallel Deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)

| | DeepSeek EPLB | SGLang | vLLM |
|---|---|---|---|
| 角色 | 放置算法 | 服务集成 + dispatch 改写 | 服务集成 + 权重搬迁 |
| 冗余专家 | `num_replicas` | `ep_num_redundant_experts` 等 | `num_redundant_experts` |
| 进 DeepEP 的 id | （由集成方决定） | **物理** | **物理**（若用兼容 EP 后端） |

---

## 5. 和训练反传的衔接

反传仍是 [27](./27-deepep-ep-train-infer-and-backward-dual.md) 的 dispatch↔combine 对偶，用的是**同一套物理 handle**。
EPLB 若在 step 间换表：必须保证 **前向 remapped 的物理路由** 与 **当前权重放置** 一致；训练框架通常在同一张表下完成 forward/backward，再择机 rebalance。

---

## 6. 可运行小验证：逻辑→物理→rank

文件：`communication/28_eplb_logical_physical_sim.py`。

```python
#!/usr/bin/env python3
"""Simulate EPLB logical→physical remap vs naive logical dispatch."""

from __future__ import annotations


def rank_of(expert_id: int, experts_per_rank: int) -> int:
    return expert_id // experts_per_rank


def main() -> None:
    # 4 logical experts, +4 redundant → 8 physical slots, 4 ranks → E=2
    experts_per_rank = 2
    # physical slot -> logical expert (hot logical 0 replicated on slots 0 and 4)
    phy2log = [0, 1, 2, 3, 0, 1, 2, 3]
    # pick first physical replica for each logical
    log2phy = {}
    for phy, log in enumerate(phy2log):
        log2phy.setdefault(log, phy)

    logical_topk = [0, 3]  # gate output
    print("naive: pass logical ids to DeepEP (WRONG after move):")
    for lid in logical_topk:
        print(f"  logical {lid} -> rank {rank_of(lid, experts_per_rank)} (thinks weight at old place)")

    print("framework: remap then DeepEP (CORRECT):")
    for lid in logical_topk:
        pid = log2phy[lid]
        print(f"  logical {lid} -> physical {pid} -> rank {rank_of(pid, experts_per_rank)}")

    print("physical layout still symmetric: ranks",
          [rank_of(p, experts_per_rank) for p in range(len(phy2log))])


if __name__ == "__main__":
    main()
```

运行：

```bash
python3 communication/28_eplb_logical_physical_sim.py
```

期望：

```text
naive: pass logical ids to DeepEP (WRONG after move):
  logical 0 -> rank 0 (thinks weight at old place)
  logical 3 -> rank 1 (thinks weight at old place)
framework: remap then DeepEP (CORRECT):
  logical 0 -> physical 0 -> rank 0
  logical 3 -> physical 3 -> rank 1
physical layout still symmetric: ranks [0, 0, 1, 1, 2, 2, 3, 3]
```

（示例里逻辑 3 的物理槽仍在 rank1；若 `phy2log` 把逻辑 3 摆到 slot 6，则 remap 后会去 rank3，而 naive 仍去 rank1——这就是 bug 形态。）

---

## 7. 常见误判

| 误判 | 实际 |
|---|---|
| EPLB 改了 DeepEP 的整除公式 | 公式不变；改的是传入的 id 与权重放置 |
| 对称切分禁止搬专家 | 禁止的是「非均匀物理槽」；搬的是槽里装谁 |
| `recv_stats` 会自动改路由 | 只统计，供框架决策 |
| 逻辑 id == 物理 id | 有冗余/重排后一般不等；必须查表 |
| 权重搬完、表未切换也没事 | 会发错卡；要原子切换 |

---

## 8. 自测题

1. **DeepEP 用哪条公式定 rank？**
   答：`physical_expert_id // num_experts_per_rank`（hybrid 再拆 scaleout/scaleup）。

2. **只搬权重、不改 `topk_idx` 会怎样？**
   答：token 飞到「逻辑 id 对应的对称 rank」，与权重错位。

3. **冗余专家后 DeepEP 为何仍能用？**
   答：物理槽总数仍被 `ep_size` 整除，每卡槽数相等。

4. **SGLang/vLLM 在哪一步改写 id？**
   答：gate 之后、A2A/DeepEP 之前的 logical→physical dispatch。

5. **DeepEP 库内有没有 EPLB 放置算法？**
   答：没有；只有可选 recv 统计；算法在 deepseek-ai/EPLB 与各推理框架。

---

## 9. 学习进度

- [x] EP 训练/推理与反传对偶（27）
- [x] EPLB × DeepEP：逻辑/物理分层与框架解法
- [ ] AGRS session（若继续 Elastic）

### 下一知识点

`create_agrs_session` / AGRS，或深入某一框架的 `rearrange_expert_weights` 执行序。
