# SGLang DP attention 的混合 P/D batch 与 prefill delayer

> 源码快照：SGLang `af9bb2a32d4a313d20a629ac1b2cb8e6ef245b95`（2026-10-19）。
> 本文讨论 attention DP 下本地调度与跨 rank 执行的边界，重点解释 `--enable-prefill-delayer` 为什么能减少空泡，以及 EP dispatch/combine 为什么要求每个 rank 继续参与。

```text
本次讲解位置
章节：inference-scheduling
小节：SGLang DP attention 的混合 P/D batch 与 prefill delayer
知识点：跨 DP rank collective、EP dispatch 与有界 prefill 延迟
上次：vLLM V1 的两本 token budget
下次：TensorRT-LLM 的调度与 attention DP
PTX：不适用
```

## 为什么现在讲这个

vLLM V1 的 token budget 解释了单个 scheduler 如何把 prefill chunk 和 decode 折算成每轮输入 token，但它默认有一个全局 scheduler 来形成 batch。SGLang 的 attention DP 不是这样：每个 DP rank 维护自己的 running/waiting 请求和本地 batch，attention 也只在本地序列上执行；到 MLP、TP 或 MoE 时，各 rank 又进入同一条跨 rank 执行路径。

这会产生一个仅看 token budget 看不出的约束：rank 0 可能在做 64-token prefill，rank 1 只在做 2-token decode，而它们在下一个 collective 必须一起到达。EP dispatch/combine 更是按固定通信顺序交换 token，某个 rank 若因为没有本地请求就完全跳过 forward，其他 rank 会等待或直接死锁。

`--enable-prefill-delayer` 是一个吞吐优化，正确性不依赖它：当只有部分 rank 能 prefill 时，先压住新 prefill，让各 rank 的本地工作更接近，或者在短时间内等到更多 rank 同时具备 prefill；等到超时后再放行，以限制 TTFT 恶化。下面把这个策略放回 scheduler、forward metadata gather 和 EP 数据路径中解释。

## 1. 心智模型：本地 scheduler，全局 forward

可以先把一个 DP attention step 拆成三段：

```text
每个 DP rank 本地调度
  -> attention 在本地 request/sequence 上执行
  -> MLP / TP / MoE 跨 DP rank 汇聚 token
  -> collective 或 all-to-all 完成后，各 rank 取回自己的结果
```

“本地”和“全局”的边界很重要：

| 阶段 | 数据作用域 | 是否允许不同 rank 有不同本地 batch |
|---|---|---|
| Scheduler 选 batch | 每个 DP rank 独立 | 允许，rank 可有 prefill、decode、混合 batch 或空 batch |
| Attention | 本 DP rank 的序列 | 允许行数和模式不同；attention TP 在 DP group 内另行同步 |
| MLP / MoE gather | 跨 DP rank 的 global buffer | 逻辑行数可不同，但所有 rank 必须按同一协议调用 collective |
| EP dispatch/combine | 跨 EP rank 的 token/expert 流 | 每个 EP rank 必须参与同一轮 dispatch/combine；空 rank 也要发零 token |
| Sampling / 结果散回 | 各 rank 的本地请求 | 恢复成本地 request ownership |

因此，“各 rank 的请求模式可以不同”和“各 rank 必须参与同一个 global forward”并不矛盾。前者属于请求调度，后者属于模型执行契约。

## 2. SGLang 如何选择 prefill 或 decode

`get_next_batch_to_run()` 主路径很直接：先尝试构造新 prefill batch；只要 `new_batch` 非空，本轮就运行它，否则再更新并运行已有的 decode batch。源码见 `srt/managers/scheduler.py:3909-3947`：

```python
prefill_plan = self.get_new_batch_prefill(running_batch)
new_batch = prefill_plan.batch_to_run
if new_batch is not None:
    # Run prefill first if possible
    ret = new_batch
else:
    ret = self.update_running_batch(running_batch)
```

这里的 prefill-first 只约束本地 scheduler 的选择顺序。SGLang 还有 chunked prefill、mixed chunk 和 DP attention 的 global metadata 协商；具体能否把两类 batch 合成一个可执行 batch，要看相关开关和后端能力。

在这条主路径之外，有三层节流：

### 2.1 `--prefill-decode-interval`

`_arm_prefill_decode_interval()` 在本轮 batch 是 extend 时，把剩余计数器设为 `prefill_decode_interval`。之后的 `_should_defer_prefill()` 每调用一次就减一，并在计数非零时禁止构造新 prefill。

```text
extend step
  -> 接下来 N 轮跳过新 prefill，优先推进 decode
  -> 计数器归零后，重新尝试 prefill
```

DP attention 下它用 gather 后的 `batch.is_extend_in_batch` 判断本轮是否存在 extend，因此各 rank 同步同一条 cadence。否则某个 rank 本地有 prefill 就重置间隔，另一个 rank 没有就会错位。

### 2.2 `--enable-prefill-delayer`

delayer 先 gather 各 rank 的“现在是否有可 prefill 请求”、KV token 使用率、running batch、最大 prefill batch 和 waiting queue 长度。源码见 `prefill_delayer.py:183-217`。

依据 `global_prefillable` 分成三种状态：

| 全局状态 | 含义 | 默认动作 |
|---|---|---|
| `none` | 没有 rank 可 prefill | 不构造 prefill，继续 decode |
| `all` | 所有 rank 都可 prefill | 通常放行；也可因 batch 太碎或 waiting queue 太短而短暂等待 |
| `mixed` | 只有部分 rank 可 prefill | 在 `max_delay_passes` 范围内压住 prefill，继续跑 decode |

`mixed` 分支的核心条件是 `prefill_delayer.py:323-349`：

```text
prev_delayed_count < max_delay_passes - 1
  -> 本轮 output_allow=False
否则
  -> output_allow=True, reason=wait_timeout
```

延迟有硬上界，不会为了等待“完美对齐”无限拖住请求。低 KV 使用率还有 watermark 旁路：GPU 明显空闲时不做这种延迟，因为此时等来的对齐收益很小。

这一机制针对的不是单个 rank 内部的 prefill/decode 选择，而是跨 rank 的到达时间差。rank 0 在 t=0 有 prefill，rank 1 在 t=1 才有，rank 2 在 t=2 才有；若立刻执行，三个 global step 交替出现“一个 rank 重 prefill、两个 rank 轻 decode”。若延迟窗口能覆盖这 2 个 step，则前两步都跑 decode，第三步三个 rank 一起 prefill，collective 两侧的工作量更接近。

### 2.3 `all` 状态下的批聚合

即使所有 rank 都可 prefill，delayer 也可能根据 `running_batch` 和剩余 slot 判断“现在开 prefill 会把 batch 切得太碎”。queue trigger 则等待 waiting queue 达到由 running 请求数推导出的阈值，或者等到 wall-clock timeout。这两条路径主要改善小 prefill 频繁插入造成的低利用率，同时用超时保护尾延迟。

## 3. DP attention 的同步与数据路径

### 3.1 Scheduler 先交换每 rank 的 forward metadata

`prepare_mlp_sync_batch_raw()` 在 `srt/managers/scheduler_components/dp_attn.py:518` 开始。它把以下信息打包成定长 tensor，再在 DP 维度上 all-gather：

- 本地 token 数，以及需要 logprob 的 token 数；
- 本地 forward mode 和 `is_extend_in_batch`；
- decode/prefill/draft CUDA graph 是否可运行；
- TBO 兼容性、prefill graph 的最大 prefix 长度；
- 是否 health check；
- causal attention pairs，用于 DP 平衡统计。

源码中的字段定义见 `MLPSyncBatchInfo`（`dp_attn.py:145-193`）。gather 后，所有 rank 得到同一份 `global_num_tokens`、`global_forward_mode` 和各 rank 的 graph 能力，再据此选择本轮能共同执行的 forward 形式。

本地没有 batch 时，只要其他 rank 有 token，本 rank 就调用 `get_idle_batch()` 生成 idle batch：

```text
max(global_num_tokens) > 0
  -> 即使 local_batch is None，也构造 idle batch
  -> 该 rank 继续进入 MLP sync / collective / CUDA graph 路径
```

对应 `dp_attn.py:662-684`。idle batch 只携带零 token 参与一个仍然存在的 global step，不会伪造用户请求。

### 3.2 Attention 本地执行，MLP/MoE 使用 global token buffer

Attention 完成后，DP rank 的 token 被写入一个跨 rank 的 global buffer。常见顺序是 rank-major：

```text
global rows =
  [DP rank 0 的 token 行]
  [DP rank 1 的 token 行]
  ...
  [DP rank R-1 的 token 行]
```

实现会在两种布局中选择（`srt/layers/dp_attention.py:143-192`）：

| 模式 | buffer 形状 | 通信 | 特点 |
|---|---|---|---|
| `MAX_LEN` | `R * max(local_len)` | all-gather，短 rank 补 padding | 形状整齐，适合 symmetric memory/CUDA graph；有 padding 计算 |
| `SUM_LEN` | `sum(local_len)` | 零 buffer + all-reduce | 不按最大 rank 放大；形状随全局 token 数变化 |

`MAX_LEN` 下 rank `r` 的真实 token 位于 `[r * max_len, r * max_len + local_len[r])`。短 rank 的 padding 行仍可能被 router 读到，所以后续会把 pad 行的 top-k 标成 `-1`，避免它们进入真实 expert 计算。`SUM_LEN` 下 rank `r` 从 `sum(local_len[:r])` 开始写入，再做 all-reduce 合并。

无论选哪种模式，所有 DP rank 都必须执行同一轮 gather。全局 token 数为 0 时可以直接 idle，但只要至少一个 rank 有 token，空 rank 也不能跳过。

### 3.3 EP dispatch/combine 为什么要求继续同步

MoE 路由器产生的是逻辑 expert top-k。EP dispatch 根据当前 EP 布局把 token 发往持有目标专家的 rank，接收端通常按 expert-major 重新组织 token；compute 完成后再按原 token 顺序 combine 回来。

这里有三条不变量：

1. router 的 top-k 是逻辑专家身份，EPLB 改变物理专家位置时，必须经过逻辑到物理的映射，而不是把旧物理 rank 直接写回 top-k。
2. 每个 EP rank 的 dispatch/combine 必须按同一轮、同一协议发起。即使本地 token 为 0，也要用零 token 或定容空槽参与，避免对端等待一个永远不会到来的消息。
3. combine 必须恢复每个 token 的原 rank 和原行位置；否则 rank-major 输入经过 expert-major 计算后无法正确散回本地 request。

rank-major/expert-major 的布局和 EPLB 的逻辑/物理映射可分别参考：

- [`communication/23-deepep-rank-major-vs-expert-major-dataflow.md`](../communication/23-deepep-rank-major-vs-expert-major-dataflow.md)
- [`communication/28-deepep-eplb-logical-physical-expert-map.md`](../communication/28-deepep-eplb-logical-physical-expert-map.md)

这也是前面那个问题的答案：expert 计算本身不区分 prefill 或 decode，但同一个 global step 中的 rank 要共同完成 dispatch/combine。一个 rank 单独插入很重的 prefill，其他 rank 要么在 collective 前等它，要么只能执行很轻的 decode，于是等待时间变成 DP attention 空泡。prefill delayer 试图让各 rank 要么都在推进 decode，要么在更接近的时刻一起提交 prefill。

## 4. 数值 trace：同样的工作量，放在哪里不同

下面的模型只模拟调度结构，不是 SGLang 的真实性能测量。三个 rank 各有一个 64-token prefill，分别在 step 0、1、2 就绪；prefill 的 attention pair 数按无 prefix 的完整 causal chunk 计算：

```text
prefill_pairs = 64 * (64 + 1) / 2 = 2080
decode_pairs  = 2 rows * 64 context = 128
idle_units    = max(local_work) * num_ranks - sum(local_work)
```

`idle_units` 衡量一个 global step 中被最慢 rank 拖住的工作量，不是实际秒数。实际空泡还要乘上 kernel/通信时间，并受 attention pair、expert 分布和网络影响。

### 4.1 就绪即 prefill

| step | rank 0 | rank 1 | rank 2 | global tokens | max work | idle units |
|---:|---|---|---|---:|---:|---:|
| 0 | P 2080 | D 128 | D 128 | 68 | 2080 | 3904 |
| 1 | D 128 | P 2080 | D 128 | 68 | 2080 | 3904 |
| 2 | D 128 | D 128 | P 2080 | 68 | 2080 | 3904 |

总工作量为 `3 * (2080 + 128 + 128) = 7008`，总 idle units 为 `11712`。每个 step 都有一个 rank 做重 prefill，另外两个 rank 只做轻 decode。

### 4.2 有界延迟到全部就绪

`max_delay_passes=3` 时，前两步处于 `mixed` 状态，delayer 不让任何一个 rank 开新 prefill；各 rank 先推进 decode。第三步三个 rank 全部可 prefill，状态变为 `all`，一起放行。

| step | rank 0 | rank 1 | rank 2 | global tokens | max work | idle units |
|---:|---|---|---|---:|---:|---:|
| 0 | D 128 | D 128 | D 128 | 6 | 128 | 0 |
| 1 | D 128 | D 128 | D 128 | 6 | 128 | 0 |
| 2 | P 2080 | P 2080 | P 2080 | 192 | 2080 | 0 |

两个方案的总 token 工作量相同，但第二种把三个重 prefill 放进同一个 global step，collective 两侧不再一边 2080、一边 128。代价是 rank 2 的 prefill 被推迟到 step 2 才执行，前两步仍消耗 decode 时间；真实 workload 中还会增加 TTFT，并可能让 decode 被 prefill 抢占。因此 `max_delay_passes` 是收益和尾延迟之间的上界，不是越大越好。

## 5. 可运行 artifact

完整代码见 [`examples/sglang_dp_prefill_delayer.py`](examples/sglang_dp_prefill_delayer.py)。它实现了上面的 `mixed/all/none` 状态判断、有界延迟和每 step 的本地 work/idle units 计算。

运行：

```bash
python3 inference-scheduling/examples/sglang_dp_prefill_delayer.py
```

预期输出：

```text
without delayer
step status allow rank_modes            local_work             global_tokens idle_units
0    mixed  yes   P:2080 D:128 D:128     2080/128/128           68            3904
1    mixed  yes   D:128 P:2080 D:128     128/2080/128           68            3904
2    mixed  yes   D:128 D:128 P:2080     128/128/2080           68            3904
total_idle_units=11712

with delayer, max_delay_passes=3
step status allow rank_modes            local_work             global_tokens idle_units
0    mixed  no    D:128 D:128 D:128      128/128/128             6             0
1    mixed  no    D:128 D:128 D:128      128/128/128             6             0
2    all    yes   P:2080 P:2080 P:2080   2080/2080/2080         192            0
total_idle_units=0
```

示例里“delayer 为 no”时仍让三个 rank 执行 decode，对应 SGLang 在延迟 prefill 时继续推进已有 running batch 的路径。真实实现还会结合 token watermark、running slot、waiting queue 和 CUDA graph 条件；示例没有模拟这些旁路。

## 6. 常见错误

**把 prefill delayer 当成 token budget。** 它不改变每轮 token 上限，也不合并请求；它只决定本地 scheduler 这一轮是否构造新 prefill。预算仍是 chunked prefill、KV cache 和 batch 容量共同决定的。

**认为 rank 没有本地请求就可以跳过 forward。** 只要其他 rank 还有 token，空 rank 仍要以 idle batch 或零 token 参与 MLP/EP collective。跳过会表现为 collective hang、EP dispatch 等消息超时，而不是简单的“少算一个 rank”。

**认为所有 rank 必须拥有相同的 prefill/decode 比例。** 请求归属是本地状态，delayer 也不能凭空给 rank 分配请求。它只能推迟新工作。真正的负载均衡还依赖请求路由、KV 使用率、prefix cache 命中和 expert 分布。

**EPLB 后继续使用旧物理 expert id。** top-k 的语义应是逻辑专家；EPLB 改变的是逻辑 expert 到物理 slot/rank 的映射。若 router 输出仍被解释成旧物理位置，token 会发到错误专家，症状可能是输出质量下降而不是显式崩溃。

**把 `idle_units=0` 当成没有性能损失。** 本 trace 只说明各 rank 的 attention work 对齐，不包含 decode 尾部延迟、网络抖动、expert 热点和 collective 实际耗时。prefill delayer 改善的是执行重叠机会，不是消除所有 DP 空泡。

## 7. 自检

**问题 1：SGLang 主路径里 `new_batch is not None` 时为什么先运行 prefill？**

答：这是本地 scheduler 的优先级选择，先处理新到达或尚未完成的 prefill；没有可运行 prefill 时才更新 running decode batch。DP attention 下它还会先 gather global metadata，决定本轮能否共同执行。

**问题 2：`--enable-prefill-delayer` 在 `mixed` 状态做了什么？**

答：部分 rank 可 prefill、部分不可时，它按 `max_delay_passes` 延迟所有 rank 的新 prefill，让已有 decode 先推进，并等待更多 rank 进入可 prefill 状态。超过上限就放行，避免无限拖延 TTFT。

**问题 3：为什么 attention 可以本地做，后面却必须跨 rank 同步？**

答：attention 只访问本 DP rank 拥有的序列和 KV；MLP/MoE 需要把各 rank 的 token 汇总成 global buffer，EP 还要按 expert 目标做 all-to-all 和 combine。数据作用域从本地变成全局，collective 的调用顺序也就变成所有 rank 的共同契约。

**问题 4：EPLB 改变专家位置后，top-k 还能关联到原专家吗？**

答：能，但前提是 top-k 保留逻辑 expert id，并在 dispatch 前应用当前逻辑到物理映射，combine 后再按原 token 位置还原。若把 router 输出直接当成物理专家槽，EPLB 后就会错位。

**问题 5：示例中延迟后总 idle units 从 11712 变成 0，是否说明吞吐一定提升？**

答：不能直接下结论。示例只计算了 attention work 的对齐程度，没有测真实 kernel、通信和 decode 尾延迟。延迟 prefill 还会增加 TTFT，并可能让 decode 占掉原本用于 prefill 的算力；实际收益要在目标模型、并行布局和请求分布上测量。
