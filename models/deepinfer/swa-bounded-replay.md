# DeepInfer 的 SWA KV 与有界重放

> 根据 [DeepSeek-V4.1-Flash 论文 §3.2.2](https://arxiv.org/html/2609.19969#S3.SS2.SSS2) 和 DeepInfer 源码仓库 `774cc244b87cc0eb8d4b94ab9f5de2ab22049f01` 的 [部署 QA](https://github.com/deepseek-ai/DeepInfer/blob/774cc244b87cc0eb8d4b94ab9f5de2ab22049f01/doc/inference_deployment_qa.md#L129-L191)、[README](https://github.com/deepseek-ai/DeepInfer/blob/774cc244b87cc0eb8d4b94ab9f5de2ab22049f01/README.md#L20-L27) 与 YoCo prefill 实现整理。部署策略和开源运行时开关分开说明；两者相关，但不是同一个配置项。

## 本次讲解位置

```text
章节：models/deepinfer
小节：SWA KV cache、cache miss 与 bounded replay
知识点：主 KV 与 SWA KV 的状态边界；窗口截断公式；重放的近似来源
上次：DeepInfer 执行骨架与 YOCO 的 source / consumer / cache alias
下次：DeepInfer zero-overhead asynchronous scheduling
PTX：不适用
```

## 为什么现在讲这个

前一节区分了 YOCO 的主 KV source/consumer，但这还不足以解释 prefix cache 命中时为什么可能要重算一段 token：主 KV 命中不代表 SWA 的局部状态也在同一个缓存里。把两类 cache 混成一份，会误以为“Global KV 命中就不需要重放”，或反过来误以为“重放会重算并覆盖全部 KV”。本节把缓存归属、精确窗口和近似边界拆开，避免这两种判断错误。

## 1. 心智模型：两条 KV 状态路径

这里沿用我们讨论时的 **Global KV** 说法，指部署文档里的 block KV（main KV 与 indexer K）；它不是 SWA KV 的另一个名字。两者服务于不同的注意力分支，格式、生命周期和缓存策略也不同。

```text
已命中的 prefix
       │
       ├── 主 / block KV（main KV + indexer K）
       │      └── 3FS 持久 prefix cache；执行时按需进入 GPU
       │
       └── SWA 局部状态
              ├── 部分 SWA 状态放在本机 host state cache
              ├── YOCO consumer 的部分 SWA 状态可以丢弃
              └── miss 时可重放 suffix，重建供后续 decode 使用的 SWA 状态
```

因此，“SWA KV 不存”只能在限定语境下成立：它**不进入 3FS 的持久 block-KV prefix cache**；不能据此说系统完全不保存 SWA 状态。部署 QA 描述的实现是在本机 host memory 的 state cache 中保存单份 SWA 状态（V4.1 例子保存前 20 层需要跨请求保留的 SWA）；对可通过 prefill 重算恢复的 YOCO consumer SWA 则可以不存。跨机命中时，SWA 状态可以经部署侧目录定位并从持有它的机器拉回。QA 给出的一个容量口径是 `20 layers × 134 tokens × 528 B = 1,415,040 B` 每序列；这是该模型/部署格式的存储示例，不是任意 SWA 模型都适用的常数。

| 状态 | 主要内容 | 部署中的位置/生命周期 | 能否和另一种互换 |
|---|---|---|---|
| 主 / block KV | main KV、indexer K | 3FS 持久 prefix cache；运行时有 in-flight GPU 数据 | 不能当作 SWA KV |
| SWA KV | 滑动窗口注意力的局部 K/V 状态 | 本机 host state cache 中保存需要持久化的部分；部分 consumer 状态可丢弃并重算 | 不能用主 KV 的命中状态替代 |

## 2. 精确 SWA 窗口与有界重放窗口

令 token 位置从 0 开始，窗口宽度为 `W`，正在计算位置 `i`。完整、精确的因果滑窗是：

$$
J_{\text{exact}}(i)=[\max(0, i-W+1),\ i]
$$

若重放从位置 `s` 开始，SWA 分支被限制在重放片段内，则可见集合变为：

$$
J_{\text{bounded}}(i;s)=[\max(s, i-W+1),\ i], \qquad i\ge s
$$

区间两端均包含。两式的差异只会出现在精确窗口还需要访问 `s` 之前的位置时；但缺失的上下文会先影响重放 token 的 hidden/KV 表示，并可能沿后续层传播。因此，即便窗口区间后来覆盖了完整的 `W` 个位置，重建出的状态也不保证和完整历史精确重算的状态相同。

### 数值例子

设 `W=128`，重放起点 `s=1000`：

| 当前 token `i` | 精确窗口 `J_exact` | 有界重放窗口 `J_bounded` | 直观结果 |
|---:|---|---|---|
| 1000 | `[873, 1000]`，128 个位置 | `[1000, 1000]`，1 个位置 | 重放第一 token 缺少 127 个更早 SWA 位置 |
| 1063 | `[936, 1063]`，128 个位置 | `[1000, 1063]`，64 个位置 | 仍缺少重放起点前的局部上下文 |
| 1127 | `[1000, 1127]`，128 个位置 | `[1000, 1127]`，128 个位置 | 位置集合相同，但此前 token 的表示可能已受截断影响 |

最后一行是容易漏掉的细节：只看 attention 的 key 位置集合会觉得“已经完整了”，但位置 `1000...1126` 的中间表示是在截短的 SWA 历史下计算出来的，所以后面的 K/V 仍可能与精确路径不同。

## 3. 重放时哪些复用，哪些重建

部署 QA 描述的 encoder bounded replay 是：主/block KV cache 决定 prefix 命中长度；若对应 SWA 状态 miss，则把命中 prefix 的末尾一段重新跑过模型，以重建 SWA 状态，再与真正未命中的 suffix 一起继续 prefill。重放段用于重建 SWA；已命中的 block KV 复用原缓存，不重算、不覆盖。换句话说，重放时不会因为缺 SWA 状态就把已命中的主 KV 重新生成一遍；两条 cache 路径独立处理。

```text
命中 prefix:   [0 ........................................ L-1]
主 KV:         [=========== 从 cache 复用，不重写 ===========]
SWA replay:                                      [L-128 ... L-1]
新 suffix:                                                     [L ...]
```

这不是“整个模型只看最近 128 个 token”。重放段里的主 KV 仍按其原有缓存路径使用；被截断的是 **SWA 局部分支**的可见范围。近似误差来自 SWA 分支少看了 replay 起点之前的局部上下文，而不是主 KV 历史被统一裁成 128 个 token。

### Encoder 与 decoder 两处重放

| 场景 | 为什么要重放 | 重放结果 |
|---|---|---|
| Encoder SWA bounded replay | 主 KV prefix hit、SWA cache miss | 只重建 SWA 状态；block KV 保持原缓存不变；与未命中 suffix 一起继续 prefill |
| Decoder SWA bounded replay | CED decoder 侧主/block KV 可由最后 encoder hidden 投影，主要缺的是 decoder 的 SWA 状态 | 每次只重算 prompt 末尾受限 suffix；这份 decoder SWA 状态供当前 decode 使用，不进入 prefix cache |

部署 QA 给出的量级例子是 20 个 decoder 层、SWA 窗口 128：要精确恢复窗口内状态，依赖链可能要求把前面约 `20 × 128 = 2560` 个 token 送过 decoder 层；有界做法只重放最后 128 个 prompt token。按“token × 层”粗略计，二者分别约 `2560 × 20 = 51,200` 与 `128 × 20 = 2,560` token-layer 工作。这个估算用于解释该 CED decoder 场景，不应当当作所有模型结构的固定比例。

## 4. 不接受近似时怎么办

如果要求与原始完整前缀严格一致，就不能丢掉重放起点之前所需的 SWA 状态。可选策略是：

1. **保存并命中完整所需的 SWA 状态**：减少重算，但增加 state cache 容量、写入和跨机传输成本。
2. **精确重建**：保留足够长的历史并按完整模型依赖重算；结果精确，prefill 成本随窗口依赖和层数上升。
3. **bounded replay**：只重放固定长度 suffix，计算量受控，但接受 SWA 状态近似。

因此 bounded replay 是缓存容量、prefill 延迟与数值一致性之间的策略选择，不是“无损恢复”的另一种写法。

## 5. 命中长度与 128-token 回退

部署 QA 给出的 hit policy 是先由 block KV cache 找到候选命中长度 `L1`，再检查 SWA cache：如果所需 SWA 状态存在，则保留 `L1`；若不存在，则回退一个 128-token 单位并重放。YOCO truncation 还要求为窗口和 backtracking 留空间：

$$
L_{\max}=L-(W+B)
$$

其中本例 `W=128`，部署 QA 使用 `B=6`，所以最大 hit 长度为 `L-134`。候选 block-KV hit 长度按 128-token block 对齐；SWA miss 时按策略退一个 block，而不是把两种 cache 当成同一份命中结果。

## 6. 开源开关：和生产 cache-miss 策略分开看

DeepInfer 的开源代码提供 YoCo consumer **prefill truncation** 配置：

| 环境变量 | 值 | 效果 |
|---|---|---|
| `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE` | `off` / `mask` / `compact`，默认 `off` | `off` 不截断；`mask` 保留所有 query 但收窄 SWA 可见域；`compact` 真正移除被截断的 consumer query，减少计算 |
| `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS` | 正整数 | 保留末尾 query token 数；必须不少于 `window_size + DEEPINFER_MAX_BACKTRACKING_TOKENS` |

默认 V4.1 配置示例使用 `compact` 与 `KEEP_TOKENS=140`；实际运行时仍须满足当前模型窗口与 backtracking 的断言。`mask` 和 `compact` 的保留区域可见性相同，但只有 compact 会跳过被移除 query 的计算。仓库的 `examples/yoco/yoco_equiv.py mask-compact` 可先在无权重的单设备元数据路径对比二者；完整数值对齐需使用兼容 checkpoint 与 L2 测试。

重要边界：这两个环境变量控制源码里的 YoCo prefill 截断路径（对应 [环境变量定义](https://github.com/deepseek-ai/DeepInfer/blob/774cc244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/environ.py#L187-L194)）；它们**不是**一键配置“整个生产服务是否存 SWA KV”、3FS 行为或 cache miss 路由。完整生产行为还由部署侧 state cache、目录、回退 hit policy 等组件共同决定。

在 DeepInfer 仓库根目录的元数据检查示例（见 [YoCo 等价性测试说明](https://github.com/deepseek-ai/DeepInfer/blob/774cc244b87cc0eb8d4b94ab9f5de2ab22049f01/examples/yoco/README.md#L1-L18)）：

```bash
python examples/yoco/yoco_equiv.py mask-compact --keep-tokens 140 --window 128
```

预期可观察结果是脚本逐项比较 mask 与 compact 的 SWA pivots、可见 key、store slots 和压缩 key 范围；此 L1 检查不加载模型权重，也不等于验证了有界重放在完整生产服务中的 cache miss 策略。

## 7. 常见误解

| 误解 | 正确理解 | 典型后果 |
|---|---|---|
| “SWA KV 完全不存” | 它不进 3FS 持久 block-KV cache，但部分 SWA 状态存在本机 host state cache；一些 consumer 状态才被丢弃并重算 | 错估状态容量与跨机传输 |
| “Global KV 和 SWA KV 是同一份” | 它们是不同注意力分支的状态，命中和生命周期独立 | 误判主 KV hit 可免除 SWA replay |
| “重放会重算并覆盖 block KV” | replay 主要重建缺失的 SWA 状态，已命中的 block KV 复用 | 错估 replay 写流量及 cache 污染 |
| “整个模型只看 replay 的 128 个 token” | 有界窗口只截断 SWA 分支；主 KV 按自身缓存路径复用 | 把局部近似说成全局上下文截断 |
| “`compact` 等于生产缓存服务的完整 bounded replay 开关” | 它是开源 YoCo prefill query 截断模式；生产缓存路由和状态存储还有部署侧逻辑 | 误以为只设环境变量就复现了全套线上策略 |

## 自测题

1. `Global/block KV` 命中但 SWA 状态 miss，是否一定无需重放？
   - 答：不是。主 KV 与 SWA 状态分开命中；SWA miss 可触发 suffix replay。
2. `W=128, s=1000, i=1000` 时两个窗口分别是什么？
   - 答：精确窗口 `[873,1000]`；有界窗口 `[1000,1000]`。
3. 为什么到 `i=1127` 即使位置集合都是 `[1000,1127]`，重建状态仍可能不精确？
   - 答：更早 replay token 的表示已在缺少 `s` 之前 SWA 上下文时计算，误差可沿层传播。
4. `mask` 和 `compact` 在可见范围上的区别是什么？
   - 答：保留范围相同；mask 仍计算全部 query，compact 移除被截断的 consumer query 以减少计算。
5. 想要严格精确恢复缺失的 SWA 状态，主要有哪两类办法？
   - 答：保存所需 SWA 状态并命中，或保留足够上下文做精确重建；两者都比 bounded replay 消耗更多存储或计算。

## 本节小结

```text
主/block KV 与 SWA KV 是不同状态
  -> 主 KV hit 不保证 SWA hit
  -> SWA miss 时可以 suffix replay 重建
  -> bounded replay 把 SWA 可见范围限制在 replay 起点之后
  -> 节约计算，但缺失上下文让重建状态成为近似
```

下一节转向另一个 DeepInfer 性能点：zero-overhead asynchronous scheduling，它优化 CPU/GPU 工作重叠，不改变上述 KV 正确性边界。
