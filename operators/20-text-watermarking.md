# Text Watermarking：保持采样分布的文本溯源

## 本次讲解位置

```text
章节：operators / 推理系统
小节：SGLang Text Watermarking
知识点：Aaronson-Gumbel keyed sampling、MurmurHash3 哈希契约、Gamma 检测统计量
上次：RadixAttention 的前缀缓存、paged KV 与 attention 计算的关系
下次：speculative decoding 下 forced-token acceptance 与 watermark 的真实开销
PTX：本节不是指令课；关注采样逻辑、密钥哈希与统计检测
```

### 为什么现在讲这个

上一轮讨论 RadixAttention 时，问题是如何复用已算好的 KV，减少重复 prefill。Text watermarking 处理的是另一个环节：文本离开模型服务之后，如何判断它是否带有某个部署密钥对应的生成痕迹。

两者位于推理链路的不同阶段：

- watermark 不缓存 KV，也不替代 Radix tree；
- 它不会给文本加可见标签，也不会在响应末尾附加 metadata；
- 在理想独立均匀哈希模型下，它保持每个位置的边际采样分布；
- 它改变随机数的来源：普通采样使用 RNG，水印采样使用 `key + context + token_id` 的哈希结果。

若把它理解成“给某些 token 加权”，会误以为水印必然降低生成质量，或以为检测时能寻找某个固定 token。实际信号来自大量上下文上的统计偏移，检测器据此做假设检验。

---

## 1. 它到底解决什么问题

普通采样生成的文本不包含可稳定复现的来源信息。即使重新运行同一模型，采样随机性也会让输出变化；模型更新后更难对齐。

水印希望支持这样的流程：

```text
服务端用密钥生成文本
        ↓
文本被复制、转发或发布
        ↓
检测方拿到文本的 token IDs 和密钥
        ↓
判断这段文本是否与密钥对应的水印统计相符
```

典型用途是来源核验、泄漏追踪、生成内容审计，以及监管场景中的机器可读标记。它检测的是**统计关联**，不是密码学签名。

因此，低 p-value 的含义是：

```text
“这段文本与这个密钥对应的水印相符”
```

而不是：

```text
“这段文本一定由某个模型生成”
“这段文本一定由某个人生成”
```

---

## 2. 心智模型：一段看不见的统计序列号

可以把水印理解成：模型仍然按原来的概率大小抽签，但每个候选 token 拿到的随机号码由密钥决定。

```text
普通采样：
  候选 token 的概率 p_i
  用 RNG 抽一个结果

水印采样：
  候选 token 的概率 p_i
  用 key、前文、候选 token 算一个 u_i
  根据 p_i 和 u_i 选一个结果
```

单个位置看不出异常，单个 token 也不能证明什么。检测器利用的是整段文本中大量位置的共同偏移。

与 RadixAttention 的关系也要分开看：

- watermark 决定当前 token 选谁；
- Radix tree 根据实际 token 序列匹配可复用的 KV 前缀；
- paged attention 根据 KV 页映射读取历史 KV。

水印不直接修改 attention kernel。它可能间接改变前缀命中情况，因为不同采样结果会形成不同的 token 序列，但 KV 本身仍然是正常生成的实际 token 的 KV。

---

## 3. 生成路径

一次生成的数据流可以写成：

```text
prompt
  ↓
模型 forward 得到 logits
  ↓
频率/存在/重复惩罚、logit bias、语法 mask
  ↓
temperature、top-k、top-p、min-p
  ↓
重新归一化得到 p_i
  ↓
watermark eligibility 检查
  ├─ 不满足：普通采样
  └─ 满足：keyed selector
  ↓
选出 token
  ↓
把 token 追加到上下文
  ↓
为该 token 生成并保存 KV
  ↓
继续下一步
```

最后一步保存的是实际被选中 token 的 KV。水印选择只发生在采样阶段；KV 上没有特殊标记，attention kernel 也不需要知道密钥。

---

## 4. 选择规则

设当前位置的候选 token 为 `i`，经过所有采样变换后的概率为 `p_i`。对于每个 `p_i > 0` 的候选，计算：

```text
u_i = Hash(key, context, token_id_i)
```

再把 `u_i` 当作 `(0, 1)` 内的均匀随机数，选择：

```text
selected = argmax_i  log(u_i) / p_i
```

语法 mask 掉或截断掉的 token 有 `p_i = 0`，不能参与选择。精确并列时取最小 token ID。

### 为什么它不改变采样概率

令：

```text
E_i = -log(u_i) / p_i
```

若各候选的 `u_i` 在 `(0, 1)` 上独立均匀，`-log(u_i)` 服从速率为 1 的指数分布。再除以 `p_i`，`E_i` 就服从速率为 `p_i` 的指数分布。

独立指数分布中，事件“第 i 个到达”的概率是：

\[
P(E_i \text{ 最小}) = \frac{p_i}{\sum_j p_j}
\]

而概率已经归一化，`\sum_j p_j = 1`。所以：

\[
P(\text{选中 token } i) = p_i
\]

因此，选择器保持边际采样分布。每个候选的随机数都与密钥和上下文绑定，但 token 被选中的边际概率仍是 `p_i`。

### 一个具体位置

参考实现使用：

```text
probabilities = [0.7, 0.2, 0.1]
context = [11, 12, 13, 14]
key = 0x0123456789abcdef
```

脚本算出的中间值：

| token | p_i | u_i | log(u_i) / p_i |
|---:|---:|---:|---:|
| 0 | 0.7 | 0.021917 | -5.457851 |
| 1 | 0.2 | 0.454224 | -3.945826 |
| 2 | 0.1 | 0.464096 | -7.676643 |

因为 `-3.945826` 最大，选中 token 1。这个位置的概率是 0.2，所以单次选中 token 1 并不奇怪；水印检测依赖大量位置上的统计偏移。

---

## 5. 哈希契约

SGLang 文档给出的哈希计算使用 MurmurHash3 x86 32-bit body mix 和 finalizer。所有 32 位运算都需要回绕。

```text
mix(state, value):
  value = value * 0xcc9e2d51
  value = rotl32(value, 15)
  value = value * 0x1b873593
  state = rotl32(state XOR value, 13)
  return state * 5 + 0xe6546b64

fmix32(value):
  value = value XOR (value >> 16)
  value = value * 0x85ebca6b
  value = value XOR (value >> 13)
  value = value * 0xc2b2ae35
  return value XOR (value >> 16)
```

密钥是 1 到 16 个十六进制数字，解释成 64 位无符号整数，再拆成 `key_lo` 和 `key_hi`。对于长度为 `L` 的上下文 `c_1 ... c_L` 和候选 token `t`：

```text
context_hash = fmix32(
    mix(...mix(mix(0, c_1), c_2)..., c_L)
    XOR (4 * L)
)

token_hash = fmix32(
    mix(mix(mix(mix(0, key_lo), key_hi), context_hash), t)
    XOR 16
)

u = (token_hash + 0.5) / 2^32
```

三个容易写错的点：

1. `mix` 的每一步都要按 32 位回绕；
2. `context_hash` 只包含上下文 token，密钥只进入 `token_hash`；
3. 加 `0.5` 是为了把结果放入 `(0, 1)`，避免 `u=0` 或 `u=1` 进入对数。

双密钥模式会先用上下文哈希计算一个 `coin`，再按 `mixing_probability` 选择 key A 或 key B；单密钥模式不需要这一步。

---

## 6. 哪些位置会水印

只有同时满足以下条件，当前位置才走水印选择器：

| 条件 | 含义 |
|---|---|
| 请求启用水印 | 请求级 opt-in，或服务端配置默认启用 |
| 不是 greedy / `top_k=1` | 只有一个候选时没有可编码的选择空间 |
| 上下文非空 | 第一个生成的 token 至少要能看到一个前文 token |
| 上下文没有水印过 | 避免同一个上下文反复强制成同一个 token |
| 未达到上下文预算 | 生产上限是 `min(4096, per-request token-pool capacity)` |
| 最大截断概率不超过 `max_probability` | 低熵位置可以回退普通采样 |

只有真正进入水印选择的位置，才会把上下文记录进当前请求的历史。被概率门跳过、重复上下文跳过或预算耗尽的位置，继续使用普通采样。

这里有一个容易混淆的细节：被跳过的位置不一定不参与检测。检测器不知道某个位置是否被 `max_probability` 跳过，因此它可能把该位置当成普通样本计入统计；这会稀释信号，但不会制造虚假证据。

---

## 7. 检测流程

检测器需要：

- 密钥；
- prompt token IDs（可选；缺失时，开头 `h` 个生成 token 无法重建上下文，会被跳过）；
- 生成文本的 token IDs；
- 请求使用的上下文窗口 `h`；
- 用于 `max_contexts` 的生成请求预算。

对每个生成位置，检测器重建上下文，并计算实际 token 的 `u`：

```text
s = -log(1 - u)
```

为什么用这个分数：

- 普通采样下，实际 token 的 `u` 近似均匀，`s` 近似服从均值为 1 的指数分布；
- 水印选择偏向较大的 `u`，而 `-log(1-u)` 会把这些较大的 `u` 映射成较大的分数；
- 因此水印文本的累计分数通常偏高。

如果检测了 `n` 个不同上下文，定义：

\[
S = s_1 + s_2 + \dots + s_n
\]

在没有水印的零假设下：

\[
S \sim \mathrm{Gamma}(n, 1)
\]

检测器报告上尾概率：

\[
p = P(\mathrm{Gamma}(n,1) \ge S)
\]

同时也常用正态近似：

\[
z = \frac{S - n}{\sqrt{n}}
\]

### `n` 不是上下文窗口大小

`n` 是不同上下文的样本数，不是 `h`，也不是上下文窗口的个数。

```text
h：每个位置向前看多少 token
n：最后有多少个不同上下文参与统计
```

检测器对生成 token 逐个重建上下文：

```text
context_t = [t-h, ..., t-1]
```

规则是：

- 同一个上下文只计一次；
- 开头没有 prompt IDs 时，前 `h` 个生成 token 无法重建上下文，会被跳过；
- 达到 `max_contexts` 后停止；
- 语法或概率门跳过但检测器仍能重建的位置，可能作为普通样本被计入。

例如生成了 100 个 token：

```text
有 90 个位置可以重建上下文
其中 70 个上下文不重复
```

那么 `n=70`，而不是 100，也不是 `h`。如果只生成了很短的一段文本，不同上下文数量少，统计功效自然弱。

### 检测边界

- 没有 prompt token IDs 时，开头 `h` 个生成 token 被跳过。
- 检测器按精确 token tuple 比较上下文；生成端按 32 位哈希判重。哈希碰撞只会让检测器多算一个普通样本，通常只会削弱信号。
- 改写、翻译、重新分词可能破坏原始 token 序列。
- 水印不是密码学签名；持有密钥的人可以检测，也可以伪造。
- 低 p-value 只是“与该密钥水印相符”的证据，不是作者身份或模型归属的校准概率。

---

## 8. 完整可运行参考实现

下面的脚本只依赖 Python 标准库，完整实现了：

- MurmurHash3 风格的单密钥哈希契约；
- `argmax(log(u_i)/p_i)` 选择器；
- 上下文判重、重复上下文跳过、概率门和上下文预算；
- `s=-log(1-u)` 与 Gamma 上尾检测；
- 分布保持检查和普通/水印文本的对照实验。

它使用固定的 5 类 toy token 分布，不依赖模型、tokenizer 或 GPU。生产环境应使用 SGLang 的融合 kernel 和真实 token IDs，不要把这份 CPU 参考实现当成完整服务端代码。

```python
#!/usr/bin/env python3
"""Standalone demo of SGLang's Aaronson-Gumbel watermark contract.

This is a reference walkthrough, not the production SGLang kernel. It implements
the same single-key hash contract and selector on the CPU so that the sampling
distribution and detection statistic can be inspected without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, lgamma, log, sqrt
from random import Random
from typing import Sequence

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
KEY = 0x0123456789ABCDEF
PROMPT = (101, 102, 103, 104, 105, 106, 107, 108)
PROBABILITIES = (0.40, 0.25, 0.20, 0.10, 0.05)


def rotl32(value: int, shift: int) -> int:
    value &= MASK32
    return ((value << shift) | (value >> (32 - shift))) & MASK32


def mix(state: int, value: int) -> int:
    """MurmurHash3 x86 32-bit body mix."""
    value = (value * 0xCC9E2D51) & MASK32
    value = rotl32(value, 15)
    value = (value * 0x1B873593) & MASK32
    state = rotl32(state ^ value, 13)
    return (state * 5 + 0xE6546B64) & MASK32


def fmix32(value: int) -> int:
    """MurmurHash3 32-bit finalizer."""
    value &= MASK32
    value ^= value >> 16
    value = (value * 0x85EBCA6B) & MASK32
    value ^= value >> 13
    value = (value * 0xC2B2AE35) & MASK32
    value ^= value >> 16
    return value & MASK32


def context_hash(context: Sequence[int]) -> int:
    state = 0
    for token_id in context:
        state = mix(state, int(token_id))
    return fmix32(state ^ ((4 * len(context)) & MASK32))


def token_u(key: int, ctx_hash: int, token_id: int) -> float:
    key &= MASK64
    key_lo = key & MASK32
    key_hi = (key >> 32) & MASK32
    state = mix(0, key_lo)
    state = mix(state, key_hi)
    state = mix(state, ctx_hash)
    token_hash = fmix32(mix(state, int(token_id)) ^ 16)
    return (token_hash + 0.5) / (1 << 32)


def _validate_probabilities(probabilities: Sequence[float]) -> None:
    if not probabilities:
        raise ValueError("probabilities must not be empty")
    if any(p < 0.0 for p in probabilities):
        raise ValueError("probabilities must be non-negative")
    if abs(sum(probabilities) - 1.0) > 1e-9:
        raise ValueError("probabilities must sum to 1")


def watermark_select(
    probabilities: Sequence[float], context: Sequence[int], key: int = KEY
) -> int:
    """Select a token with argmax(log(u_i) / p_i)."""
    _validate_probabilities(probabilities)
    ctx_hash = context_hash(context)
    best_token = -1
    best_score = float("-inf")
    # Iterating in token-ID order makes the first candidate win an exact tie.
    for token_id, probability in enumerate(probabilities):
        if probability <= 0.0:
            continue
        score = log(token_u(key, ctx_hash, token_id)) / probability
        if best_token < 0 or score > best_score:
            best_token = token_id
            best_score = score
    if best_token < 0:
        raise ValueError("at least one probability must be positive")
    return best_token


def ordinary_sample(probabilities: Sequence[float], rng: Random) -> int:
    _validate_probabilities(probabilities)
    draw = rng.random()
    cumulative = 0.0
    for token_id, probability in enumerate(probabilities):
        cumulative += probability
        if draw < cumulative:
            return token_id
    return len(probabilities) - 1


@dataclass
class WatermarkState:
    """Per-request eligibility and distinct-context history for the demo."""

    key: int
    context_window: int
    max_contexts: int = 4096
    max_probability: float = 1.0
    enabled: bool = True
    seen_hashes: set[int] = field(default_factory=set)
    forced_contexts: int = 0

    def __post_init__(self) -> None:
        if not 1 <= self.context_window <= 64:
            raise ValueError("context_window must be in [1, 64]")
        if not 0.0 < self.max_probability <= 1.0:
            raise ValueError("max_probability must be in (0, 1]")

    def choose(
        self, probabilities: Sequence[float], context: Sequence[int], rng: Random
    ) -> tuple[int, bool]:
        _validate_probabilities(probabilities)
        context = tuple(context[-self.context_window :])
        positive = sum(p > 0.0 for p in probabilities)
        if (
            not self.enabled
            or not context
            or positive <= 1
            or max(probabilities) > self.max_probability
            or self.forced_contexts >= self.max_contexts
        ):
            return ordinary_sample(probabilities, rng), False

        ctx_hash = context_hash(context)
        if ctx_hash in self.seen_hashes:
            return ordinary_sample(probabilities, rng), False

        token_id = watermark_select(probabilities, context, self.key)
        self.seen_hashes.add(ctx_hash)
        self.forced_contexts += 1
        return token_id, True


def _regularized_gamma_q(shape: float, x: float) -> float:
    """Upper regularized incomplete gamma Q(shape, x), stdlib-only."""
    if shape <= 0.0 or x < 0.0:
        raise ValueError("shape must be positive and x non-negative")
    if x == 0.0:
        return 1.0
    log_prefactor = shape * log(x) - x - lgamma(shape)
    epsilon = 3e-14
    tiny = 1e-300

    if x < shape + 1.0:
        term = 1.0 / shape
        total = term
        ap = shape
        for _ in range(10_000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * epsilon:
                break
        lower = total * exp(log_prefactor)
        return max(0.0, min(1.0, 1.0 - lower))

    b = x + 1.0 - shape
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - shape)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break
    upper = exp(log_prefactor) * h
    return max(0.0, min(1.0, upper))


@dataclass(frozen=True)
class DetectionResult:
    num_contexts: int
    score: float
    z_score: float
    p_value: float
    tokens_examined: int


def detect(
    completion: Sequence[int],
    *,
    prompt: Sequence[int] = (),
    key: int = KEY,
    context_window: int = 4,
    max_contexts: int = 4096,
) -> DetectionResult:
    """Score a completion using exact context tuples, as the detector does."""
    if context_window <= 0:
        raise ValueError("context_window must be positive")
    if max_contexts <= 0:
        raise ValueError("max_contexts must be positive")

    prompt = tuple(prompt)
    text = prompt + tuple(completion)
    seen_contexts: set[tuple[int, ...]] = set()
    score = 0.0
    num_contexts = 0
    tokens_examined = 0
    # Without prompt IDs, the first h completion tokens have unknown contexts.
    start_position = len(prompt) if prompt else context_window

    for position in range(start_position, len(text)):
        tokens_examined += 1
        context = tuple(text[max(0, position - context_window) : position])
        if not context or context in seen_contexts:
            continue
        seen_contexts.add(context)
        u = token_u(key, context_hash(context), text[position])
        score += -log(1.0 - u)
        num_contexts += 1
        if num_contexts >= max_contexts:
            break

    if num_contexts == 0:
        raise ValueError("no context was eligible for detection")
    z_score = (score - num_contexts) / sqrt(num_contexts)
    return DetectionResult(
        num_contexts=num_contexts,
        score=score,
        z_score=z_score,
        p_value=_regularized_gamma_q(float(num_contexts), score),
        tokens_examined=tokens_examined,
    )


def generate(
    *,
    watermarked: bool,
    seed: int,
    num_tokens: int = 256,
    context_window: int = 4,
) -> tuple[tuple[int, ...], int]:
    rng = Random(seed)
    text = list(PROMPT)
    state = WatermarkState(
        key=KEY,
        context_window=context_window,
        max_contexts=num_tokens,
        max_probability=1.0,
    )
    forced = 0
    for _ in range(num_tokens):
        context = tuple(text[-context_window:])
        if watermarked:
            token_id, was_forced = state.choose(PROBABILITIES, context, rng)
            forced += int(was_forced)
        else:
            token_id = ordinary_sample(PROBABILITIES, rng)
        text.append(token_id)
    return tuple(text[len(PROMPT) :]), forced


def main() -> None:
    example_context = (11, 12, 13, 14)
    example_probabilities = (0.7, 0.2, 0.1)
    ctx_hash = context_hash(example_context)
    us = [token_u(KEY, ctx_hash, token_id) for token_id in range(3)]
    scores = [log(u) / p for u, p in zip(us, example_probabilities)]
    selected = watermark_select(example_probabilities, example_context)

    print("1) One watermarked selection")
    print(f"context_hash = 0x{ctx_hash:08x}")
    for token_id in range(3):
        print(
            f"token={token_id} p={example_probabilities[token_id]:.1f} "
            f"u={us[token_id]:.6f} score={scores[token_id]:.6f}"
        )
    print(f"selected token = {selected}\n")

    distribution_draws = 50_000
    counts = [0] * len(PROBABILITIES)
    for i in range(distribution_draws):
        context = (9000 + i, 31, 32, 33)
        counts[watermark_select(PROBABILITIES, context)] += 1
    print(f"2) Marginal distribution over {distribution_draws:,} contexts")
    for token_id, expected in enumerate(PROBABILITIES):
        observed = counts[token_id] / distribution_draws
        print(
            f"token={token_id} expected={expected:.3f} "
            f"observed={observed:.3f} delta={observed - expected:+.3f}"
        )
    print()

    trials = 100
    print(f"3) Detector over {trials} independent sequences")
    for label, watermarked in (("ordinary", False), ("watermarked", True)):
        p_values = []
        forced_counts = []
        for seed in range(trials):
            completion, forced = generate(watermarked=watermarked, seed=seed)
            result = detect(completion, prompt=PROMPT, key=KEY, context_window=4)
            p_values.append(result.p_value)
            forced_counts.append(forced)
        flagged = sum(p < 0.01 for p in p_values)
        print(
            f"{label:11s}: flagged_at_p<0.01={flagged}/{trials} "
            f"mean_forced={sum(forced_counts) / trials:.1f} "
            f"min_p={min(p_values):.3e}"
        )


if __name__ == "__main__":
    main()
```

运行：

```bash
python3 code/watermarking/text_watermark_demo.py
```

---

## 9. 实测输出

```text
1) One watermarked selection
context_hash = 0x76c6c792
token=0 p=0.7 u=0.021917 score=-5.457851
token=1 p=0.2 u=0.454224 score=-3.945826
token=2 p=0.1 u=0.464096 score=-7.676643
selected token = 1

2) Marginal distribution over 50,000 contexts
token=0 expected=0.400 observed=0.400 delta=-0.000
token=1 expected=0.250 observed=0.248 delta=-0.002
token=2 expected=0.200 observed=0.201 delta=+0.001
token=3 expected=0.100 observed=0.100 delta=+0.000
token=4 expected=0.050 observed=0.051 delta=+0.001

3) Detector over 100 independent sequences
ordinary   : flagged_at_p<0.01=0/100 mean_forced=0.0 min_p=1.742e-02
watermarked: flagged_at_p<0.01=100/100 mean_forced=164.6 min_p=2.243e-35
```

这组实验说明两件事：

1. 选择器在大量不同上下文上仍然保持了接近 `p_i` 的边际频率；
2. 检测器能在普通文本上控制误报，同时在水印文本上得到明显更低的 p-value。

它不说明真实模型、真实 tokenizer、投机解码或 GPU 上的性能表现。

---

## 10. 与缓存、投机解码的关系

### 与 RadixAttention / paged attention

RadixAttention 的复用对象是实际 token 序列对应的 KV。水印采样改变实际 token 序列，所以可能间接改变缓存命中率和共享前缀；但 attention kernel 不需要理解水印。

```text
watermark → 选 token → 正常 KV 写入 → Radix tree / paged cache
```

不是：

```text
watermark → 修改 attention 的 Q/K/V 计算
```

### 与 speculative decoding

投机解码先由 draft model 提出若干 token，再由 target model 验证。开启水印后，target 的目标 token 是强制选择的结果；draft 提议只有在与目标 token 一致时才能被接受。

因此可能出现：

- 接受长度下降；
- TPOT 上升；
- 输出吞吐下降；
- 树形投机路径的重建和记录逻辑更复杂。

文档报告的具体百分比依赖模型、硬件、batch 和接受率，不能直接外推到所有部署。做优化时应把 watermark selector 与 draft/accept 逻辑融合，而不是在 CPU 上逐 token 做额外 Python 调用。

---

## 11. 常见误区

### 误区一：水印是文本中的固定标记

不是。它没有固定字符串、固定 token 或显式 metadata。单个 token 也不携带可独立验证的标签。

### 误区二：水印一定改变生成质量

在理想独立均匀哈希模型下，每个位置的边际采样分布与普通采样相同。实际系统仍需测量哈希质量、数值精度、候选集大小和投机解码带来的影响。

### 误区三：p-value 是作者概率

不是。它只是“无该密钥水印”这一零假设下的统计显著性。它不能证明具体作者、模型或数据来源。

### 误区四：拿到文本就能检测

检测需要匹配的密钥、上下文窗口、tokenizer 和可重建的 token IDs。重新分词或改写文本可能让信号显著变弱。

### 误区五：`n` 是 token 总数

不是。`n` 是不同上下文样本数；重复上下文和前面无法重建的 token 都可能被跳过。

---

## 12. 自检

**Q1. 为什么 `argmax(log(u_i)/p_i)` 仍然会以 `p_i` 的概率选中 token i？**

<details><summary>答案</summary>

令 `E_i = -log(u_i)/p_i`。`u_i` 均匀时，`E_i` 是速率为 `p_i` 的指数分布。独立指数分布中，第 i 个事件最先发生的概率是 `p_i / Σp_j`；概率归一化后就是 `p_i`。

</details>

**Q2. 检测时为什么用 `s=-log(1-u)`，而不是 `-log(u)`？**

<details><summary>答案</summary>

水印选择偏向较大的 `u_i`。`-log(1-u)` 会随 `u` 增大而增大，因此水印文本的分数会整体偏高；普通文本中 `u` 均匀，`s` 近似指数分布，累计后可用 Gamma 检验。

</details>

**Q3. 为什么检测统计量里的 `n` 不是上下文窗口 `h`？**

<details><summary>答案</summary>

`h` 决定每个位置的上下文长度；`n` 是实际参与统计的不同上下文数量。重复上下文只算一次，缺少 prompt 时的开头 token 和达到预算后的 token 也可能不计入，所以 `n` 和总 token 数、`h` 都不同。

</details>

**Q4. 水印和 RadixAttention 会不会互相修改内部状态？**

<details><summary>答案</summary>

不会直接互相修改。水印只参与采样，选出实际 token；RadixAttention 和 paged attention 继续处理这些 token 的正常 KV。水印可能通过改变 token 序列间接影响缓存命中，但不需要 attention kernel 识别水印。

</details>

**Q5. 为什么低 p-value 不等于“证明某模型生成了文本”？**

<details><summary>答案</summary>

它只表示文本与某个密钥对应的统计水印相符。作者归属还涉及密钥管理、文本复制、改写、密钥共享和误报率等问题；而且这里的 MurmurHash3 是非密码学哈希，持有密钥的人可以伪造水印。

</details>

---

## 关联

- 官方文档：[SGLang Text Watermarking](https://docs.sglang.io/docs/advanced_features/text_watermarking)
- 参考实现：[code/watermarking/text_watermark_demo.py](code/watermarking/text_watermark_demo.py)
- 相关缓存背景：[16-official-flash-attention-varlen-paged.md](16-official-flash-attention-varlen-paged.md)
- 相关采样与投机路径：SGLang runtime 的 `python/sglang/srt/sampling/watermarking/`，以及 `test/registered/kernels/ops/sampling/test_watermark_*.py`
