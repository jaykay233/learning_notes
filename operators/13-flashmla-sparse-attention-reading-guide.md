# FlashMLA：从 Python API 到稀疏 Attention Decode

> 本笔记对应 `/Users/saboxu/Downloads/FlashMLA` 的本地快照 `2e5429f`（2026-09-30）。FlashMLA 的接口和 KV 格式有过 breaking change；下面讲的是这个快照，不要直接套用到旧版 Hopper / DeepSeek V3.x 分支。

## 本次讲解位置

```text
章节：算子开发 / FlashMLA 阅读路线
小节：DeepSeek Sparse Attention 的 Decode API → 调度 → kernel
知识点：稀疏索引如何驱动量化 paged KV 读取，以及长 Top-K 为什么要 split-KV
上次：[12-triton-gluon-multicta-and-capstone.md] 的 Blackwell tile / 数据流与 kernel 开发流程；此前在对话里刚讨论 Triton 循环调度参数
下次：沿 get_decoding_sched_meta.cu 的 DecodingSchedMeta 与 combine.cu 深挖 split-KV 的工作分配、partial output/LSE 合并与同步
```

## 为什么现在讲这个

前面在 Triton 里看到的是“一个 program 如何循环处理 tile、编译器如何安排流水”。FlashMLA 展示的是生产级 Attention 算子还要解决的上一层问题：**哪些 KV token 要算、KV 如何压缩并分页、一个请求的长 Top-K 如何铺到 GPU 上，以及分块后的 softmax 怎样合并才仍然精确**。这能避免两个误解：把 FlashMLA 当作 Triton 示例 kernel；或以为它会自己产生 DSA 的稀疏索引。当前仓库的关键 decode kernel 是 C++/CUDA（CuTe/CUTLASS）实现，Ascend 路径则是 Ascend C，并不是 Triton kernel。

## 先建立整体心智模型

把 FlashMLA 看成一个**注意力算子库**，不是一个模型，也不是 DSA 的完整稀疏检索系统：

```text
模型 / 上游 Indexer
  ├─ 产生 Q
  ├─ 写入量化、分页的 KV cache
  └─ 产生每个 query 要看的 indices（FlashMLA 消费它，不负责挑选它）
           ↓
Python API（flash_mla_interface.py）
           ↓
C++ 扩展（校验 shape/dtype、选实现、准备临时区和调度 metadata）
           ↓
CUDA SM100/SM103 kernel 或 Ascend 950 kernel
           ↓
输出 attention result + log-sum-exp（LSE）
```

当前快照对外主要包含四类工作：

| 工作 | 做什么 | 当前实现边界 |
|---|---|---|
| DSA sparse decode | 单步/少量 query token，只访问 `indices` 指定的 KV token；decode KV 为 FP8/FP4 cache | CUDA SM100/SM103 与 Ascend 950；当前 CUDA 路径针对 `head_dim=512`、`h_q=64/128`、`h_kv=1` 等固定模型形状 |
| DSA sparse prefill | 对一段 query 序列执行 token-level sparse attention | CUDA 与 Ascend 的稀疏路径；输入 KV 是 BF16 张量，接口与 paged decode 不同 |
| Dense MHA prefill / backward | 普通稠密 MHA 的变长前向与反向 | 当前 README 标为 CUDA SM100 路径 |
| Fused norm/RoPE/attention/RoPE/cast | 把注意力前后的小算子与 core attention 融合，减少 launch / 中间读写 | CUDA；依赖额外的权重预排布，Ascend 不支持该 fused API |

所以 **FlashMLA ≠ 只有一个 kernel**。用户看到统一的 Python package，后面可能走完全不同的 C++ dispatch 和 kernel family。

## 这节聚焦：sparse decode 的语义和数据形状

对 batch `b`、query 长度 `s_q`、query heads `h_q`、维度 `d=512`，接口大致是：

```text
q:       [b, s_q, h_q, 512]                 BF16
k_cache:[num_blocks, page_size, 1, bytes]   FP8/打包量化 cache
indices: [b, s_q, topk]                    int32
out:     [b, s_q, h_q, 512]                 BF16
lse:     [b, h_q, s_q]                     FP32
```

对一个 query/head，数学上做的是：

```text
x_j = sm_scale * dot(q, K[indices[j]])
p_j = exp(x_j) / Σ_k exp(x_k)
out = Σ_j p_j * V[indices[j]]
LSE = log(Σ_j exp(x_j))
```

`indices` 是**稀疏访问清单**。上游（例如 DSA 的 indexer）决定挑哪些 token；FlashMLA 的职责是只 gather 这些 token、算 attention、写结果。若未提供 `indices`，这个 decode API 会直接拒绝，因为它当前就是 sparse decode 接口。

### Paged index 是什么

索引编码为：

```text
flat_token_index = page_id * page_size + offset_in_page
```

例如 `page_size=16`，某 token 在 page 3 的 offset 5：

```text
flat_token_index = 3 * 16 + 5 = 53
page_id           = 53 // 16 = 3
offset_in_page    = 53 % 16  = 5
```

decode API 的 `block_table` 和 `cache_seqlens` 参数在这个快照里只是为兼容旧接口而保留，**不参与这条 sparse decode 路径**；`indices` 已编码 KV cache 的 page 与 page 内位置。不要因为参数列表里有 `block_table`，就误以为 kernel 仍通过它查页表。

CUDA decode 对 invalid index 的约定也要特别小心：**仅 `-1` 被视为无效**，kernel 不为其他正数做上界检查。超范围正数可能变成非法 TMA 地址。和它不同，sparse prefill 的文档允许 `-1` 或 `>= s_kv` 表示无效。因此不要把 prefill 的 sentinel 规则照搬到 decode；decode 中无效项必须写 `-1`。

## 从调用到 kernel：代码路径

### 1. Python 层：接口与 metadata 生命周期

入口在 `flash_mla/flash_mla_interface.py`：

- `get_mla_metadata()` 当前返回一个空的 `FlashMLASchedMeta` 占位对象，不在这里真正计算调度表。
- 当 C++ dispatch 判定本轮需要 split-KV 时，第一次 `flash_mla_with_kvcache(...)` 才会创建 tile scheduler metadata 与 `num_splits`，并返回给 wrapper 存回对象；短 Top-K 不走 split-KV 时，不需要这张 scheduler 表。
- 相同对象可以复用，但 shape、top-k 配置以及 `topk_length` / `extra_topk_length` 的取值必须与生成 metadata 时匹配。源代码特别提醒：部分值变化时可能复用过期 split 调度，且不会逐值做完整运行时校验。
- `num_splits` 这个位置参数在当前 public wrapper 必须传 `None`；保留它是历史接口兼容，真正 split 信息保存在 metadata 对象中。

另外，Python 层按 `/dev/davinci_manager` 是否存在来选择 NPU 或 CUDA backend。这是**平台 backend 选择**；后续 C++ 还会按架构、head 数、cache 格式和可选 feature 继续 dispatch。

### 2. C++ API：先校验，再选择实现

入口为 `csrc/api/sparse_decode.cpp::sparse_decode_fwd`。它会校验设备、shape、dtype、stride 与特性组合；例如当前路径要求 `h_kv=1`、`d_qk=d_v=512`，head 数支持 64/128。CUDA 还会检查 SM100 系列架构。

CUDA 侧大致有这些分支：

- `h_q=64`：使用独立的 SM100 sparse decode head64 kernel。
- `h_q=128`：复用 `fwd_for_small_topk` 的 head128 kernel family，并用 Decode / DecodeWithSplitKV 模式运行。
- 按主 cache / extra cache 的 FP8、FP4 格式组合实例化模板；当前主 cache 要是 FP8，FP4 可以出现在额外 cache 路径。
- `attn_sink`、`topk_length`、`extra_k_cache`、`batch_invariant` 会影响 feature dispatch，而非一个通用 kernel 内的无条件分支。

在当前 CUDA API 中，是否启用 split-KV 的主门槛是：

```text
enable_split_kv = (topk + extra_topk > 640) AND (NOT enable_batch_invariant)
```

这里比较的是 `indices` 张量末维 `topk` 的容量加上 `extra_topk` 容量，不是对每个运行时 `topk_length` 重新套这个常数阈值。这只是当前实现的 dispatch 规则，不是所有 attention kernel 的普遍规律。短 Top-K 的并行工作足够少，split 后多一次 partial 输出与 combine 不划算；长 Top-K 则可以拆成更多 CTA/SM 工作。Ascend 当前不启用 split-KV。

### 3. KV 格式：cache 每 token 的字节数是 layout 契约

当前 V4.1 的 cache 把每个 token 的量化数据和 scale 连续放在一起：

| 格式 | raw 部分 | scale 部分 | 每 token 总字节 | scale 粒度 |
|---|---:|---:|---:|---:|
| FP8 E4M3 | 512 B（512 个数） | 16 B | 528 B | 每 32 个 FP8 共用 1 scale |
| FP4 E2M1 | 256 B（512 个 4-bit 数） | 32 B | 288 B | 每 16 个 FP4 共用 1 scale |

FP4 中每字节打包两个 4-bit 值；cache 最后一维不是浮点元素数量，而是**token 的字节数**。C++ 依据该维度识别格式。主 `k_cache` 不接受 FP4；FP4 只允许出现在 `extra_k_cache`，且主 cache 要是 V4.1 FP8。

## Split-KV：分块不等于分别做完 softmax

长 Top-K 直接交给少数 CTA 时，单个 CTA 需要遍历很多 KV token；并行度和 latency hiding 可能不足。split-KV 把同一个 query 的 KV 列表分段，让多个分区并行算 partial attention，最后再 combine。

关键难点：不能对每个 split 的输出做普通平均，因为每段 softmax 的分母不同。可稳定合并每段的 `(m, l, u)`：

```text
m = max(score)
l = Σ exp(score - m)
u = Σ exp(score - m) * value
output = u / l
LSE = m + log(l)
```

若某 split 有 `(m_s, l_s, u_s)`，全局 `m = max_s(m_s)`，则：

```text
l = Σ_s l_s * exp(m_s - m)
u = Σ_s u_s * exp(m_s - m)
```

这是在线 softmax 的稳定合并：每段可独立处理，合并时把局部统计量重标到共同最大值。实际 FlashMLA kernel 使用 base-2 形式和硬件友好的缓存/寄存器布局；数学语义等价。当前 head64 CUDA kernel 还做了多级异步流水、量化 KV gather/dequant、QK 与 PV Tensor Core 运算和 barrier 协作；这比“一个 Triton `for` 循环”多了明确的硬件数据流和生命周期管理。

### 手算轨迹：4 个 KV 分成两段

为只看清 split 合并，构造一维例子：`q=1`、`scale=1`，四个 key 使 logits 分别为 `0, ln2, ln3, ln4`，对应 value 是 `10,20,30,40`。因此 softmax 未归一化权重正好是 `1,2,3,4`。

| split | logits | `m_s` | `l_s=Σexp(x-m_s)` | `u_s=Σexp(x-m_s)v` | 局部 output |
|---|---|---:|---:|---:|---:|
| A | `0, ln2` | `ln2` | `1/2+1=1.5` | `10/2+20=25` | `25/1.5=16.6667` |
| B | `ln3, ln4` | `ln4` | `3/4+1=1.75` | `30*3/4+40=62.5` | `62.5/1.75=35.7143` |

统一到 `m=ln4`：

```text
l = 1.5 * exp(ln2-ln4) + 1.75 = 0.75 + 1.75 = 2.5
u = 25  * exp(ln2-ln4) + 62.5 = 12.5 + 62.5 = 75
output = 75 / 2.5 = 30
LSE = ln4 + ln2.5 = ln10 ≈ 2.302585
```

这与不切分时 `(10*1+20*2+30*3+40*4)/(1+2+3+4)=300/10=30` 完全一致。局部 output 的简单平均是 `(16.6667+35.7143)/2≈26.19`，明显错误；split merge 必须保留 softmax 归一化信息。

## 完整可运行的 CPU 数学示例

下面不是替代 FlashMLA kernel 的实现，而是可在普通 Python 上复现 sparse gather 与 split-KV 统计量合并。它不依赖 PyTorch、CUDA 或第三方包。

文件：`operators/code/flashmla/sparse_attention_split_merge.py`

```python
from math import exp, log
from typing import List, Sequence, Tuple

Partial = Tuple[float, float, float]  # (m, l, u)


def make_partial(q: float, keys: Sequence[float], values: Sequence[float],
                 indices: Sequence[int]) -> Partial:
    """对一个 index 子集计算稳定的 softmax 统计量 (m, l, u)。"""
    scores = [q * keys[index] for index in indices]
    m = max(scores)
    weights = [exp(score - m) for score in scores]
    l = sum(weights)
    u = sum(weight * values[index] for weight, index in zip(weights, indices))
    return m, l, u


def merge_partials(partials: Sequence[Partial]) -> Tuple[float, float]:
    """返回 (attention output, log-sum-exp)，等价于一次性计算全部 token。"""
    m = max(partial[0] for partial in partials)
    l = sum(partial_l * exp(partial_m - m)
            for partial_m, partial_l, _ in partials)
    u = sum(partial_u * exp(partial_m - m)
            for partial_m, _, partial_u in partials)
    return u / l, m + log(l)


def main() -> None:
    q = 1.0
    keys = [0.0, log(2.0), log(3.0), log(4.0)]
    values = [10.0, 20.0, 30.0, 40.0]
    sparse_indices = [0, 1, 2, 3]  # 真实场景由上游 indexer 提供
    chunks = [sparse_indices[:2], sparse_indices[2:]]

    partials: List[Partial] = []
    for split_id, indices in enumerate(chunks):
        partial = make_partial(q, keys, values, indices)
        partials.append(partial)
        m, l, u = partial
        print(
            f"split {split_id}: m={m:.6f}, l={l:.6f}, "
            f"u={u:.6f}, local_output={u / l:.6f}"
        )

    output, lse = merge_partials(partials)
    print(f"merged output={output:.6f}, lse={lse:.6f}")

    # 与未切分的直接计算互相校验。
    direct = make_partial(q, keys, values, sparse_indices)
    assert abs(output - direct[2] / direct[1]) < 1e-12
    assert abs(lse - (direct[0] + log(direct[1]))) < 1e-12


if __name__ == "__main__":
    main()
```

运行命令：

```bash
python3 operators/code/flashmla/sparse_attention_split_merge.py
```

预期输出（末位可能因平台浮点误差略有差异）：

```text
split 0: m=0.693147, l=1.500000, u=25.000000, local_output=16.666667
split 1: m=1.386294, l=1.750000, u=62.500000, local_output=35.714286
merged output=30.000000, lse=2.302585
```

## 常见误读与可观察后果

1. **以为 FlashMLA kernel 会挑 DSA Top-K。** 实际 sparse API 接收 `indices`；错误或没准备好 index 时，算的是错误的注意力 token 集合，而不是由 kernel 自动修复。
2. **把它当作 Triton 项目。** Python 是包装与 dispatch 层，核心 CUDA sparse decode 是 C++/CuTe/CUTLASS kernel；仓库测试工具里虽有 Triton benchmark helper，但去找 `@triton.jit` 会找不到主实现。
3. **把 decode 的 invalid sentinel 写成大于等于 cache 长度。** decode 只检查 `-1`；错误正数可能触发越界访存/非法地址。不要混淆 prefill 与 decode 的约定。
4. **复用过期 scheduler metadata。** 改了 topk 长度分布却沿用原对象，可能让请求被错误 split/调度；应保持 metadata contract 或重建对象。
5. **把每个 split 的 output 直接平均。** 不同分区 softmax mass 不等，输出会偏离 reference；症状通常是结果数值错而非 kernel launch 报错。
6. **拿当前主分支去跑旧 Hopper / V3.x。** README 明确当前版本移除了 Hopper/旧模型支持并改了 cache format；需要旧行为时必须 checkout README 指定的历史版本，不能仅改一个 dtype 参数。

## 如何验证

CPU 数学示例可在没有 GPU 的机器上运行：

```bash
python3 operators/code/flashmla/sparse_attention_split_merge.py
```

要验证真实 extension，需要 README 列出的目标硬件和工具链：NVIDIA SM100/SM103 与 CUDA 13.1+（推荐 13.2+），或 Ascend 950 与 CANN 9.2+、`torch_npu`。在对应设备上，仓库的 sparse decode 回归入口是 `python tests/test-sparse-decode.py`；该测试还覆盖 correctness、invalid indices、Top-K 长度和性能。没有目标设备时，CPU 例子只证明 split-softmax 数学，不证明 CUDA/Ascend kernel 的编译与运行。

本机静态阅读环境是 macOS arm64，未发现 `nvidia-smi`；因此此处不声称真实 FlashMLA 扩展已通过本机 GPU 运行。

## 自我检查（含答案）

1. FlashMLA sparse decode 会不会自己挑选 DSA indices？
   - **答：** 不会；它消费调用方给的 token indices 并执行稀疏 attention。
2. `block_table` 和 `cache_seqlens` 在这个快照的 sparse decode 路径里是否用于查 KV 页？
   - **答：** 不用；保留以兼容接口，paged token 已通过 indices 编码。
3. 当前 decode 为什么只允许 `-1` 作为 invalid index？
   - **答：** kernel 对其他正数不做上界检查，可能形成非法 TMA 地址；prefill 的规则不同。
4. split 后为什么不能平均各块的 output？
   - **答：** 每块的 softmax mass 不同；必须按 LSE / 局部归一化量加权，并在共同最大值下稳定合并。
5. 当前源码里的 `get_mla_metadata()` 是否立即生成 GPU scheduler 表？
   - **答：** 否；它返回占位对象。需要 split-KV 时，实际 scheduler metadata 在首次 decode 调用期间建立并存回对象。

## 进度

已经覆盖：

- [x] FlashMLA 仓库定位、sparse decode 的 API/shape/index/cache 契约，以及 split-KV 在线 softmax 的数学合并。

下一知识点：

- [ ] 逐行跟踪 `DecodingSchedMeta`：一个 partition 如何映射请求与 Top-K block，combine kernel 如何消费 partial output/LSE。
