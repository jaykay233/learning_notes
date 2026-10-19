# DeepSelect：用动态阈值把精确 TopK 缩成候选集选择

## 本次讲解位置

```text
章节：算子开发（operators）
小节：阈值筛选式精确 TopK / DeepSelect
知识点：初始精确 TopK 建立阈值，扫描时过滤，候选区周期性精确重选
上次：14 · FlashKDA 的 KDA 递归与 K1/K2 数据流
下次：BF16 RadixSelect 的双字节直方图、pivot 与相等值 quota
PTX：不是 PTX ISA 教程；源码锚点为 common_parts.cuh 的 scan_segs / inline PTX
```

### 为什么现在讲这个

前面读过 FlashMLA / FlashKDA 的完整算子数据流，现在看 DeepSelect 的关键设计：**如何在仍然精确的前提下，不对整条长向量反复做 TopK。**这里容易把“随机访问”误读成“随机抽 k 个元素”。DeepSelect 实际先在一个较大的初始窗口内精确选 TopK，以窗口内第 k 大值建立阈值；若只抽 k 个值，阈值通常太松，几乎所有元素都会通过过滤。

### 1. 先建立 mental model

想象已经扫描了一部分数据，并维护当前最大的 k 个元素。记其中最小值为 `T`。那么任何新元素 `x <= T` 都不可能进入最终 TopK：已扫描部分里已经有 k 个值不小于 `T`，自然也不小于 `x`。所以后续只需把 `x > T` 的元素暂存为候选。

当候选变多时，把“当前 TopK + 新候选”做一次精确 TopK，得到更高的新阈值。筛选和精确选择交替进行。对全局正确性的关键不变量是：

> 在每次重选结束后，`survivor` 是已处理元素的精确 TopK；`T` 是这组 survivor 的第 k 大值。任何被过滤掉的元素都不可能进入全局 TopK。

此处的“随机”主要指**输入数据块的伪随机访问顺序**，不是只随机采 k 个数。

### 2. 术语与代码参数

| 符号 / 状态 | 含义 | 本仓库中的作用 |
|---|---|---|
| `N` | 一行输入元素数（例如词表或 context 长度） | 整行数据只扫描一遍 |
| `k` | 每行要选出的元素数 | 当前实现支持到 4096 |
| `T` | 当前第 k 大值阈值 | 只接受 `x > T` 的新元素 |
| `survivor` | 当前精确保留的 TopK 值-索引对 | 常驻 shared memory 的 A/B 缓冲区之一 |
| `incoming` | 上次重选后新通过阈值的候选 | 达到触发容量时参与下一次重选 |
| segment | 扫描/重排的连续数据段 | BF16 主路径每段 512 个元素 |
| init window | 用于建立初始阈值的窗口 | 共 32 KiB；BF16 为 16,384 个值，FP32 为 8,192 个值 |

### 3. DeepSelect 的真实数据路径

针对仓库当前 CUDA 源码（snapshot `0f03b68`）的 BF16 主路径：

1. **准备初始窗口。**普通 kernel 将尾部最多 4096 个元素顺序访问，并将输入其余部分切成 512 元素的 segment，以伪随机段顺序取一部分，使总初始工作集达到约 32 KiB。短行会直接在初始窗口里完成。
2. **初始窗口内精确选 TopK。**不是从窗口随机抽 k 个值。内核基于直方图/pivot 精确选出窗口内最大的 k 个，设置 `T = min(survivor)`。
3. **预取和扫描后续 segment。**每行通常由一个 CTA 处理。TMA 把数据块搬到 shared memory；访问保持块状连续，块的先后顺序做伪随机化，以免输入本身的排列让阈值长期不够紧。
4. **阈值筛选和候选压紧。**每个线程比较自己负责的值与 `T`。主 BF16 路径用 packed compare 产生 hit mask，再用 `popc`、`ffs` 和 warp prefix 计算紧凑写入位置，避免把整块都写到候选区。
5. **周期性精确重选。**`survivor + incoming` 用两轮 8-bit radix histogram 找 pivot：先缩小高字节桶，再在该桶中选低字节；随后 census `> pivot` 与 `== pivot` 的数量，为等于 pivot 的元素分配 quota，恰好保留 k 项。更新 `T`，切换 A/B survivor buffer。
6. **结束与输出。**扫描完成后做最终重选，再按接口配置决定是否返回值、是否按 index 排序等。BF16 主路径并非默认按值排序输出；选择集合和输出排列是两回事。

源码结构可从这些位置核对：

- `csrc/cuda_kernels/common_parts.cuh`：segment 参数、`scan_segs`、hit mask、候选重建、radix histogram/pivot。
- `csrc/cuda_kernels/v3/topk_select.cuh`：BF16 普通路径的 init 与调用。
- `csrc/cuda_kernels/v3_fp32/topk_select.cuh`：FP32 专用路径。
- `csrc/cuda_kernels/v3_cluster/topk_select.cuh` 与 `csrc/api.cpp`：小 batch、大 vocab 下将一行拆给 CTA cluster 的专用分支。

这些 kernel 用到 CUDA / TMA / shared memory / inline PTX；本机可读代码，但 M5 Pro 的 Apple GPU 不能直接运行这些 CUDA kernels。

### 4. 与“随机抽 k 个值”以及反向最小 TopK 的比较

| 方案 | 阈值 | 保留候选 | 典型候选比例 | 正确性条件 |
|---|---|---|---:|---|
| 抽 k 个输入值，再取样本最小值，找最大 TopK | 样本最小值 | `x > T` | 连续同分布下约 `1/(k+1)` | 样本是 k 个不同输入位置，且样本也被纳入已处理集合 |
| 抽 k 个输入值，再取样本最大值，找最小 TopK | 样本最大值 | `x < U` | 连续同分布下约 `k/(k+1)`，几乎全过 | 同上；需正确保留等值项 |
| 在 M 个初始值里精确选最大 k 个（DeepSelect） | 初始窗口的第 k 大值 | `x > T` | IID 近似约 `k/(M+1)` | 初始 TopK 精确，未扫描元素不被漏掉 |
| 在 M 个初始值里精确选最小 k 个（反向版本） | 初始窗口的第 k 小值 | `x < U` | IID 近似约 `k/(M+1)` | 同样需要精确 seed、分批重选、等值 quota |

以 `k=512` 为例，随机取 512 个数后用样本最大值筛选最小 TopK，下一独立数不超过该最大值的概率为 `512/513 ≈ 99.805%`。相反，如果从 `M=16,384` 个初始值中精确选最大 512 个，IID 近似下超过阈值的比例约为 `512/16,385 ≈ 3.125%`。这正是“初始池要比 k 大，而且池内必须精确选 TopK”的原因。

最小 TopK 版本在数学上完全对称：维护当前 k 个最小值，令 `U=max(survivor)`，只保留 `x < U` 并周期性精确选最小 k 个。对 value-only TopK，等于当前 cutoff 的元素不比已保留项更优，可以按任意 tie 规则跳过；但 radix pivot 落在重复值上时，必须精确统计并决定保留多少个等值项，才能恰好得到 k 个。若接口要求按 index 的确定性 tie-break，就必须把 index 纳入比较键，不能随意跳过阈值相等项。当前 DeepSelect 的已实现主路径是最大 TopK，反向版本需要改写 compare、radix 顺序与阈值更新，不是现成参数开关。

### 5. 具体执行轨迹：k=2 的最大 TopK

为手算方便，令初始池为 `[1,2]`，每个后续块有两个值，候选容量阈值设为 2。生产实现的窗口更大、重选使用 radix-select。

| 步骤 | 输入 / 通过值 | 重选前 survivor | 更新后 survivor | 新阈值 `T` |
|---|---|---|---|---:|
| 初始池精确 TopK | `[1,2]` | — | `[2,1]` | 1 |
| 扫描块 0 | `[3,9]` → `[3,9]` | `[2,1]` | `[9,3]` | 3 |
| 扫描块 1 | `[4,10]` → `[4,10]` | `[9,3]` | `[10,9]` | 9 |
| 扫描块 2 | `[8,0]` → 无命中 | `[10,9]` | `[10,9]` | 9 |
| 结束 | 最终重选 | `[10,9]` | `[10,9]` | 9 |

阈值单调增大：`1 → 3 → 9`。块 `[8,0]` 中的 8 被过滤，因为已处理数据里已有 10 和 9 两个更大的值。最终答案是全局 Top2 `[10,9]`。

### 6. 可运行的 CPU 教学模型

下面的完整脚本同时跑最大 TopK 与反向最小 TopK，对照全排序 reference。为聚焦阈值逻辑，它要求示例 value 互不相同；真实 CUDA kernel 还会正确处理重复值，并且用 radix-select，而不是 Python sort。

文件：[`code/deepselect_threshold_demo.py`](code/deepselect_threshold_demo.py)

```python
#!/usr/bin/env python3
"""Tiny CPU model of DeepSelect's threshold-and-reconstruct TopK strategy.

This is a teaching model, not a performance implementation. It uses Python sort
for each exact reconstruction; the CUDA kernel uses radix selection in shared
memory and PTX-assisted candidate compaction.
"""

from __future__ import annotations

from itertools import chain
from typing import Iterable, Sequence

Item = tuple[int, int]  # (value, original_index)


def threshold_topk(
    seed: Sequence[Item],
    blocks: Sequence[Sequence[Item]],
    k: int,
    *,
    largest: bool,
    candidate_limit: int,
    trace: bool = False,
) -> list[Item]:
    """Return exact TopK after an exact seed selection and threshold filtering.

    The values in this demo are unique, so the pivot/tie-quota corner case is
    deliberately excluded. Production kernels must handle equal values.
    """
    all_items = list(seed) + list(chain.from_iterable(blocks))
    values = [value for value, _ in all_items]
    assert len(values) == len(set(values)), "demo expects unique values"
    assert 0 < k <= len(seed)
    assert candidate_limit > 0

    def exact_topk(items: Iterable[Item]) -> list[Item]:
        return sorted(items, key=lambda item: item[0], reverse=largest)[:k]

    def passes(value: int, threshold: int) -> bool:
        return value > threshold if largest else value < threshold

    survivor = exact_topk(seed)  # exact within seed; not k random picks
    threshold = survivor[-1][0]
    incoming: list[Item] = []
    if trace:
        direction = "largest" if largest else "smallest"
        print(f"{direction}: seed survivor={survivor}, threshold={threshold}")

    for block_id, block in enumerate(blocks):
        hits = [item for item in block if passes(item[0], threshold)]
        incoming.extend(hits)
        if trace:
            print(
                f"  block {block_id}: values={[v for v, _ in block]}, "
                f"hits={hits}, incoming={incoming}"
            )

        if len(incoming) >= candidate_limit:
            survivor = exact_topk(chain(survivor, incoming))
            threshold = survivor[-1][0]
            incoming.clear()
            if trace:
                print(
                    f"    reconstruct: survivor={survivor}, "
                    f"new threshold={threshold}"
                )

    if incoming:
        survivor = exact_topk(chain(survivor, incoming))
        threshold = survivor[-1][0]
        if trace:
            print(f"  final reconstruct: survivor={survivor}, threshold={threshold}")

    return survivor


def main() -> None:
    largest_seed = [(1, 0), (2, 1)]
    largest_blocks = [
        [(3, 2), (9, 3)],
        [(4, 4), (10, 5)],
        [(8, 6), (0, 7)],
    ]
    got_largest = threshold_topk(
        largest_seed, largest_blocks, 2,
        largest=True, candidate_limit=2, trace=True,
    )
    ref_largest = sorted(
        chain(largest_seed, chain.from_iterable(largest_blocks)),
        key=lambda item: item[0], reverse=True,
    )[:2]
    assert got_largest == ref_largest == [(10, 5), (9, 3)]
    print(f"  result={got_largest}; reference={ref_largest}\n")

    smallest_seed = [(8, 0), (9, 1)]
    smallest_blocks = [
        [(7, 2), (1, 3)],
        [(6, 4), (0, 5)],
        [(2, 6), (10, 7)],
    ]
    got_smallest = threshold_topk(
        smallest_seed, smallest_blocks, 2,
        largest=False, candidate_limit=2, trace=True,
    )
    ref_smallest = sorted(
        chain(smallest_seed, chain.from_iterable(smallest_blocks)),
        key=lambda item: item[0],
    )[:2]
    assert got_smallest == ref_smallest == [(0, 5), (1, 3)]
    print(f"  result={got_smallest}; reference={ref_smallest}")
    print("PASS: thresholded selection matches full-sort references")


if __name__ == "__main__":
    main()
```

在 learning_notes 根目录执行：

```bash
python3 operators/code/deepselect_threshold_demo.py
```

预期最后几行：

```text
result=[(10, 5), (9, 3)]; reference=[(10, 5), (9, 3)]
...
result=[(0, 5), (1, 3)]; reference=[(0, 5), (1, 3)]
PASS: thresholded selection matches full-sort references
```

该脚本让你看到阈值怎么收紧、候选怎样周期性重选，以及最大 / 最小选择如何对偶。它没有模拟 TMA、warp、shared-memory bank 或 PTX 指令，因此不代表 GPU 性能。

### 7. 性能取舍与常见错误

- **候选池大，阈值更新太晚：**筛选率低，incoming 写入和重选都变贵。
- **重选过于频繁：**radix histogram、同步和 survivor 搬运开销盖过过滤收益。`candidate_limit` / `B2` 是“候选存储压力”和“重选成本”的折中。
- **把抽 k 个样本等同 DeepSelect init：**两者候选比例可能相差几十倍；DeepSelect 是在更大的窗口中精确选 k 个，初始阈值更有代表性。
- **忽视重复值与 tie 语义：**value-only TopK 可以任意选取 cutoff 上的等值项，但 pivot 重建仍须统计 `> pivot` 与 `== pivot` 并分配 quota；若要求稳定的 index tie-break，则比较键必须包含 index。
- **把值顺序当作索引顺序：**TopK 选出了什么元素，与最终是否按值或 index 排列是两种语义。
- **把 CUDA 代码当成可在 Mac 直接跑：**此实现依赖 NVIDIA CUDA；CPU 脚本只验证算法模型。

复杂度直觉：每个输入值只做一次全局扫描；昂贵的精确选择只在容量为 `O(k + B + B2)` 的共享内存候选集合上重复进行。DeepSelect 文档在随机块顺序假设下给出总 TopK 处理量的期望界；它不是对所有输入分布、所有硬件和所有配置的统一延迟保证。

### 自我检查（含答案）

1. 为什么初始窗口不是随便抽 `k` 个值？
   - **答：**样本极值阈值通常很松；对最小 TopK，随机抽 k 个后用最大值过滤，约 `k/(k+1)` 的值仍会留下。DeepSelect 在更大的初始窗口里精确选 k 个，阈值更紧。
2. 最大 TopK 中，当前阈值 `T` 为什么可以安全丢弃 `x <= T`？
   - **答：**已处理集合里已有 k 个值 `>= T >= x`，所以 x 无法进入全局前 k 大。
3. 阈值为何单调不降？
   - **答：**处理集合只增不减；已处理集合的第 k 大值不会因为加入新元素而下降。
4. 反向找最小 TopK 时保留条件是什么？
   - **答：**当前 k 个最小值的最大值作为 `U`，保留 `x < U` 的候选，周期性精确重选最小 k 个；重复值需 quota 处理。
5. 这个 Python 脚本验证了什么、没有验证什么？
   - **答：**验证 threshold/filter/reconstruct 的集合正确性和手算轨迹；没有验证 CUDA kernel、TMA、PTX、数值类型边界或性能。

## 进度

已经覆盖：

- [x] DeepSelect 阈值式精确 TopK：初始窗口内精确选 k、动态阈值过滤、候选区周期性重选、最大 / 最小 TopK 对偶、抽 k 样本的候选率反例。

下一知识点：

- [ ] BF16 RadixSelect：高低字节直方图如何找 pivot、如何统计相等值并分配 quota。
