# DeepGEMM MegaMoE：L1 / L2 epilogue

## 本次讲解位置

```text
章节：DeepGEMM MegaMoE 执行路径
小节：Linear1 / Linear2 的输出收尾
知识点：L1 epilogue 与 L2 epilogue 的职责、数据格式和去向
上次：`x_sf` 与 UTCCP 4×32 SF 布局，以及 producer / consumer 布局契约
下次：L2 输出进入 combine buffer 后，top-k 结果如何被 combine
PTX：本节聚焦 kernel 数据路径；相关操作涉及 TMEM load、STSM、TMA store 与对称内存写入
```

### 为什么现在讲这个

在 MegaMoE 代码里看到 “L1 epilogue”“L2 epilogue” 很容易误以为它们是 GPU 的 L1/L2 cache 阶层。但这里的分支条件实际是 `BlockPhase::Linear1` / `SharedLinear1`，而另一分支处理第二层 GEMM 的 BF16 结果。分清它们才能知道 FP8 中间激活在哪里产生、第二层 GEMM 从哪里消费，以及最终结果怎样回到 token 所在 rank；也能避免把 ring-buffer 的 `l1_empty_count` / `l2_empty_count` 错当成 cache 命中计数。

## 1. 心智模型：两个 GEMM，两种“收尾”

MegaMoE 的 expert MLP 可以先简化为：

```text
路由后的 expert token
    │
    ▼
Linear1 GEMM：产生 gate / up 两路值
    │
    └─ L1 epilogue：SwiGLU + 路由权重 + FP8/scale，生成下一层输入
                    │
                    ▼
              Linear2 GEMM
                    │
                    └─ L2 epilogue：转 BF16，按 token 来源 metadata 写入 combine buffer
```

所以，L1/L2 在这个命名里是 **Linear1 / Linear2 阶段**，不是片上 L1 cache、L2 cache，也不是“第一级/第二级 epilogue 算法”。“Epilogue”是 GEMM 累加完成后，对累加值做激活、类型转换、量化、重排、同步并写到下一阶段的位置。

## 2. 两条路径对照

| 项目 | L1 epilogue | L2 epilogue |
|---|---|---|
| 对应阶段 | 第一层 Linear1 GEMM | 第二层 Linear2 GEMM |
| 输入 | TMEM 中的 GEMM 累加值；gate/up 成对解释 | TMEM 中的 GEMM 累加值 |
| 主要计算 | `SiLU(gate) * up`，并乘 top-k 路由权重 | 通常不再做 SwiGLU；把结果整理成 BF16 |
| 输出格式 | FP8 E4M3 激活 + UE8M0 scale | BF16 输出 |
| 输出位置 | 本地/共享的 L1 输出池，供 Linear2 读取 | 对应源 rank 的 combine token buffer，远端写入可经 NVLink |
| 关键同步 | 等 TMA store 完成后发布 L2 可读依赖；释放 L1 ring 槽 | 共享内存写完后读出，再发布/完成远端写，保护 SMEM 复用 |

“L1 输出池”是中间张量的 pipeline/ring 存储，不是 GPU 的 L1 cache；“L2 output”同理是阶段命名。

## 3. L1 epilogue 逐步做什么

仓库源文件：

`/Users/saboxu/Downloads/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`

可从这些位置对照阅读：

- 1058 行附近：`Linear1` / `SharedLinear1` 分支；
- 1066 行附近：注释标明 `Unified L1 epilogue: SwiGLU in-place`；
- 1100 行附近：gate/up、路由权重和 SwiGLU 计算；
- 1150–1191 行附近：跨 warp amax、算 UE8M0 scale、转 FP8 并写 SMEM；
- 1193–1220 行附近：写给后续 Linear2 使用的 scale；
- 1220–1240 行附近：TMA 将中间激活写入 L1 输出池，并发布依赖/槽位计数。

对一个元素而言，核心数学关系是：

```text
h = SiLU(gate) * up * route_weight
SiLU(x) = x / (1 + exp(-x))
```

随后 kernel 在对应量化分组上求 `amax`，构造 UE8M0 scale，把 `h` 缩放并转换为 FP8 E4M3。实际路径还包括 activation clamp、分组布局、跨 warp amax 归约和 SF 地址变换；因此上式表达语义，不是完整的位精确量化定义。

注意 top-k 权重在此实现中被提前乘到 Linear2 的输入激活上。若 Linear2 是无 bias 的线性变换，实数算术下有：

```text
(route_weight * h) @ W2 = route_weight * (h @ W2)
```

这样 L2 输出可直接送进 combine 目标槽。实际 kernel 中间还存在 FP8 量化，所以量化误差会影响严格的数值等价；这是实现安排，不是所有 MoE kernel 都必须采用的权重放置位置。

### L1 路径的同步不是装饰

1. L1 epilogue warpgroup 从 TMEM 分块加载累加结果；代码用有限个 `raw_values` buffer 做预取，避免一次读完整个 warpgroup tile。
2. warp 计算 SwiGLU 与 amax。warp-pair 间通过 shared memory 暂存 amax 并同步，得到量化 scale。
3. warp 用 STSM 将 FP8 数据写入 shared memory；scale 按后续 block-scaled GEMM 需要的布局写入 SF buffer。
4. 一个 elected thread 发起 TMA store；全体需要等 `tma_store_wait` 和相应 barrier，才能认为数据已写出并安全发布依赖。
5. ring 模式下更新 `L2KBlockDependency` / `l1_empty_count` 一类计数，表示下游依赖和 buffer 槽生命周期。共享 expert 路径使用对应的 shared counter。

第 4 步是异步边界：**发起 TMA store 不等于 store 已经完成**。若过早复用 shared memory 或让 Linear2 消费尚未发布的数据，会出现竞争或偶发错误。

## 4. L2 epilogue 逐步做什么

源文件 1273 行附近的注释直接写着：

```text
L2 BF16 epilogue: write GEMM output to remote combine buffer via NVLink
```

它的关键数据路径如下：

1. 从 TMEM 加载第二层 GEMM 的输出分块。
2. 对每个 M 行读取/缓存 token source metadata。非 shared 路径从 workspace 取 metadata；shared 路径可从本地 pool 信息构造。metadata 给出 `(dst_rank_idx, dst_token_idx, dst_topk_idx)`。
3. 各 warp 用 STSM 将 BF16 数据写入经过 swizzle 的 SMEM 暂存区。
4. warpgroup barrier 确认 SMEM 数据齐备后，按 lane/row 读取打包的 `float4`。
5. 通过 `sym_buffer.map(dst_ptr, dst_rank_idx)` 把输出写到目标 rank 对应的 `combine_token_buffer`。这是逻辑上的远端 symmetric-buffer 写路径；代码注释指出该写出经 NVLink。
6. barrier 保护 SMEM 在远端写读取结束后再用于下一个 store block。

`dst_topk_idx` 用来选 combine buffer 中相应 top-k 的区域，`dst_token_idx` 选择 token，`dst_rank_idx` 选择 rank。这里不是在 epilogue 中把多个 expert 的结果求和；它是把各路结果写到 combine 能找到的位置。真正的 top-k 聚合属于后续 combine 阶段。

## 5. 用一个小数值例子重建两段语义

假设两个 token 的 Linear1 gate/up 标量值分别为：

| routed row | gate | up | route weight | 目的地 `(rank, token, top-k)` |
|---:|---:|---:|---:|---|
| 0 | 2 | 3 | 0.25 | `(0, 17, 1)` |
| 1 | -1 | 4 | 0.75 | `(1, 42, 0)` |

先算 L1 epilogue 的浮点语义：

```text
row 0: SiLU(2) = 2 / (1 + exp(-2)) ≈ 1.761594
       1.761594 * 3 * 0.25 ≈ 1.321196

row 1: SiLU(-1) = -1 / (1 + exp(1)) ≈ -0.268941
       -0.268941 * 4 * 0.75 ≈ -0.806824
```

假设 Linear2 是一个输入、两个输出的无 bias 线性层，权重为 `[2, -1]`，则 L2 输出的语义结果是：

```text
row 0: [1.321196 * 2, 1.321196 * -1] = [ 2.642391, -1.321196]
row 1: [-0.806824 * 2, -0.806824 * -1] = [-1.613649,  0.806824]
```

L2 epilogue 并不按 row 顺序把这两行写回原始输入 buffer，而是依据 metadata 分别写到 `(rank=0, token=17, top-k=1)` 和 `(rank=1, token=42, top-k=0)` 对应的 combine 槽。

## 6. 可运行的 CPU 语义模型

下面的完整脚本只模拟数学语义与目的地路由。它**不模拟** E4M3/UE8M0 的位级量化、TMA、TMEM、barrier、NVLink 或 GPU 并发；这些硬件路径需读上面的 CUDA 源码并在支持的 NVIDIA GPU 上测试。

保存为 `megamoe_epilogue_semantics.py`：

```python
import math


def silu(x):
    return x / (1.0 + math.exp(-x))


def l1_epilogue(gates, ups, route_weights):
    """教学用标量参考：Linear1 的 gate/up -> weighted SwiGLU。"""
    if not (len(gates) == len(ups) == len(route_weights)):
        raise ValueError("gates, ups, route_weights must have equal length")

    intermediate = []
    for gate, up, weight in zip(gates, ups, route_weights):
        intermediate.append(silu(gate) * up * weight)
    return intermediate


def l2_epilogue(intermediate, linear2_weights, destinations):
    """教学用无 bias Linear2，并按 metadata 放入 combine 槽。"""
    if len(intermediate) != len(destinations):
        raise ValueError("one destination is required for each routed row")

    combine_slots = {}
    for x, destination in zip(intermediate, destinations):
        rank, token, topk = destination
        if rank < 0 or token < 0 or topk < 0:
            raise ValueError("destination indices must be non-negative")
        combine_slots[(rank, token, topk)] = tuple(
            x * weight for weight in linear2_weights
        )
    return combine_slots


def main():
    gates = [2.0, -1.0]
    ups = [3.0, 4.0]
    route_weights = [0.25, 0.75]
    destinations = [(0, 17, 1), (1, 42, 0)]
    linear2_weights = [2.0, -1.0]

    intermediate = l1_epilogue(gates, ups, route_weights)
    combine_slots = l2_epilogue(
        intermediate, linear2_weights, destinations
    )

    print("L1 intermediate:", [round(x, 6) for x in intermediate])
    for destination, values in sorted(combine_slots.items()):
        print(f"combine{destination}:", [round(x, 6) for x in values])


if __name__ == "__main__":
    main()
```

运行：

```bash
python3 megamoe_epilogue_semantics.py
```

预期输出：

```text
L1 intermediate: [1.321196, -0.806824]
combine(0, 17, 1): [2.642391, -1.321196]
combine(1, 42, 0): [-1.613649, 0.806824]
```

## 7. 常见误解和症状

| 误解/错误 | 会造成什么问题 | 可能看到的症状 |
|---|---|---|
| 把 L1/L2 epilogue 当作 L1/L2 cache 处理 | 把计算阶段和 buffer 同步混为一谈 | 误读代码中的 `Linear1` 分支、ring empty/full counter |
| 认为 L1 epilogue 只做 SwiGLU | 漏掉 top-k 权重、amax、FP8 和 SF 输出 | Linear2 输入数值或量化尺度与参考不符 |
| 认为 L2 epilogue 已经做完 top-k combine | 混淆分发与归约 | 误以为一个输出位置已包含所有 expert 的加权和 |
| 将发起 TMA / NVLink 写当作完成 | 过早消费或复用缓冲区 | race、偶发数据损坏、不同 tile/负载下结果变化 |
| 不用 metadata，而按本地 routed-row 顺序写回 | routed 顺序和原 token 位置不一定相同 | token 输出串位，或 top-k 分支写入错误 |
| 把 CPU 语义模型当成硬件验证 | 模型没有模拟 FP8 rounding、barrier 和远端写 | Python 测试通过但 GPU kernel 仍可能有数值/同步错误 |

## 8. 验证边界与阅读命令

在当前 DeepGEMM checkout 可以直接定位主要分支：

```bash
rg -n "Unified L1 epilogue|L2 BF16 epilogue|BlockPhase::Linear1" \
  /Users/saboxu/Downloads/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh
```

CPU 脚本能验证公式与 metadata 路由映射。它不证明 DeepGEMM 的 CUDA kernel 正确或性能达标；验证真实执行需要与该实现兼容的 Blackwell GPU、CUDA/CUTLASS 构建环境，以及对应 MegaMoE 测试。

## 9. 本章进度

已覆盖：

- [x] 区分 Linear1 / Linear2 epilogue 与 GPU L1/L2 cache
- [x] L1 epilogue 的 SwiGLU、路由权重、FP8/UE8M0 中间输出
- [x] L2 epilogue 的 BF16、source metadata 和 combine buffer 写出
- [x] 两段之间的 TMA、barrier 与 ring-buffer 生命周期语义
- [x] 用 CPU 语义模型和具体数值追踪两段路径

下一知识点：L2 epilogue 写入 combine buffer 后，dispatch 的 top-k 结果如何被 pull/combine 归并回原 token。

## 自测（含答案）

1. **这里的 L1/L2 指 GPU 的哪两级 cache 吗？**
   答：不是；在这条代码路径中指 Linear1 和 Linear2 GEMM 阶段。
2. **L1 epilogue 产出的主要数据是什么，谁消费它？**
   答：FP8 激活及其 UE8M0 scale，供后续 Linear2 GEMM 消费。
3. **L2 epilogue 根据什么决定输出写到哪里？**
   答：token source metadata 中的目标 rank、token 和 top-k 槽信息。
4. **L2 epilogue 写完一个 expert 结果后，是否已完成 top-k 求和？**
   答：没有；它先把结果放进 combine buffer，后续 combine 再归并。
5. **发出 TMA store 后能否立刻复用共享内存？**
   答：不能；必须遵守 TMA completion 和相应 barrier 的同步契约。
