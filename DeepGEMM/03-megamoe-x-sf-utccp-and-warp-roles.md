# DeepGEMM MegaMoE：`x_sf`、UTCCP 4×32 布局与 A/SFA、B/SFB warp

## 本次讲解位置

```text
章节：DeepGEMM MegaMoE 执行路径
小节：FP8 scale 与 block-scaled GEMM 的 producer/consumer 布局
知识点：x_sf 的语义、4×32 转置为何发生、不同 warp 负责什么
上次：workspace、expert slot、metadata、跨 rank pull
下次：Linear1/Linear2 epilogue 如何消费并生产量化数据
PTX：UTCCP 是 SMEM → TMEM 的专用 scale-copy 路径；tcgen05.mma/UMMA 消费矩阵与 scale 做 Tensor Core 运算
```

### 为什么现在讲这个

pull 阶段拿到的 FP8 token 数据只是压缩后的数值；没有匹配的 scale 就无法按正确幅值参与 GEMM。即使 token 与 scale 都搬对了，scale 的物理顺序若不符合 UTCCP/UMMA 约定，硬件 consumer 仍可能把 scale 对到错误的 M/K 坐标。因此要同时看“scale 表示什么”“scale 放在哪”“哪个 warp 搬它”。

## 1. `x_sf` 是什么

`x_sf` 可以按名字拆解为：

- `x`：输入激活矩阵/输入 token 数据；
- `sf`：scale factor；
- `x_sf`：与输入 `x` 的 FP8 激活相配套的 scale-factor 数据。

在 MegaMoE pull 场景中，dispatch 不只取远端的 FP8 token bytes，还会取对应的 `x_sf`，再把二者放入本地 L1 输入池。`x_sf` 不是 token 向量本身，也不是 top-k 权重或 expert bias。它记录量化块的尺度信息，供后续 block-scaled GEMM 将低精度输入按约定缩放。

注意 SF 往往是按一个小的元素块而非每个 scalar 独立存储；具体一份 SF 覆盖哪些 M/K 元素、是什么编码，要看对应 GEMM 数据类型和 kernel 配置。不要只凭变量名推断精确的分组尺寸。

## 2. SFA / SFB 与 warp 名称

矩阵乘有两个输入操作数 A、B。block-scaled GEMM 对应地有两组 scale：

- **SFA**：与矩阵操作数 A 对应的 scale factors；
- **SFB**：与矩阵操作数 B 对应的 scale factors。

MegaMoE kernel 中一些 warp 被分配成专门的 TMA loader：

| warp 角色 | 搬运对象 | 直观职责 |
|---|---|---|
| A/SFA warp | 当前 GEMM 的 A tile 与 SFA | 同一生产角色处理 A 操作数及其 scale |
| B/SFB warp | 当前 GEMM 的 B tile 与 SFB | 同一生产角色处理 B 操作数及其 scale |
| MMA issue warp | 等待输入准备好，执行 scale copy / 发起 MMA 路径 | 把 scale 放入 TMEM，并发起 UMMA/tcgen05 MMA |

“A/SFA warp”不是一种特殊硬件 warp 类型；它是 kernel 的角色命名。TMA 也不是 32 个 lane 各自做 32 次标量 copy：通常由满足条件的 elected thread 发起描述符 copy，TMA 硬件完成大块传输；warp/warpgroup 通过 barrier 协议等待和交接。

高层 producer/consumer 关系：

```text
A/SFA loader ─┐
               ├─ TMA 到 SMEM → mbarrier/full barrier → MMA warp 消费
B/SFB loader ─┘                                  │
                                                 ├─ UTCCP：SF 从 SMEM 到 TMEM
                                                 └─ tcgen05.mma / UMMA 使用 operands + SF
```

具体 warp 编号依赖 kernel 的编译期配置。读实现时按其 `warp_idx` 分支和常量偏移确认角色，不要把某个绝对 warp id 当成所有 kernel 的普遍规定。

## 3. 4×32 转换：把 128 个 M/token 位置重排

MegaMoE 这条 SF 路径按每 128 个 M/token 位置分一组，拆成 4 行 × 32 列来做置换。令源索引为 `idx`：

```text
q = idx // 32       # 0..3，四个 32 元素段中的哪一段
r = idx % 32        # 0..31，段内位置
目标位置 = 4*r + q
```

等价于把一个概念上的 `4 × 32` 行优先矩阵转成 `32 × 4` 行优先矩阵。源位置写为 `idx=32*q+r`；目标 `4*r+q`。

具体轨迹：固定段内位置 `r=5`，看四个段中的值：

| 来源 `idx=32*q+5` | `q` | `r` | 目标 `4*r+q` |
|---:|---:|---:|---:|
| 5 | 0 | 5 | 20 |
| 37 | 1 | 5 | 21 |
| 69 | 2 | 5 | 22 |
| 101 | 3 | 5 | 23 |

所以同一 `r=5` 在四段中的四个 SF，转置后落到 20、21、22、23 四个相邻位置。数值没有变化；变化的是物理地址/索引排列。

再看小块示意（只画 q=0..3、r=0..3）：

```text
源 4×32 的左上角：         转置后对应位置：
q\r  0   1   2   3         r\q  0   1   2   3
 0   a00 a01 a02 a03          0   a00 a10 a20 a30
 1   a10 a11 a12 a13          1   a01 a11 a21 a31
 2   a20 a21 a22 a23          2   a02 a12 a22 a32
 3   a30 a31 a32 a33          3   a03 a13 a23 a33
```

该表强调它是“地址重排”，不是把 scale 数值做转置计算或乘法。

## 4. 为什么要转置

直接原因是 consumer 的布局契约：UTCCP 按它规定的 4×32 组织从 SMEM 搬 SF 到 TMEM，之后 block-scaled UMMA 会按矩阵 M/K 坐标消费这些 scale。producer 若给出不兼容的顺序，consumer 会把 scale 解释到错误的矩阵位置。

布局转换可以有两种付费时机：

1. **producer 预变换**：dispatch/pull 把 SF 写进 UTCCP 友好的地址，consumer 路径更直接；
2. **consumer 显式转置**：数据先按普通顺序进 SMEM，再由 warp 读写重排，然后再消费。

MegaMoE 的某些输入 SF 路径会在生产/接收阶段预先变换布局，避免把转换留到每次 consumer 使用时。普通 GEMM 的实现也可能有显式的 UTCCP transposer warp。两者差别是转换成本在哪里承担，不代表 SF 数值语义不同。

**不要过度概括为“转置就是为了避免 bank conflict”。**这里最直接、可验证的要求是 UTCCP 与 UMMA 的 scale 布局一致。相邻写入可能有访存组织上的好处，但性能收益要针对具体路径测量。

## 5. scale 和 token 数据如何一起到 GEMM

可用这条流水线理解：

```text
远端 FP8 token + x_sf
  -> dispatch/pull：按 chunk 取回
  -> 本地 L1 token pool + UTCCP 友好的 SF pool
  -> A/SFA 或 B/SFB warp：TMA 装载 tile 到 SMEM
  -> barrier 确认 producer 完成
  -> MMA warp：UTCCP 把 SF 从 SMEM 搬入 TMEM
  -> tcgen05.mma / UMMA：按同一矩阵 tile 使用 operand 与 scale
```

scale-layout 转换本身不完成反量化；它确保硬件 MMA 在正确的坐标使用 scale。具体乘法/缩放的数值语义由对应 block-scaled MMA 格式和指令定义。

## 6. 常见误解与症状

| 误解 | 正确理解 | 可能症状 |
|---|---|---|
| `x_sf` 是另一份 token 特征 | 它是与 FP8 输入 X 配套的 scale-factor 信息 | 把 bytes 当激活值，导致输入解释完全错误 |
| SFA/SFB 是两种 scale 数值精度 | A/B 是操作数角色，SF 是对应 scale 数据 | A/B scale 指针交换，两个操作数尺度错配 |
| 4×32 是每个 token 有 128 个 SF | 这是 M/token 轴 128 个位置的排列示意；K 方向 scale 分组另有维度 | SF buffer stride/坐标认错 |
| 转置改变了 scale 值 | 转置只改变地址排列 | 对值本身做额外变换，结果错误 |
| UTCCP 会替你猜布局 | consumer 按约定解释输入 | scale 与 M/K 坐标错配，GEMM 误差显著 |
| A/SFA warp 是 32 lanes 手动逐元素拷贝 | 是软件分工角色，常由 elected thread 发起 TMA | 误判参与线程数与同步范围 |
| `tcgen05.mma` 自己等价于整个 tcgen05 指令族 | 它是矩阵乘指令；tcgen05 还有其他指令 | 漏掉 TMEM copy/wait 等独立操作与完成边界 |

## 自测（含答案）

1. **`x_sf` 中的 `x` 和 `sf` 分别指什么？**  答：输入激活 X 和其 scale factor。
2. **SFA / SFB 根据什么命名？**  答：按 GEMM 的 A/B 操作数归属命名。
3. **源索引 69 经 4×32 映射后到哪里？**  答：`q=2,r=5`，目标 `4*5+2=22`。
4. **来源 5、37、69、101 为什么映射为连续 20..23？**  答：它们的 `r=5` 相同、`q=0..3`，转置后四个 q 成为连续内层位置。
5. **UTCCP 转换是否改变 scale 值？**  答：不改变，只把值放到 consumer 约定的物理索引。
6. **A/SFA warp 等名字表示什么？**  答：kernel 软件角色，不是硬件定义的新 warp 类型。
