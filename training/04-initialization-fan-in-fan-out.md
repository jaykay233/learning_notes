# 参数初始化中的 `fan_in` 与 `fan_out`

## 本次讲解位置

- 章节：`training` / 参数初始化
- 小节：用权重张量形状判断 fan 的含义
- 知识点：fan-in / fan-out 与 Kaiming 初始化模式
- 上次：autograd、并行训练与数据随机性基础
- 下次：结合实际网络层检查初始化尺度与训练稳定性

## 为什么现在讲这个

初始化函数常要求选择 `mode='fan_in'` 或 `mode='fan_out'`。如果只记住英文名字、却没从权重形状数连接数，就容易在输入输出维理解反了，导致权重初始方差按错误方向缩放。

## 直觉模型

- **fan-in**：一个输出单元接收多少个输入连接。
- **fan-out**：一个输入单元把值送往多少个输出连接。

对全连接层 `Linear(in_features=5, out_features=3)`，PyTorch 权重形状是 `[out_features, in_features]=[3,5]`：

```text
每个输出单元汇入 5 个输入 => fan_in  = 5
每个输入单元连到 3 个输出 => fan_out = 3
```

对卷积核权重 `[out_channels, in_channels, kH, kW]`，连接数还要乘上卷积核的空间面积：

```text
fan_in  = in_channels  × kH × kW
fan_out = out_channels × kH × kW
```

## 为什么初始化要看 fan

若一个输出是多个独立输入的加权和，输入数越多，输出方差通常越容易放大。因此初始化会随 fan 调整权重方差，避免信号在很多层后爆炸或逐渐消失。

PyTorch Kaiming 初始化的简化尺度为：

```text
mode="fan_in" : weight variance ≈ gain² / fan_in
mode="fan_out": weight variance ≈ gain² / fan_out
```

ReLU 常用 gain `sqrt(2)`，所以方差约为 `2/fan`。`fan_in` 侧重保持前向信号尺度；`fan_out` 侧重保持反向梯度尺度。此处是初始化推导直觉，具体分布（uniform/normal）、非线性 gain、分组卷积和实现细节仍应看调用参数。

### 数值例子

对于 `[3,5]` 的线性权重和 ReLU：

- `fan_in=5`：方差目标约 `2/5=0.4`。
- `fan_out=3`：方差目标约 `2/3≈0.667`。

同一个权重形状选择不同 mode，尺度不同；`fan_out` 并不是简单地把网络“向后多算一层”。

## 可运行检查

```python
import math
import torch

linear = torch.nn.Linear(5, 3, bias=False)
weight = linear.weight
fan_out, fan_in = weight.shape
print("weight shape:", tuple(weight.shape))
print("fan_in:", fan_in, "fan_out:", fan_out)

for mode in ("fan_in", "fan_out"):
    torch.nn.init.kaiming_uniform_(weight, mode=mode, nonlinearity="relu")
    print(mode, "finite:", bool(torch.isfinite(weight).all()))

conv = torch.nn.Conv2d(4, 8, kernel_size=3, bias=False)
print("conv fan_in:", 4 * 3 * 3, "fan_out:", 8 * 3 * 3)
```

运行：`python fan_demo.py`（需要 PyTorch）。预期关键信息：

```text
weight shape: (3, 5)
fan_in: 5 fan_out: 3
fan_in finite: True
fan_out finite: True
conv fan_in: 36 fan_out: 72
```

uniform 初始化会采样随机数，具体权重数值不固定；检查关注的是维度计算及输出有限，而不是某一组样本值。

## 常见误区

1. 对 `Linear` 把权重 `[out,in]` 误看成 `[in,out]`：会把 fan-in / fan-out 颠倒。
2. 卷积层只数 channels、忽略 kernel 高宽：卷积 fan 还要乘 `kH×kW`。
3. 把 `mode='fan_out'` 理解成“输出通道更多”：它是选择哪种 fan 计算初始化尺度，不会改变网络结构。
4. 把 Kaiming 当成所有激活函数都通用：其 gain 要和非线性匹配；Sigmoid、Tanh 等应按相应初始化假设选择。

## 自测

1. `Linear(5, 3)` 权重形状是什么？——`[3,5]`。
2. 该层的 fan-in / fan-out 分别是多少？——5 / 3。
3. `Conv2d(4,8,kernel_size=3)` 的 fan-in / fan-out 是多少？——36 / 72。
4. Kaiming `fan_in` 和 `fan_out` 分别侧重什么？——前者保持前向信号尺度，后者保持反向梯度尺度。
