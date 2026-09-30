# PyTorch 自动微分：梯度追踪、保留与 `derivatives.yaml`

## 本次讲解位置

- 章节：`training` / 自动微分与梯度
- 小节：张量何时追踪、何时保存梯度，以及 PyTorch 如何描述导数规则
- 知识点：`requires_grad`、`retain_grad()`、`create_graph`、`retain_graph`、YAML 的 `result`
- 上次：开始整理训练相关概念
- 下次：并行训练里的 rank / group 如何映射到物理 GPU

## 为什么现在讲这个

读 autograd 源码或打印张量时，最容易把三件事混在一起：**建立正向运算图**、**在反向时计算梯度**、**把某个梯度留在 `.grad` 属性里**。它们不是同一个开关。区分它们，才能解释为什么某个 tensor 的 `.grad` 是 `None`，或为什么二阶导数还可以继续算。

## 先记住这四件事

| 名称 | 控制什么 | 典型用途 |
|---|---|---|
| `requires_grad=True` | 是否追踪该张量参与的运算 | 让参数/输入可求导 |
| `retain_grad()` | 是否把非叶子张量的梯度保存到 `.grad` | 调试中间激活梯度 |
| `create_graph=True` | 是否追踪“求梯度这一步”的运算 | 二阶导、梯度惩罚 |
| `retain_graph=True` | 本次反向后是否不释放原来的反向图 | 对同一前向图再反向一次 |

`requires_grad` 是张量属性；`retain_grad()` 是对张量调用的方法。不存在常用的张量属性 `retains_grads` 来替代 `requires_grad`。

## 一个完整可运行例子

保存为 `autograd_basics.py`：

```python
import torch

x = torch.tensor(2.0, requires_grad=True)  # 叶子张量
h = x * 3.0                                 # 中间的非叶子张量
h.retain_grad()                             # 请求反向后把 h.grad 留下来
y = h.square()                              # y = (3x)^2

y.backward()
print(f"x.grad={x.grad.item():.1f}")
print(f"h.grad={h.grad.item():.1f}")

# 再单独演示 create_graph：g = dy/dx，且 g 自己还可以继续求导。
x2 = torch.tensor(2.0, requires_grad=True)
y2 = x2**3
(g,) = torch.autograd.grad(y2, x2, create_graph=True)
(g2,) = torch.autograd.grad(g, x2)
print(f"g={g.item():.1f}, g2={g2.item():.1f}")
```

运行：

```bash
python autograd_basics.py
```

预期输出：

```text
x.grad=36.0
h.grad=12.0
g=12.0, g2=12.0
```

### 把数字手算一遍

这里 `h=3x=6`，`y=h²=36`：

- `dy/dh = 2h = 12`，因此 `h.grad=12`。
- `dy/dx = (dy/dh)(dh/dx)=12×3=36`，因此 `x.grad=36`。
- `y2=x2³` 的一阶导数 `g=3x2²=12`；再求导得 `g2=6x2=12`。

`x` 是叶子张量，因此默认会把梯度写入 `x.grad`。`h` 是中间张量；即使 autograd 为反向传播临时算出了流经 `h` 的梯度，默认也不保证把它保留在 `h.grad`。`retain_grad()` 请求保存这份中间梯度。

## 反向计算本身也可以建图

默认反向传播的主要目标是得到数值梯度；求导过程中产生的运算通常不会再被记录成一张可微分的图。设置 `create_graph=True`，相当于要求“算出梯度的同时，也追踪这个梯度是怎样算出来的”，于是可以对梯度再次求导。

这和 `retain_graph=True` 不同：

- `create_graph=True`：**把梯度计算过程也做成可微的**。
- `retain_graph=True`：**保留原先的前向计算图，以便再次使用**。

例如，`torch.autograd.grad(y, x, create_graph=True)` 会返回带有梯度历史的 `g`。如果只是把同一个 `y` 对不同输入反向两次，且第一次没有创建高阶图，则可能需要 `retain_graph=True`；但它本身不会让梯度变得可继续求导。

## `derivatives.yaml` 的 key / value 怎么读

PyTorch 源码中的 `derivatives.yaml` 由 codegen 使用，描述算子的反向（VJP）以及在适用时的前向模式导数（JVP）规则。可把一个条目看成：

```yaml
- name: mul.Tensor(Tensor self, Tensor other) -> Tensor
  self: mul_tensor_backward(grad, other, self.scalar_type())
  other: mul_tensor_backward(grad, self, other.scalar_type())
  result: other_t * self_p + self_t * other_p
```

- `-`：YAML 列表中的一条算子规则。
- `name`：匹配的算子 schema；`mul.Tensor` 是 Tensor × Tensor 的乘法 overload。
- `self` / `other`：**输入名**，对应的 value 是反向传播时返回给该输入的梯度公式。`grad` 是从下游传来的上游梯度。
- `result`（作为 YAML key）：描述输出在前向模式下的切向量/JVP。`self_t`、`other_t` 是输入切向量，`self_p`、`other_p` 是输入的 primal（原始值）。乘法法则是 `d(self*other)=dself*other+self*dother`。

要区分 **key `result:`** 与 value 表达式中出现的名字 `result`。在其他算子的反向公式里，`result` 也可能作为前向输出值被引用，例如 `result.scalar_type()`；此时它是输出 Tensor，不是 YAML key。

数学上，若 `z = a × b`：

- 反向 VJP：给定上游梯度 `g=∂L/∂z`，得到 `∂L/∂a=g×b`、`∂L/∂b=g×a`。
- 前向 JVP：给定输入方向 `ȧ`、`ḃ`，得到 `ż=ȧ×b+a×ḃ`。

**key 决定公式属于哪条路径，value 是公式本身。**具体算子字段以及表达式会随 PyTorch 版本变化；读源码时要以对应 checkout 中的 YAML 与 codegen 实现为准。

## 常见误区

1. `requires_grad=True` 不等于“所有相关中间张量都有 `.grad`”。它打开追踪，不负责保留每个中间梯度。
2. `retain_grad()` 不会给一个未追踪的张量凭空补出梯度；对不参与可微计算的张量调用也没有意义。
3. `create_graph=True` 不是“保留原图”的同义词；高阶求导关注的是梯度是否可微。
4. 不要把 YAML `result` key 误读成某个普通反向输入。它所在条目、冒号位置和 value 中的 `_t` / `_p` 都很关键。

## 自测

1. 为什么 `h.grad` 默认可能是 `None`，而 `x.grad` 有值？——叶子张量的 `.grad` 默认累积；中间张量需请求 `retain_grad()`。
2. 想算 Hessian 向量积，需要关注哪个开关？——通常要在求一阶导时设 `create_graph=True`。
3. 只为同一个前向图再反向一次，和对一阶梯度再求导是一回事吗？——不是；前者涉及 `retain_graph`，后者涉及 `create_graph`。
4. `result: ...` 里的 `result` 表达什么？——这是 YAML 的字段 key；在常见规则中它写前向模式输出切向量公式。
