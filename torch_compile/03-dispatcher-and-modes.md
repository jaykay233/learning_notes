# Dispatcher、DispatchKey、TorchDispatchMode，以及和 AOT 的衔接

> 承接 [02-aot-autograd.md](./02-aot-autograd.md)。  
> 整理讨论：`torch_dispatch` 钩子、`DispatchKeySet`、vtable、Mode 栈、TLS，以及 AOT 如何「故意让 Mode 排到前面」。

---

## 1. `torch_dispatch` / `__torch_dispatch__` 是什么

**Dispatcher 上的通用挂钩**，不只是「给算子选后端」。

| 用法 | 做什么 |
|---|---|
| 后端 / 设备 | 改写到 XLA、自定义加速器 |
| Tensor subclass | 稀疏、量化、NestedTensor… |
| **追踪（AOT）** | 不改语义，只 `record(op, args, out)` |
| Functionalization | mutation → 纯函数 |
| FakeTensor | 只跑 meta，不跑真计算 |

选后端是应用之一；AOT 用的是 **旁路观察并记账**。

### 和 AOT 的正确说法

**不是**「先有一张图，再用钩子 DFS/BFS 遍历」。  
**而是**：执行 `joint_fw_bw`（假数据）时，每个 Aten op 经 dispatcher → 钩子被调用 → **边跑边 record**（执行驱动的追踪）。

```text
执行驱动的追踪（trace by running）
  ≠ 对已有图结构做遍历（walk an existing graph）
```

---

## 2. DispatchKey 与 64-bit bitset

### 2.1 不是「8-bit 里塞几种 key」

- `DispatchKeySet` 内部通常是 **64-bit bitset**（每位对应一类能力/后端相关信息）。  
- 源码里的 `uint8_t` 多半是把 `DispatchKey` 枚举转成整数去算 `1ULL << …`，**不是**「只有 8 种 key」。  
- 量化 INT8 是 dtype/kernel 问题，一般 **不叫** DispatchKey 里的 “8bit key”。

另有 **functionality bits + backend bits** 打包进同一套表示（既有「干什么」也有「在哪台设备」）。

### 2.2 常见类别（可同时出现在 set 里）

| 类别 | 例子 | 干什么 |
|---|---|---|
| Backend | `CPU`, `CUDA`, `XLA`, `MPS`, `Meta`… | 真正算数 |
| Autograd | `Autograd` / `AutogradCUDA`… | 挂反传，再 redispatch |
| 功能 / Mode | `Functionalize`, `Autocast`… | 改语义或插一层 |
| Python / 扩展 | `Python`、Mode 相关 | 进 `__torch_dispatch__` |
| Fake 等 | 历史 `Fake` key；现多靠 Mode | 假执行 / 建图 |

### 2.3 多 key 并存，还是只能一种？

**64-bit 里可以同时开很多位**；不是整个 set 永远只有一种。

```text
DispatchKeySet（例如同时有 Autograd* 和 CUDA）
  → 取优先级最高的一个 key
  → 查该 op 的 dispatch table，跳到对应 KernelFunction
  → 该层干完常 redispatch（mask/exclude 掉自己）
  → 再用剩下的 set，取下一个最高 → 例如落到 CUDA
```

| 概念 | 含义 |
|---|---|
| Set 里 | 多种类别常 **同时存在** |
| 这一跳 | 只按 **一个** 最高优先级 key 进表 |
| 一次 `aten::mm` | 可能 **连跳好几次**（Autograd → … → CUDA） |

优先级直觉：Autograd / Python / Mode 相关 … **高于** CUDA / CPU。

---

## 3. vtable / dispatch table 是啥

类比 C++ 虚表：

```text
C++:     对象 → vtable[虚函数下标] → 函数指针
PyTorch: 每个算子 → dispatch table[DispatchKey] → KernelFunction
```

- **不是** Tensor 上挂一张巨大虚表。  
- **是** 每个 op（如 `aten::mm`）一张表：  
  `CPU → cpu_mm`，`CUDA → cuda_mm`，`Autograd → autograd_mm`，`Python → 进 Mode/subclass`…  
- 用算出来的最高优先级 key 当「下标」间接跳转 → 说法上像 vtable。

---

## 4. Autograd/CUDA 核 vs `__torch_dispatch__`

大方向：多 key → 先高优先级 → redispatch 往下。  
但 **Autograd / CUDA 一般不走 `__torch_dispatch__`**。

| 路径 | 走什么 |
|---|---|
| Autograd / CUDA / CPU … | dispatch table 里的 **KernelFunction**（多为 C++） |
| Python / TorchDispatchMode / 实现了 `__torch_dispatch__` 的 subclass | 相关 key 够高时 → **`__torch_dispatch__`** |

普通训练一步常见：

```text
{Autograd*, CUDA} → Autograd 核 → redispatch → CUDA 核
```

AOT 假跑抓图：打开 Mode，让 **Python/Mode 路径** 排到前面，在钩子里 meta + record；往往到不了真 CUDA GEMM。

---

## 5. TorchDispatchMode 是啥结构

**TorchDispatchMode** = context manager + 必须实现的 `__torch_dispatch__`，靠 **线程局部 mode 栈** 插入 Python dispatch 路径。

```python
class TorchDispatchMode:
    def __torch_dispatch__(self, func, types, args, kwargs):
        ...  # 拦截 aten op

    def __enter__(self):
        push(self)   # → TLS mode stack

    def __exit__(...):
        pop()
```

TLS 心智模型：

```text
TorchDispatchModeTLS（每线程一份）
  ├─ stack_: [user_mode, ...]
  └─ infra 槽位: FAKE / PROXY / FUNCTIONAL …
```

调用时：

```text
with LoggingMode():
  with FakeTensorMode():
    y = x.sin()

aten.sin → Python/Mode 路径 → 当前 mode.__torch_dispatch__
  内部再调 func(*args) 时：默认进下一 mode / 下层（当前 mode 会先处理掉，避免死循环）
```

| | Tensor subclass | TorchDispatchMode |
|---|---|---|
| 挂在哪 | 某类 tensor | **with 作用域** |
| factory（如 `torch.ones`） | 难拦 | **能拦** |

`FakeTensorMode` / `FunctionalTensorMode` 都是这类 Mode（基建 / infra）。

---

## 6. TLS 是啥

**TLS = Thread-Local Storage（线程局部存储）**。

- 每个线程一份自己的 Mode 栈、dispatch include/exclude。  
- 线程 A `with FakeTensorMode()` **不会**影响线程 B。  
- `with` 结束只在该线程 pop/恢复。

---

## 7. AOT「故意让 Mode 排到前面」怎么实现

不是改写 CUDA 核在表里的优先级，而是：

```text
① DispatchKey 固有优先级：Mode/Python 相关可以高于 CUDA
② 抓图前 with FakeTensorMode / FunctionalTensorMode
     → push TLS mode 栈
     → 打开相关 TLS include
③ 本次 aten 调用算出的最高 key 变成 Mode/Python 路径
     → __torch_dispatch__ 里 meta + record
     → 通常不再落到真 CUDA GEMM
```

伪代码：

```python
def aot_capture(fn, example_inputs):
    fake_in = [to_fake(t) for t in example_inputs]
    with FunctionalTensorMode():
        with FakeTensorMode():
            joint_fw_bw(fake_in, fake_grads)  # 每个 op 先撞 Mode
    # with 结束 → pop → 恢复普通 Autograd→CUDA
```

Infra mode（FAKE / FUNCTIONAL 等）还有固定槽位/顺序，避免和用户 mode 叠放时顺序乱套。

对比：

```text
未开 Mode:  {Autograd*, CUDA} → Autograd 核 → CUDA 核
AOT 抓图:   Mode 在 TLS 上 → 先 __torch_dispatch__ →（常）不到真 CUDA
```

---

## 8. 一张总图

```text
Tensor/TLS 上的 key bits（可多位）
  → 算最高优先级 DispatchKey
  → 该 op 的 dispatch table（类 vtable）[key]
       ├─ Autograd / CUDA / …  → KernelFunction
       └─ Python / Mode        → TorchDispatchMode.__torch_dispatch__
              └─ AOT：Fake meta + record → joint 图
```

**Key set = 当前生效能力集合；vtable/table = 每个算子在各 key 上挂的实现；Mode = 可入栈的作用域钩子（TLS）；AOT = 打开 Mode 做执行期追踪。**

---

## 9. 相关文档

- [01-dynamo-and-fx.md](./01-dynamo-and-fx.md) — Dynamo / FX  
- [02-aot-autograd.md](./02-aot-autograd.md) — joint 图、FakeTensor、partition  
- [PyTorch dispatcher walkthrough](https://github.com/pytorch/pytorch/wiki/PyTorch-dispatcher-walkthrough)  
- [ezyang: Let's talk about the PyTorch dispatcher](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/)
