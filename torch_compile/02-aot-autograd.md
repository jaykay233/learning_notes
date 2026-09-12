# AOTAutograd：原理与实现机制

> Ahead-Of-Time Autograd：在 **真正训练步跑起来之前**，就把某段前向对应的 **反向图也抓出来**，再交给编译器分别优化。  
> 在 `torch.compile` 里，它接在 **Dynamo（Python→FX）** 之后、**Inductor（降核）** 之前。  
> Dynamo 基础见 [01-dynamo-and-fx.md](./01-dynamo-and-fx.md)。

---

## 1. 要解决什么问题

Eager Autograd 的工作方式：

```text
前向跑一步 → 动态搭 autograd 图（每个 op 挂 grad_fn）
反向时 ← 沿这张动态图回传
```

这对正确性很好，但对 **编译器** 不友好：

- 反向公式散落在各个 `Function.backward` 里，编译器在前向结束前 **看不见完整反向计算**  
- 难以做「前向多存点东西 / 反向重算」这类 **联合** 决策（activation checkpointing、min-cut remat）  
- Dynamo 若只编译前向，反向仍完全走 eager，加速不完整  

**AOTAutograd** 的目标：对 Dynamo 交出的每一段（或 `aot_function` 包住的函数），

1. **提前**得到前向图 + 反向图（Aten IR 的 FX）  
2. 可选：在 **joint 图** 上做存/算权衡再切开  
3. 分别交给 `fw_compiler` / `bw_compiler`（常是 Inductor）  
4. 用 `torch.autograd.Function` 包回去，和周围 eager 代码无缝衔接  

「AOT」这里指 **相对一次 iteration 的运行而言提前抓反传图**，不是昇腾那种离线出 om 的 AOT（概念相近：编译/抓图提前做）。

---

## 2. 在 `torch.compile` 里的位置

```text
用户代码
  → TorchDynamo          # 字节码 → 一段段 FX（Torch IR / Python 友好）
  → AOTAutograd          # 功能化 + 抓 joint 前向/反向 → 再 partition
  → Inductor（等）       # 把 fw/bw FX 降成核
  → autograd.Function    # fw 跑编译前向；bw 跑编译反向
  → （可选）CUDA Graph …
```

要点：

- Dynamo **graph break** → 多段前向；AOTAutograd **通常每段各做一套** fw/bw（反向也对应多段）。  
- 实验性 **compiled autograd** 才尝试把多段反向再收成更大图。  
- 推理路径可以只走前向编译（跳过反传相关步骤），但同一套 normalize / functionalize 思路仍可能用到。

---

## 3. 核心机制（按流水线）

### 3.1 机制总览

```text
① Functionalization     消掉 in-place / 别名，变成「纯函数」图
② FakeTensor 追踪       不跑真大数据，用 fake/meta 定 shape 规则
③ 构造 joint 函数        forward + autograd.grad 写在一个可 trace 的闭包里
④ __torch_dispatch__ 抓图  在 dispatcher 层录 Aten op → FX GraphModule
⑤ Partition             joint 切成 forward_graph / backward_graph（含 remat 决策）
⑥ 分别 compile          fw_compiler(fw_gm) / bw_compiler(bw_gm)
⑦ 包成 autograd.Function 与 eager autograd 引擎对接
```

下面逐步展开。

---

### 3.2 Functionalization（为何必须）

Autograd / 编译器都更喜欢 **无副作用** 的计算：

- `x.add_(y)`、`view` 共享 storage、input mutation 等，会让「保存给反向的是哪块内存」极难静态描述。  

AOTAutograd 用 **Functional Tensor / FunctionalTensorMode** 一类机制：

- 把 mutation 变成「算出新 tensor，当作额外输出」  
- 图结束有 **epilogue**：再 `copy_` 回原 input（若语义需要）  
- 这些「更新后的值」若参与求导，也会进入后面的 tangent / 反向输入集合  

效果：抓到的 FX 更接近 **纯数据流**，backend 好核、做 fusion 时少踩别名坑。  
（可用配置关闭 functionalization，但那是特殊路径，不是默认心智模型。）

---

### 3.3 FakeTensor / 元数据追踪（为何不必真跑完整数据）

抓 joint 图时不需要真实 batch 的数值，需要的是：

- 每个 Aten op 的 **输出 shape / stride / dtype**  
- 哪些输出 `requires_grad`、哪些要进反向  

用 **FakeTensor**（或等价 meta）：在 dispatcher 上「假执行」一遍，只维护元数据。  
这样 AOT 抓图成本主要在图结构，而不是大 matmul 真算。

#### 伪代码：FakeTensor 在干什么

真实 tensor 带 **storage（真数据）**；FakeTensor 只带 **元数据 +（可选）device 假象**，没有可用的大块数据。

```python
# --- 1. 假张量：长得像 Tensor，但不持有真实 storage ---
class FakeTensor:
    def __init__(self, shape, dtype, device, requires_grad, stride=None):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.requires_grad = requires_grad
        self.stride = stride or contiguous_strides(shape)
        # 注意：没有 self.storage = 真实 bytes

# --- 2. 元数据规则表：每个 Aten op 如何从输入 meta → 输出 meta ---
# 真实实现里大量来自 meta kernel / structured kernels
META_RULES = {
    "aten.mm": lambda A, B: FakeTensor(
        shape=(A.shape[0], B.shape[1]),
        dtype=promote(A.dtype, B.dtype),
        device=A.device,
        requires_grad=A.requires_grad or B.requires_grad,
    ),
    "aten.sin": lambda x: FakeTensor(
        shape=x.shape, dtype=x.dtype, device=x.device,
        requires_grad=x.requires_grad,
    ),
    "aten.sum": lambda x, dim=None: FakeTensor(
        shape=reduce_shape(x.shape, dim),
        dtype=x.dtype, device=x.device,
        requires_grad=x.requires_grad,
    ),
    # ...
}

# --- 3. 在 torch_dispatch 里拦截：不调 CUDA kernel，只跑 meta + 记图 ---
class FakeTensorMode:
    def __torch_dispatch__(self, func, types, args, kwargs):
        # args 里已是 FakeTensor（或常量）
        fake_args, fake_kwargs = tree_map_to_fake(args, kwargs)

        # 只算「输出长什么样」，不做 matmul 真计算
        out_meta = META_RULES[func](*fake_args, **fake_kwargs)

        # 同时把这一步记进 FX / 追踪日志（示意）
        graph.record(op=func, inputs=fake_args, output=out_meta)

        return out_meta  # 下游继续拿 FakeTensor 当输入

# --- 4. AOT 抓图时的用法 ---
def trace_with_fake(fn, example_inputs):
    # example_inputs 可以是真 tensor；先「剥成」同 shape 的 FakeTensor
    fake_inputs = [FakeTensor(t.shape, t.dtype, t.device, t.requires_grad)
                   for t in example_inputs]

    with FakeTensorMode():
        # 这里跑 fn / joint_fw_bw 时，所有 Aten op 都走上面的 dispatch
        # 路径上没有任何 4090 上的大 GEMM，只有 shape 算术 + 建图节点
        outs = fn(*fake_inputs)

    return graph.to_fx_graph()
```

#### 和真执行对比

```text
真执行:  mm(A[M,K], B[K,N]) → 算 M*N*K 次乘加 → 写出 [M,N] 数据
Fake  :  mm(A_meta, B_meta) → 输出 shape=(M,N) → 新建 FakeTensor，0 次乘加
```

Shape 不合法（如 `K` 对不上）时，meta 规则同样可以 **在抓图期报错**，不必等到真跑 CUDA。

#### 和 Proxy（FX symbolic_trace）的差别（直觉）

| | FakeTensor | FX Proxy |
|---|---|---|
| 拦截层 | 多在 **dispatcher / `__torch_dispatch__`** | Python 对象协议 / 模块 forward |
| 关注点 | Aten 语义下的 **meta**（shape 等） | 把调用记成 FX node |
| AOT 里 | Fake 负责「假跑出正确元数据」；dispatch 追踪负责「记成 Aten FX」 | 不是 AOT 主路径的抓图方式 |

二者都「不跑真数据」，但 FakeTensor 更贴近 **Aten 算子真实 shape 语义**（含 stride、view、部分 mutation 元数据），适合 joint autograd 展开。

### 3.4 Joint forward–backward：最关键的思想

直觉上构造这样一个函数再整段 trace：

```python
def joint_fw_bw(fw_inputs, grad_outs):
    fw_out = f(*fw_inputs)
    grad_inps = torch.autograd.grad(
        fw_out, fw_inputs, grad_outputs=grad_outs, ...
    )
    return fw_out, grad_inps
```

- 前半：原前向  
- 后半：用 **autograd.grad** 把反向公式也展开成一串可 dispatch 的 Aten op  

于是得到一张 **joint 图**：既含前向计算，也含反传计算，以及「前向里哪些中间值被反向用到」。

**重要：`joint_fw_bw` 的 `return` 是张量，不是 Graph。**  
完整图在 **外面打开的 tracer** 里，作为跑这一趟的副作用被攒出来。

#### 伪代码：图从哪来（外层记账，内层只跑）

```python
# ---------- 内层：普通计算，看不见 graph ----------
def f(x):
    return torch.sin(x)          # 真跑时走 Aten；trace 时被 dispatch 拦住

def joint_fw_bw(fw_inputs, grad_outs):
    fw_out = f(*fw_inputs)
    grad_inps = torch.autograd.grad(
        fw_out, fw_inputs, grad_outputs=grad_outs
    )
    return fw_out, grad_inps     # 返回值 = FakeTensor，不是图


# ---------- 外层：Tracer 在 dispatch 里攒节点 ----------
class GraphTracer:
    def __init__(self):
        self.nodes = []          # 这才是「正在长出来的 joint 图」

    def record(self, op, inputs, output):
        self.nodes.append({"op": op, "inputs": inputs, "output": output})

    def finalize(self):
        return FxGraph(self.nodes)   # 跑完后才「得到完整一张图」


class TracingFakeMode:
    def __init__(self, tracer: GraphTracer):
        self.tracer = tracer

    def __torch_dispatch__(self, func, types, args, kwargs):
        out = run_meta_kernel(func, args, kwargs)   # Fake：只推 shape
        self.tracer.record(func, args, out)         # 记账：图多一个 node
        return out


# ---------- AOT 抓 joint 图的用法 ----------
def capture_joint_graph(f, example_inputs):
    tracer = GraphTracer()
    fake_in = [to_fake(t) for t in example_inputs]
    fake_grads = [to_fake_like(y) for y in output_tangents_spec(f, fake_in)]

    with TracingFakeMode(tracer):          # 打开「飞行记录仪」
        # 这里像开飞机：只执行航线，不 return 地图
        _fw_out, _grad_inps = joint_fw_bw(fake_in, fake_grads)

    joint_graph = tracer.finalize()        # ← 完整一张图在这里
    return joint_graph


# 之后才能：
#   fw_g, bw_g = partition(joint_graph)
#   compiled_fw, compiled_bw = compile(fw_g), compile(bw_g)
```

对照：

```text
joint_fw_bw(...)     → 规定「飞哪条航线」（前向 + grad）
TracingFakeMode      → 每个 Aten op 往 tracer.nodes 里记一笔
tracer.finalize()    → 航线飞完，地图（joint FX）才交到你手里
```

注意：这里的 tracing **不是** `torch.fx.symbolic_trace` 那套 Proxy 故事；官方说明是 **`__torch_dispatch__` 机制** 抽图，再用 **FX GraphModule 当容器** 表示结果。

---

### 3.5 `__torch_dispatch__` 抓图（实现落点）

| 层次 | 做什么 |
|---|---|
| Dynamo | Python 字节码 → 较高层 FX |
| AOTAutograd | 在 **C++ dispatcher / torch_dispatch** 看到的 Aten 流上录图 |

好处：

- 自然经过 autograd、AMP、functorch、不少 subclass 行为（在 mode 打开时）  
- 落到 **Aten IR**（还可经 **decompositions** 拆成更细的 core/prim ops，方便 Inductor）  

可以把它想成：Dynamo 负责「Python 世界」；AOTAutograd 负责「Autograd + Aten 世界」。

---

### 3.6 Partition：从 joint 切回 fw / bw

Joint 图很大。训练运行时仍要：

- **Forward**：算 loss 需要的输出，并 **save** 一批给反向用的 activation  
- **Backward**：吃 `grad_out` + saved，算 `grad_in`

**Partitioner**（`partition_fn`）做两件事：

1. 切开 forward / backward 两张 FX  
2. 决定每个中间值是 **存下来** 还是 **反向时重算**（recomputation / rematerialization）

| 策略直觉 | 效果 |
|---|---|
| 默认 partition | 偏「该存就存」，实现直接 |
| `min_cut_rematerialization_partition` 等 | 按带宽/算力权衡，少存多算或相反 |

前向图往往会 **多输出** 一些 tensor：那些就是 saved for backward 的激活。  
反向图的输入里能看到这些 primal / saved，以及上游来的 `tangents`（grad_outputs）。

这正是 AOT 相对「只编译前向」的价值：**存算权衡可以在看见反向公式之后做。**

---

### 3.7 编译与 `autograd.Function` 胶水

```text
fw_gm → fw_compiler → compiled_fw
bw_gm → bw_compiler → compiled_bw

class CompiledFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        outs, saved = compiled_fw(...)
        ctx.save_for_backward(...)
        return outs

    @staticmethod
    def backward(ctx, *grads):
        return compiled_bw(grads, ctx.saved_tensors, ...)
```

（示意；真实实现更复杂，还要处理 mutation epilogue、alias、subclass 等。）

这样：

- **Autograd 引擎**仍按熟悉的 Function 节点调度  
- 每个节点内部跑的是 **已编译的整段 fw/bw 图**，而不是一长串小 `grad_fn`  
- Dynamo break 出的多段 → 多个 Compiled Function，中间穿插 eager Python，和「部分图编译」模型一致  

---

## 4. 和小例子对齐（sin）

假设 Dynamo 交出一段：`y = x.sin()`。

1. Functionalize（若无 mutation，几乎原样）  
2. Joint trace：前向 `sin`；反向需要 `cos(x) * dy` → joint 里出现 `cos` 等  
3. Partition：前向可能 **save `x`（或 cos 的输入）**；反向吃 `dy` 和 saved 算 `dx`  
4. Inductor 分别编 fw/bw  
5. 运行：`loss.backward()` 时走到该 Function 的 `backward` → 调 `compiled_bw`

若 partition 选择对 `cos` 重算而不是存结果，前向输出的 saved 集合会变——这就是 remat 的可见效果。

---

## 5. 和几件事的边界

| 概念 | 和 AOTAutograd |
|---|---|
| Eager Autograd | 动态挂 `grad_fn`；AOT 提前展开成静态 Aten FX |
| Dynamo | 前段：抓 Python；不管联合反传 |
| FX symbolic_trace | 另一套抓法；AOT 用 dispatch 抓，FX 仅作图容器 |
| LazyTensor | 也在 dispatcher 攒图，但偏每步执行/XLA；AOT 明确产 fw/bw 供编译 |
| `torch.utils.checkpoint` | 手写重算边界；AOT partition 可自动做类似权衡 |
| AOTInductor | 名字都有 AOT：偏 **导出/部署** 把编译结果落成可加载物；训练路径上的 AOTAutograd 是 **抓反传图** 那一层 |

---

## 6. 直接 API（理解用）

不必手写，但有助于对照源码心智：

```python
from torch._functorch.aot_autograd import aot_function

compiled = aot_function(
    fn,
    fw_compiler=my_fw_compiler,  # (gm, example_inputs) -> callable
    bw_compiler=my_bw_compiler,
    partition_fn=...,            # 默认或 min_cut_remat 等
    decompositions=...,          # 可选：复杂 op → 细粒度 op
)
```

`torch.compile` 内部会把 Dynamo 的每段图接到类似管道上，backend 默认指向 Inductor。

---

## 7. 一句话总结

**AOTAutograd = 在 dispatcher 上用 FakeTensor 把「前向 + autograd.grad」扫成一张 Aten FX joint 图 →（functionalize 清 mutation）→ partition 成可编译的 fw/bw（顺带决定存还是重算）→ 编完塞进 `autograd.Function`，让训练循环仍像普通 `backward()`。**

机制关键词：**Functionalization、FakeTensor、joint + autograd.grad、`__torch_dispatch__`、partitioner、autograd.Function 胶水。**

Dispatcher / Mode / TLS /「Mode 如何排到前面」见 [03-dispatcher-and-modes.md](./03-dispatcher-and-modes.md)。
