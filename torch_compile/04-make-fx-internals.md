# make_fx 内部机制：记账循环，以及值/图两侧的边界

> 承接 [03-dispatcher-and-modes.md](./03-dispatcher-and-modes.md)。
> 可运行实现：[`code/mini_make_fx.py`](./code/mini_make_fx.py)（约 400 行，只依赖 PyTorch）。
>
> 环境：`torch 2.14.0` / `Python 3.14.7` / macOS。源码行号以该版本为准。

---

```text
本次讲解位置
章节：torch_compile
小节：make_fx 的记账循环与两侧边界
知识点：proxy_call 四步 + memo 存 Proxy + Graph 不存边 + args 不收真值 + 模式进出栈
上次：Dispatcher / DispatchKey / TorchDispatchMode 的拦截条件与 TLS mode 栈
下次：把 FX 图翻译成自有 SSA IR（capture_fx 的 node_ids、DCE 后向扫描）
```

## 为什么现在讲这个

03 讲清了「Mode 打开后 Aten 算子会进 `__torch_dispatch__`」，但没有回答**进去以后做什么**。这里是整条编译链路唯一一处「值变成图」的地方：写错一步，后面所有 pass 都在处理一张错误的图，而且不会报错。

四个具体问题，前面都没有答案：

1. memo 里到底该存什么？存 `Node` 看起来更直接，实际会让 `make_fx` 在**最简单的单算子**上崩掉。
2. 图里要不要存边？存了以后图会有两份互相冲突的拓扑描述，且是**静默**的。
3. 节点能不能存真值？实测**建图不报错、`GraphModule` 也不报错**，直到运行时才炸——而炸之前数据已经悄悄退化成 shape + dtype。
4. 七个 context 谁先谁后？顺序决定 `__torch_dispatch__` 的优先级，也决定退出时状态能否还原。

这四个问题不是四个话题，它们是同一个设计的四个面：**记账循环只在「值侧」写一条记录，图的全部信息都压在这条记录的操作数里**。

---

## 1. 心智模型：两套东西同时存在

追踪时同一个进程里有两套数据：

| | 值世界 | 图世界 |
|---|---|---|
| 装的是 | 真实计算状态 | 计算过程的描述 |
| 典型对象 | `torch.Tensor`、`int`、`float`、`Proxy` | `Node`、`Graph`、`node.meta` |
| 能做什么 | 参与运算、读数据、看形状 | 遍历、改写、去重、删除 |
| 不能做什么 | — | 不能参与运算 |

实测（`_demo_two_worlds`）：

```text
  value side, torch.Tensor:
    tensor + 1  -> (2, 3)
    tensor.shape[0] -> 2
  graph side, Node:
    node + 1     -> TypeError: unsupported operand type(s) for +: 'Node' and 'int'
    node.shape   -> AttributeError: 'Node' object has no attribute 'shape'
    node[0]      -> TypeError: 'Node' object is not subscriptable
```

`Node` 上没有任何运算符重载，这不是缺功能，是故意的：一旦允许 `node + 1`，就分不清你是在**算**还是在**添加一条计算描述**。

`Proxy` 住在值侧（有 `__getitem__`，能当操作数），`.node` 指向图侧。它是唯一横跨两界的对象。

---

## 2. 完整实现

四个层，每层对应 PyTorch 的一个真实组件：

| 层 | 本文实现 | 对应 PyTorch |
|---|---|---|
| Layer 0 | `Node` / `Graph` | `torch.fx.Graph`、`Node` |
| Layer 1 | `Proxy`、`PythonKeyTracer` | `torch.fx.Proxy`、`set_proxy_slot`、`fetch_object_proxy` |
| Layer 2 | `ProxyTorchDispatchMode`、`proxy_call` | `ProxyTorchDispatchMode.__torch_dispatch__` |
| Layer 3 | `make_fx(fn, *inputs)` | `torch.fx.experimental.proxy_tensor.make_fx` |

完整代码见 [`code/mini_make_fx.py`](./code/mini_make_fx.py)。核心只有下面这些，其余是容器和打印：

```python
class Node:
    def __init__(self, kind, target, args, kwargs, name):
        self.kind = kind      # placeholder / call_function / output
        self.target = target  # aten.mm.default 之类
        self.args = args      # 操作数：Node 引用或真常量
        self.kwargs = kwargs
        self.name = name


class Graph:
    def __init__(self):
        self.nodes = []       # 只有这个字段，没有 edges


class Proxy:
    def __init__(self, node, tracer):
        self.node = node
        self.tracer = tracer

    def __getitem__(self, index):
        # 多输出算子被下标访问时就地补一个 getitem 行
        return self.tracer.create_proxy(
            "call_function", operator.getitem, (self, index), {}
        )


class PythonKeyTracer:
    def create_arg(self, value):
        if isinstance(value, Proxy):
            return value.node          # 图引用 -> 行
        if isinstance(value, Node):
            return value
        if isinstance(value, tuple) and not isinstance(value, torch.Size):
            return tuple(self.create_arg(item) for item in value)
        if isinstance(value, list):
            return [self.create_arg(item) for item in value]
        if isinstance(value, dict):
            return {key: self.create_arg(item) for key, item in value.items()}
        return value                    # 真常量原样放

    def bind(self, tensor, proxy):
        self._memo[id(tensor)] = (tensor, proxy)      # 存 Proxy

    def fetch(self, value):
        if not isinstance(value, torch.Tensor):
            return value
        entry = self._memo.get(id(value))
        if entry is None:
            raise RuntimeError("operand was not produced by a traced input")
        return entry[1]                                # 取 Proxy
```

---

## 3. 记账循环：`proxy_call` 四步

这是全部机制所在。

```python
def proxy_call(mode, func, args, kwargs):
    tracer = mode.tracer

    # 1. 真实张量 -> Proxy（同时校验每个操作数都可追踪）
    proxy_args = tree_map(tracer.fetch, args)
    proxy_kwargs = tree_map(tracer.fetch, kwargs)

    # 2. 先记账，再算
    out_proxy = tracer.create_proxy("call_function", func, proxy_args, proxy_kwargs)

    # 3. 让真实 kernel 跑一遍
    out = func(*args, **kwargs)

    # 4. 把真实输出绑回这一行
    track_tensor_tree(out, out_proxy, tracer)
    return out
```

对应真实源码（`torch/fx/experimental/proxy_tensor.py`）：

```text
_fetch_proxies_and_all_constant_flag   →  第 1 步
tracer.create_proxy(...)               →  第 2 步
out = func(*args, **kwargs)            →  第 3 步
track_tensor_tree(out, proxy_out, ...) →  第 4 步
```

### 3.1 为什么必须先记账再算

```python
out_proxy = tracer.create_proxy(...)   # 先
out = func(*args, **kwargs)            # 后
```

`func` 内部大概率会再 dispatch 别的算子。若顺序反过来，内层节点先落图，父节点的行号就排到了子节点后面——图变成「先换乘后上车」。

这与 Aotolith 里 `fuse_pointwise` 遇到的 use-before-def 是同一类约束：**行序必须满足数据依赖**。

### 3.2 为什么 `func(*args)` 不会递归

一直被讲错的一点。实测（`_demo_mode_is_popped`）：

```text
  mode stack before with-block: 0
  mode stack inside __torch_dispatch__: []
  mode stack after with-block: 0
```

**dispatcher 在调用 `__torch_dispatch__` 之前就把当前 mode 从栈上摘掉了**，所以在钩子里直接 `func(*args)` 不可能重新进入自己。

真 `make_fx` 额外包的 `disable_proxy_modes_tracing()` 关的是 C++ 层的 `TorchDispatchModeKey.PROXY` infra mode，不是 Python 这一层。迷你实现不需要它。

### 3.3 `track_tensor_tree` 顺带造出 `getitem`

```python
def track_tensor_tree(value, proxy, tracer):
    if isinstance(value, torch.Tensor):
        tracer.bind(value, proxy)
    elif isinstance(value, (tuple, list)) and not isinstance(value, torch.Size):
        for index, item in enumerate(value):
            track_tensor_tree(item, proxy[index], tracer)   # proxy[i] 建一行
```

`proxy[index]` 会立刻建一个 `getitem` 行。所以 FX 图里

```text
graph():
    %x_0 : placeholder
    %aten_max_dim = aten.max.dim((%x_0, -1))
    %getitem = getitem((%aten_max_dim, 0))
    %getitem_1 = getitem((%aten_max_dim, 1))
    return (%getitem, %getitem_1)
```

不是有人特判了多输出算子，而是「拿 Proxy 当值用、下标一下」的自然结果。

---

## 4. 边界推论一：memo 为什么绑 Proxy，不绑 Node

看着更直接的写法是「memo 存 Node，`fetch` 返回 Node」。在真 `make_fx` 上实测，**最简单的单算子就崩**：

```text
  ✗ mm:            NotImplementedError: argument of type: <class 'torch.fx.node.Node'>
  ✗ add+silu:      NotImplementedError: argument of type: <class 'torch.fx.node.Node'>
  ✗ 多输出 max:     NotImplementedError: argument of type: <class 'torch.fx.node.Node'>
  ✗ attention:     NotImplementedError: argument of type: <class 'torch.fx.node.Node'>
```

栈顶是建 `output` 节点那里：

```text
File "_symbolic_trace.py", line 932, in trace
  (self.create_arg(fn(*args)),),        ← 建 output 节点
File "proxy.py", line 483, in create_arg
  raise NotImplementedError(f"argument of type: {type(a)}")
```

原因在 `create_arg` 的入参白名单，`Proxy` 在其中，`Node` 不在：

```python
# torch/fx/proxy.py:436
if isinstance(a, Proxy):
    return a.node  # most common arg type goes first

# torch/fx/proxy.py:982
_create_arg_bypass[Proxy] = lambda self, a: a.node
```

### 4.1 消费端全都在取 `.proxy`

memo 里存的不是裸 Proxy，而是 `_ProxyTensor`：

```python
# torch/fx/experimental/proxy_tensor.py:1039
@dataclass
class _ProxyTensor:
    proxy: Proxy
    constant: Tensor | None
```

三处消费点都写 `lambda x: x.proxy`：

```text
proxy_tensor.py:1162   e.proxy if isinstance(e, _ProxyTensor) else e      # proxy_call 取参数
proxy_tensor.py:1719   get_proxy_slot(e, self, e, lambda x: x.proxy)     # unwrap_proxy
proxy_tensor.py:1950   get_proxy_slot(t, tracer, t, lambda x: x.proxy)   # wrap_key 收尾
```

`wrap_key` 那处最要命——它负责把**真实输出张量**换成句柄，然后交给 `create_arg` 建 `output` 节点：

```python
# proxy_tensor.py:1949
def get_tensor_proxy_slot(t: Tensor) -> Tensor | Proxy:
    return get_proxy_slot(t, tracer, t, lambda x: x.proxy)

out = pytree.tree_map_only(Tensor, get_tensor_proxy_slot, out)
```

链条：

```text
memo 存 Proxy
  └─ get_proxy_slot(...).proxy      取出来是 Proxy
       └─ tree_map_only 把 Tensor 换成它
            └─ create_arg 见到 Proxy → a.node     ✓
```

**输入侧和输出侧必须对称**：`track_tensor_tree` 塞进去的东西，`create_arg` 得能认出来。存 Node 就在这一步断掉。

### 4.2 为什么 `create_arg` 干脆不收 Node

`create_arg` 的职责是「把任意 Python 值翻译成图里能放的东西」，它需要一条无歧义的判据：

| 入参 | 判据 | 结果 |
|---|---|---|
| `Proxy` | `isinstance(a, Proxy)` | 图引用 → `a.node` |
| 未追踪的 `Tensor` | 落到 `get_attr` 分支 | 内联为常量 |
| `int` / `float` / `bool` / `None` / `dtype` | `_create_arg_bypass` | 原样放 |
| `Node` | 无匹配 | `NotImplementedError` |

`Node` 在 Python 里只是个普通对象，可以被当作元数据传进算子。一旦 `create_arg` 见 Node 就当引用，「这个 Node 恰好是个普通值」就无法表达了。`Proxy` 的唯一存在意义就是「我是图引用」，见到它 100% 确定意图——**用一个专门类型换一个无歧义判据**。

---

## 5. 边界推论二：Graph 为什么不存 edges

迷你 `Graph` 只有 `nodes`：

```text
  Graph attributes: ['nodes']
```

边全部能从 `args` 扫出来（`_demo_no_edges_stored`）：

```text
  derived edges (7):
    %x_0                  -> %aten_mm_default
    %x_1                  -> %aten_mm_default
    %aten_mm_default      -> %aten_add_Tensor
    %x_2                  -> %aten_add_Tensor
    %aten_add_Tensor      -> %aten_silu_default
    %aten_silu_default    -> %aten_mul_Tensor
    %aten_mm_default      -> %aten_mul_Tensor

  derived users (reverse edges):
    %aten_mm_default      used by ['%aten_add_Tensor', '%aten_mul_Tensor']
```

最后两条说明**扇出天然支持**——一个 Node 被多个 `args` 引用即可，不需要任何特殊处理。

### 5.1 存两份的后果是静默的

```text
  fabricated edge ('aten_mm_default', 'aten_silu_default') is absent from args
  -> with a stored edge list the graph would hold two disagreeing
     descriptions of its own topology, and nothing would report it
```

程序照跑，但 `edges` 说 A 依赖 B，`args` 说没这回事。这就是数据库里的更新异常：同一事实存两处，改一处忘一处就脏了。

### 5.2 所有 pass 改的都是 `args`

三种典型改写，全都只动 `args`：

| 改写 | 要做的操作 | 边 |
|---|---|---|
| 换算子 | 改 `node.target` | 完全不动（边只看 args） |
| 中间插节点 | `insert` 一行 + 改下游 `args` | 自动正确 |
| 删节点（DCE） | 从列表移除 + 改 `output.args` | 自动正确 |

如果存了 edges，插节点要手工删 `(mm→silu)`、加 `(mm→neg)`、加 `(neg→silu)`，漏一个就静默错。

### 5.3 什么时候才需要显式反向边

只有为了 **O(1) 查询**。`torch.fx` 的 `node.users` 是**索引**，不是事实来源：加边同步、删边同步，错了可以重建。迷你实现节点不到十个，扫一遍是微秒级，连这个缓存都省掉。

---

## 6. 边界推论三：args 里不能放真值

这一条最容易误判，因为**两道门都不拦你**。

```text
  fx.Graph.call_function took a Tensor operand without complaining
  Graph is a plain container; it does not type-check arguments
  GraphModule also built, because codegen does not validate either
  generated: mul_tensor = torch.ops.aten.mul.Tensor(x, torch.Tensor(size=[], dtype=float32))
  intent: [1.0, 2.0, 3.0] * 2.0 = [2.0, 4.0, 6.0]
  actual: NameError: name 'float32' is not defined
```

三点值得记住：

1. `Graph` 是哑容器，不检查 `args` 类型。
2. `GraphModule` 也能建出来，codegen 同样不校验。
3. 生成的源码里，`tensor(2.0)` 退化成了 `torch.Tensor(size=[], dtype=float32)`——**shape 和 dtype 留下了，数值丢了**。

即使那行代码能跑，结果也会是「乘一个未初始化的空张量」。**这正是描述与值的分界线**：图能携带 `shape`、`dtype`，携带不了 `2.0`。

### 6.1 那 `meta` 呢

`node.meta["val"]` 是官方通道，但装的是描述：

```text
  %mm: meta keys = ['tensor_meta', 'val']
    meta['val'] type  = FakeTensor
    meta['val'] shape = (2, 8)
    meta['val'] dtype = torch.float32
    declared bytes    = 64
    first byte        -> NotImplementedError: Not available for 'meta' device type
```

它知道该占多少字节（2×8×4 = 64），**一个字节都不持有**。`torch.compile` 的 min-cut partitioner 就靠它判断每个节点的输出多大，从而决定在哪切图。

### 6.2 权重到底存哪

实测（Aotolith 项目，需 MLX）：

```bash
python -c "...compile_module(M, (torch.randn(8,256),)).save('/tmp/demo.aoto')"
unzip -l /tmp/demo.aoto
```

```text
  Length      Name
  ----------  -----
     1187     manifest.json
      409     input_graph.json      ← 图，409 字节
      409     lowered_graph.json
     5160     kernels.json
      532     selections.json
      710     storage_plan.json
   524288     state/00000.bin       ← 256x512 f32 权重
```

**图 409 字节，数据 524288 字节，比值 1 : 1280。** demo 10 用手写结构复现了同一比例（380 字节 vs 524288 字节），不需要 MLX 就能跑。图里只有：

```json
"values": {
  "arg0": {"dtype": "float32", "shape": [8, 256]},
  "arg1": {"dtype": "float32", "shape": [256, 512]},
  "v0":   {"dtype": "float32", "shape": [8, 512]},
  "v1":   {"dtype": "float32", "shape": [8, 512]}
}
```

权重走「私有输入」通道，与结构平行，互不侵入：

```json
"state": [{
  "name": "w", "file": "state/00000.bin",
  "shape": [256, 512], "dtype": "float32",
  "nbytes": 524288, "digest": "56d94415..."
}]
```

`nbytes = 256 × 512 × 4`，一分不差。

### 6.3 Aotolith 的 IR 把这条路封死了

```python
@dataclass(frozen=True, slots=True)
class TensorType:
    shape: tuple[int, ...]
    dtype: DType

@dataclass(frozen=True, slots=True)
class Value:
    id: str
    type: TensorType          # 没有 data / tensor / buffer 字段
```

`Operation` 留了 `attrs: dict[str, object]`，但实际内容全是语义标量：

```text
pow                attrs={'scalar': 2, 'scalar_position': 'rhs'}
mean               attrs={'dims': (-1,), 'keepdim': True}
fused_elementwise  attrs={'body': (Operation(op='add', attrs={'alpha': 1.0, 'scalar': 1e-06, ...}), ...)}
matmul             attrs={}
```

### 6.4 唯一该把值写进图的情况

**常量折叠**，且折完是标量：

```text
折之前:  %v0 = mul(%x, 2.0);  %v1 = mul(%v0, 3.0)
折之后:  %v1 = mul(%x, 6.0)
```

写进去的是 `6.0`——`int/float` 图灵安全、与 shape 无关、可 JSON 序列化。真 `make_fx` 的对应机制是 `_ProxyTensor.constant`，它的用途是**决定要不要折叠**，不是把值放进图。

判据四问：

| 问题 | 答案 | 落到哪 |
|---|---|---|
| 数据会随调用变化吗？ | 会 | 图输入 / 运行时参数 |
| 所有 pass 都要读懂它吗？ | 不用 | `meta`（旁挂） |
| 它决定生成哪些代码吗？ | 决定 | `attrs`，标量形式 |
| 常量且已折叠完毕吗？ | 是 | `attrs`，标量形式 |

---

## 7. 模式的进出栈：七层 context

`torch/fx/experimental/proxy_tensor.py:3221`：

```python
with ExitStack() as stack:
    if self.fake_tensor_mode:
        stack.enter_context(self.fake_tensor_mode)     # 3223
    stack.enter_context(self.python_dispatcher_mode)    # 3224
    stack.enter_context(self.proxy_function_mode)       # 3225
    stack.enter_context(self.torch_fn_metadata_mode)    # 3226
    stack.enter_context(proxy_mode)                     # 3227  ProxyTorchDispatchMode
    stack.enter_context(disable_autocast_cache())       # 3228
    stack.enter_context(_set_make_fx_tracer(self))      # 3229
```

`ExitStack` 就是把嵌套 `with` 展开成数据驱动的栈：

```python
def enter_context(self, cm):
    result = cm.__enter__()          # ① 先进入
    self._push_cm_exit(cm, ...)      # ② 再把 __exit__ 压到列表尾部

def __exit__(self, *exc_details):
    while self._exit_callbacks:
        is_sync, cb = self._exit_callbacks.pop()   # ③ 从尾部取 → 逆序
```

所以 **`__enter__` 正序、`__exit__` 严格逆序**。实测：

```text
ENTER  FakeTensorMode
ENTER  <nullcontext>          ← python_dispatcher_mode（real/fake 下是空壳）
ENTER  <nullcontext>          ← proxy_function_mode（pre_dispatch 才真进）
ENTER  TorchFunctionMetadataMode
ENTER  ProxyTorchDispatchMode
ENTER  disable_autocast_cache
ENTER  _set_make_fx_tracer
   ... 真正的 trace 在这里 ...
EXIT   _set_make_fx_tracer
EXIT   disable_autocast_cache
EXIT   ProxyTorchDispatchMode
EXIT   TorchFunctionMetadataMode
EXIT   <nullcontext>
EXIT   <nullcontext>
EXIT   FakeTensorMode
```

两个细节：

- **`nullcontext` 也占位**。`python_dispatcher_mode` / `proxy_function_mode` 默认就是 `nullcontext()`，开关都压一个，`__enter__` 是空操作。位置固定，省掉一堆 `if`。
- **先 `__enter__` 再 push**。某个 `__enter__` 抛异常时它自己不会被登记，但**在它之前**已 push 的仍会被退出，不泄漏。

### 7.1 顺序为什么不能乱

dispatch mode 是栈，**后进的优先**。实测：

```python
with Named("A"):
    with Named("B"):
        torch.add(x, x)
```

```text
    [B] 先拿到 aten.add.Tensor
    [A] 先拿到 aten.add.Tensor
```

所以 `ProxyTorchDispatchMode`（3227）必须排在 `FakeTensorMode`（3223）**之后**进，才能占栈顶，才在建节点时先拿到 `func`。Aotolith 用的正是 `tracing_mode="fake"`，钩子触发瞬间栈内容是：

```text
fake 模式下的 dispatch mode 栈: ['FakeTensorMode', 'ProxyTorchDispatchMode']
```

退出必须逆序，否则某个 mode 的 `__exit__` 会在别人还挂着的时候还原全局状态。实测 trace 前后栈深度都是 0：

```text
trace 前 mode 栈深度: 0
trace 后 mode 栈深度: 0
```

---

## 8. 一次完整 trace 的数字

demo 11 在 `proxy_call` 外面裹了一层探针，同时打印两侧：

```text
  -- dispatch 1: aten.mm.default
     [value] operand shape=(2, 4)
     [value] operand shape=(4, 8)
     [value] result  shape=(2, 8)
     [graph] rows    %aten_mm_default
  -- dispatch 2: aten.add.Tensor
     [value] operand shape=(2, 8)
     [value] operand shape=(8,)
     [value] result  shape=(2, 8)
     [graph] rows    %aten_add_Tensor
  -- dispatch 3: aten.silu.default
     [value] operand shape=(2, 8)
     [value] result  shape=(2, 8)
     [graph] rows    %aten_silu_default
```

注意第 2 次 dispatch 有个 `shape=(8,)` 的操作数——那是 `%x_2`（偏置），**它也是被追踪的输入**，所以 `fetch` 能查到它。若把一个外部张量混进去，第 1 步就抛 `RuntimeError`。

**每一次 dispatch 同时推进两边。** 与「先有源码文件、再编译成 IR」不同，追踪里两边是同一份代码在同一进程里同时产生的。

追踪结束后只剩图：

```text
graph():
    %x_0 : placeholder
    %x_1 : placeholder
    %x_2 : placeholder
    %aten_mm_default = aten.mm.default((%x_0, %x_1))
    %aten_add_Tensor = aten.add.Tensor((%aten_mm_default, %x_2))
    %aten_silu_default = aten.silu.default((%aten_add_Tensor,))
    return %aten_silu_default
```

`args=(%x_0, %x_1)` 是**两个名字**，不是两个矩阵。

### 8.1 fake 与 real 的代价差

4096×4096 matmul，各 trace 一次：

```text
    real:    93.26 ms   (computes the matmul)
    fake:     0.30 ms   (records shapes only)
    speedup:  308.1x
```

绝对数字每次跑会漂，**量级差稳定在两个数量级以上**。这就是 `capture_fx` 选 `tracing_mode="fake"` 的原因。

### 8.2 和真 `make_fx` 对拍

```text
  mini : ['aten.pow.Tensor_Scalar', 'aten.mean.dim', 'aten.add.Tensor',
          'aten.rsqrt.default', 'aten.mul.Tensor', 'aten.mv.default']
  torch: ['aten.pow.Tensor_Scalar', 'aten.mean.dim', 'aten.add.Tensor',
          'aten.rsqrt.default', 'aten.mul.Tensor', 'aten.mv.default']

  9 rows each, identical operator sequence
```

---

## 9. 承接下游：FX 图 → SSA IR

`capture_fx` 拿到的已经是建好的图，**只需要翻译，不需要派生**，所以 Proxy 那一层整个不需要：

```python
node_ids: dict[object, str] = {}     # key 是 FX Node，value 是字符串 id

for node in traced.graph.nodes:
    if node.op == "placeholder":
        value_id = f"arg{len(input_ids)}"
        node_ids[node] = value_id
```

取输入是一次纯查表：

```python
def _argument_ids(argument, node_ids):
    try:
        return node_ids[argument]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unsupported FX argument: {argument!r}") from exc
```

三处对照：

| | 处于哪一侧 | 键 | 值 |
|---|---|---|---|
| `make_fx` 的 memo | 值侧 | `id(tensor)` | `_ProxyTensor(Proxy, constant)` |
| `symbolic_trace` 的 memo | 值侧 | `id(tensor)` | `Proxy` |
| `capture_fx` 的 `node_ids` | 图侧 | `Node` | `str` |

FX 图是拓扑有序且单赋值的，顺序翻译出来天然满足 SSA，不需要额外的构造算法。

`Graph.validate()` 只查一件事：**每个 `op.inputs` 都必须已被前面的 op 定义过**。这一条同时排除了悬空引用和前向引用，也让 `planning.py` 的活跃期分析直接扫 `op.inputs` 就够：

```python
for index, op in enumerate(graph.ops):
    for value_id in op.inputs:
        last_use[value_id] = index
```

---

## 10. 常见误区与症状

| 误区 | 症状 |
|---|---|
| memo 存 `Node` | `NotImplementedError: argument of type: <class 'torch.fx.node.Node'>`，在**建 output 节点**时崩，单算子必现 |
| 先算再记账 | 图能建出来，但行序违反数据依赖；下游 pass 报 use-before-def，或 CSE 误合并 |
| 在 `__torch_dispatch__` 里手动 pop mode | 冗余无害，但说明误以为会递归；真正需要关的是 C++ PROXY infra mode |
| `Graph` 同时存 nodes 和 edges | 无异常。内部矛盾静默存在，直到某次改写漏同步 |
| `args` 里放真 Tensor | 建图不报错、`GraphModule` 不报错；生成代码把它降级成 `torch.Tensor(size=[], dtype=...)`，运行时才炸 |
| 把权重内联进图 | 图从 KB 级涨到 GB 级，无法序列化、无法换权重、无法跨设备 |
| 以为 `meta["val"]` 存了数据 | 它是 FakeTensor：知道该占 64 字节，读第一个字节就 `NotImplementedError` |

---

## 11. 自测

**Q1.** 为什么 `out_proxy = tracer.create_proxy(...)` 必须在 `out = func(*args)` 之前？

<details><summary>答</summary>
因为 `func` 内部会继续 dispatch 别的算子，那些内层节点会先落图。顺序颠倒会让父节点的行号排在子节点之后，图不再满足数据依赖。约束与 `fuse_pointwise` 的 use-before-def 同源。
</details>

**Q2.** memo 里存 `Node` 而不是 `Proxy`，为什么单算子就会失败？失败点在哪一行？

<details><summary>答</summary>
失败在 `create_arg`。建 `output` 节点时 `wrap_key` 用 `get_proxy_slot(...).proxy` / `fetch` 取出句柄，再交给 `create_arg` 转成 Node 引用；而 `create_arg` 的白名单只有 `Proxy`（`torch/fx/proxy.py:436`、`:982`），遇到 `Node` 直接 `NotImplementedError`。输入侧和输出侧必须对称。
</details>

**Q3.** 为什么 `Graph` 不需要 `edges` 字段？如果非要存，会引入什么风险？

<details><summary>答</summary>
边完全由 `args`/`kwargs` 里的 Node 引用决定，扫描即可重建，反向边同理。存一份等于同一事实有两个来源：换 target、插节点、删节点时若漏同步，图会出现两份互相矛盾的拓扑描述，而且**没有任何检查会报错**。
</details>

**Q4.** 在 `args` 里放一个真 `torch.Tensor`，会在哪一步报错？

<details><summary>答</summary>
`fx.Graph.call_function` 不报错（哑容器），`fx.GraphModule(...)` 也不报错（codegen 不校验）。生成的源码会把该常量降级成 `torch.Tensor(size=[], dtype=float32)`——shape 和 dtype 保留、数值丢失，运行时才因为名字未定义或数据为空而失败。这恰好划出「描述」与「值」的边界。
</details>

**Q5.** `ExitStack` 里七个 `enter_context` 是并行进入的吗？退出顺序如何？为什么 `ProxyTorchDispatchMode` 要排在 `FakeTensorMode` 之后？

<details><summary>答</summary>
不是并行，是逐条正序执行 `__enter__`，退出时严格逆序（`_exit_callbacks` 用 `pop()` 从尾部取）。`ProxyTorchDispatchMode` 在后 → 位于 mode 栈顶 → `__torch_dispatch__` 里最先拿到 `func`，才能在建节点前拦截。fake 模式下实测栈内容为 `['FakeTensorMode', 'ProxyTorchDispatchMode']`。
</details>

---

## 12. 运行验证

```bash
cd torch_compile/code
python mini_make_fx.py
```

预期：十一个 demo 依次输出，最后是 `Next: a frontend walks ...`。关键断言在第 4 个 demo——与 `torch.fx.experimental.proxy_tensor.make_fx` 算子序列和行数完全一致，不一致会 `AssertionError`。

依赖只有 `torch>=2.7`；不需要 MLX，不需要 Aotolith，不需要 GPU。第 8、9 个 demo 用到 `torch.fx` 与 `FakeTensor`，都在 CPU 上完成。

涉及 Aotolith 的测量（第 6.2 节）需要该仓库的 `mlx` 环境与 Apple Metal，属于另一工程，本文只引用其结果。

---

## 13. 相关文档

- [01-dynamo-and-fx.md](./01-dynamo-and-fx.md) — Dynamo 抓图过程与 FX 的关系
- [02-aot-autograd.md](./02-aot-autograd.md) — joint 图、Proxy 四步追踪、partition
- [03-dispatcher-and-modes.md](./03-dispatcher-and-modes.md) — DispatchKey / Mode / TLS / 拦截边界
- [`code/mini_make_fx.py`](./code/mini_make_fx.py) — 本文完整可运行实现
- [PyTorch dispatcher walkthrough](https://github.com/pytorch/pytorch/wiki/PyTorch-dispatcher-walkthrough)
