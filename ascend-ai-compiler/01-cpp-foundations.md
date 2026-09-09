# 01 · AI 编译器中的 C++（LLVM/MLIR 风格）

> 原课：「AI 编译器中的 C/C++」  
> 覆盖：SmallVector、StringRef / ArrayRef、BumpPtrAllocator、isa/cast/dyn_cast、CRTP、Visitor、TableGen。  
> 配套：[03](./03-mlir-dialect-op-type.md)、[04](./04-mlir-pass-patterns.md)、[Lab1](./labs/lab1-redundant-op-pass.md)。

本讲按统一四问展开每个知识点：**为什么要用 → 大致怎么实现 → 带来什么效果 → 一般用在哪**。最后用「写一个消冗余 Pass」把它们串起来。

---

## 0. 先建立一张总图（避免知识点散落）

编译器（含 MLIR Pass）里一次典型操作大概是：

```text
遍历 IR 上的 Operation
  → dyn_cast 成具体 Op 类型          ← RTTI
  → 取出 operands / 名字等
       SmallVector 暂存、ArrayRef/StringRef 只读传递  ← 容器与视图
  → 可能 new 一批分析节点             ← 常挂在 Bump 池上
  → 用 Pattern / Visitor 决定怎么改写
  → Pass 本身往往用 CRTP 挂进框架
  → Op 的类定义很多来自 TableGen
```

所以不是七个无关单词：它们是同一条流水线上的不同零件。  
**SmallVector / StringRef** 管「怎么廉价地拿着一堆小数据」；**Bump** 管「怎么廉价地造大量短命对象」；**isa/dyn_cast** 管「这是哪种 Op」；**CRTP/Visitor/TableGen** 管「框架怎么扩展、怎么生成样板」。

---

## 1. `SmallVector`：小数组优化的动态数组

### 1.1 为什么要用（问题从哪来）

`std::vector` 几乎总是：

1. 对象里只存 **3 个指针**（begin/end/capacity）  
2. 元素本身在 **堆** 上  

在编译器里，这种模式会很痛：

- 一个 Op 经常只有 **2～4 个 operand**，遍历百万个 Op 时，等于百万次「为了装两三个指针去 malloc」  
- 小分配多 → 分配器锁、缓存不友好、编译变慢  
- 很多中间结果是 **临时的**（收集一下 operands 就用掉），堆分配性价比极差  

经验分布：**绝大多数向量很小，偶尔才很大**。`SmallVector` 就是为这个分布定制的。

### 1.2 具体实现（概念模型）

```text
SmallVector<T, N> 内部大致两段：

  ┌─────────────────────────────┐
  │ 内联缓冲 inline[N]          │  ← 对象内部的数组，N 个 T
  │ size / capacity 等元数据    │
  └─────────────────────────────┘
           │
           │ 当 size 要超过 N
           ▼
      在堆上分配更大缓冲，把元素搬过去
      （之后行为近似 vector）
```

API 故意对齐 `vector`：`push_back`、`emplace_back`、`operator[]`、`begin/end`、`size`…

```cpp
llvm::SmallVector<mlir::Value, 4> operands;
operands.push_back(v0);
operands.push_back(v1);
// size=2 ≤ 4 → 数据还在 SmallVector 对象内部，无堆分配
```

要点：

| 设计点 | 含义 |
|---|---|
| 模板参数 `N` | 「我猜多数时候不超过 N」；常见 4、8 |
| 内联存储 | 小数据零堆分配 |
| 溢出到堆 | 大数组仍可用，不牺牲正确性 |
| 移动/拷贝 | 溢出后要管堆缓冲；细节以 LLVM 实现为准 |

同类：`SmallString`（小字符串）、部分 `TinyPtrVector`——同一哲学：**SSO / 小缓冲优化**。

### 1.3 效果是什么

| 对比 | `std::vector` | `SmallVector<T,4>`（size≤4） |
|---|---|---|
| 堆分配 | 几乎必有（除非 SSO 特例，标准 vector 没有） | **没有** |
| 局部性 | 元数据在栈、数据在堆 | 元数据+数据常在一起，更缓存友好 |
| 编译器场景 | 正确但慢 | 热路径上可明显减分配次数 |

注意：若你经常 `N=4` 却装 100 个元素，就退化成「带一点额外逻辑的 vector」，收益变小——`N` 要贴合真实分布。

### 1.4 一般哪里会使用

- 收集 Op 的 operands / results / 后继  
- Pattern 匹配时暂存「要替换的一组 Value」  
- 分析 Pass 里「当前 block 的若干 terminator」  
- 任何「热路径上、长度几乎总是很小」的动态数组  

和后面的关系：收集完常常 **转成 `ArrayRef`** 传给只读接口（见 §2），避免再拷一份。

---

## 2. `StringRef` / `ArrayRef`：不拥有内存的视图

### 2.1 为什么要用

编译器里字符串和数组传递极频繁：

- Op 名、symbol 名、attribute 名  
- 「请处理这一段 operands」  

若接口写成 `void foo(std::string)` / `void bar(std::vector<T>)`：

- 每次调用可能 **拷贝**  
- 调用方即便已有 `SmallVector`，也得再构造一个 `vector`  

C++17 的 `string_view` / `span` 解决同一问题；LLVM 更早就有 `StringRef` / `ArrayRef`。

核心契约：**我只是「看着」别人的内存，我不负责分配/释放。**

### 2.2 具体实现（概念模型）

```text
StringRef:   { const char *Data; size_t Length; }
ArrayRef<T>: { const T *Data; size_t Length; }
```

没有所有权 → 拷贝 `StringRef` 本身极便宜（两个机器字），**不拷贝字符内容**。

```cpp
void dumpName(llvm::StringRef name);           // 不拷贝字符串内容
void useOps(llvm::ArrayRef<mlir::Value> ops);  // 不拷贝 Value 数组

llvm::SmallVector<mlir::Value, 4> tmp = ...;
useOps(tmp);   // SmallVector → ArrayRef 隐式转换（只读视图）
```

`MutableArrayRef`：可写版，仍不拥有内存。

### 2.3 效果是什么

| | 拥有型（string/vector） | Ref/View |
|---|---|---|
| 传参成本 | 可能深拷贝 | 只拷指针+长度 |
| 接口统一 | 调用方类型要凑齐 | 可接受 vector/SmallVector/C 数组等多种来源 |
| 风险 | 较少悬垂 | **底层 buffer 先死 → 悬垂** |

### 2.4 一般哪里会使用 + 铁律

**适合：**

- 函数参数（最常见）  
- 临时比较：`if (name == "add")`  
- 从持久存储上「借」一段只读窗口  

**不适合（除非你保证生命周期）：**

- 把 `StringRef` 存进长期活着的成员变量，而底层是栈上 `std::string` 临时量  
- 从函数返回指向局部 buffer 的 `StringRef`

```cpp
// 危险示意
StringRef bad() {
  std::string s = compute();
  return s;  // s 销毁后 Ref 悬垂（即便能编译也逻辑错）
}
```

需要拥有时：拷进 `std::string`，或拷进 **Bump 分配的内存**（§3），再对外发 `StringRef`。

**和 SmallVector 的连接：**  
热路径用 `SmallVector` 攒数据 → 用 `ArrayRef` 借出去给只读 API → 两者搭配是 LLVM 代码里的「默认姿势」。

---

## 3. `BumpPtrAllocator`：竞技场式内存池

### 3.1 为什么要用

编译一次可能创建 **海量** 短命对象：

- IR 节点、类型、属性  
- 分析结果、worklist 节点  

若每个都 `new` + 最后逐个 `delete`：

- 分配次数爆炸  
- 释放也爆炸  
- 内存碎片  

但这些对象往往有共同命运：**同一次编译 / 同一个 Context 结束就全部不要了**。

→ 用「整块申请、指针撞钟分配、整块扔掉」的 **arena / bump allocator**。

### 3.2 具体实现（概念模型）

```text
向 OS/堆要一大 slab（例如几 KB～几 MB）
cur 指向 slab 起点

Allocate(size, align):
  把 cur 按对齐垫一下
  p = cur
  cur += size
  若超出当前 slab → 再要新 slab，挂到链表上
  return p

销毁 Allocator:
  把所有 slab 一次性 free
  （通常不再逐对象调用析构，或只允许 trivially destructible）
```

```cpp
llvm::BumpPtrAllocator alloc;
auto *node = alloc.Allocate<MyNode>();  // 概念：在池上放置
// … 编译结束
// alloc 析构 → 整池释放
```

（实际 API 名/放置 new 写法以 LLVM 头文件为准；这里抓模型。）

### 3.3 效果是什么

| | `new/delete` | Bump |
|---|---|---|
| 单次分配 | 走堆，较慢 | 多半是指针加法 |
| 释放 | 逐对象 | **O(slab 数)** 整片丢弃 |
| 适用 | 寿命各异、要精确析构 | 同生共死的海量小对象 |

代价/约束：

- 对象在池中间 **很难单独回收**（除非另做 lifetime 设计）  
- 放入带 `std::vector` 成员的复杂 C++ 对象要非常小心（析构不会自动跑时会漏）  
- 更适合 POD / 自己管资源、或明确「永不单独析构」的 IR 节点风格  

### 3.4 一般哪里会使用

- `MLIRContext` / 类型与属性的 uniquing 存储背后  
- 一次 Pass 的分析图（点、边）  
- 编译会话级的临时 AST/IR  

**和前面的连接：**  
Bump 上可以 bump 出一长串 `char`，再对外发 `StringRef`（视图指向池，池比 Ref 活得久）——这是「既便宜又安全」的拥有方式之一。

---

## 4. LLVM RTTI：`isa` / `cast` / `dyn_cast`

### 4.1 为什么要用

IR 上你拿到的常常是基类：`Operation *`、`Type`、`Attribute`。  
真正逻辑在具体子类：`AddOp`、`FuncOp`、`IntegerType`…

标准 C++ `dynamic_cast`：

- 依赖编译器 RTTI，开销与可控性对 LLVM 不理想  
- 和 LLVM 自己的类层次、TableGen 生成代码集成差  

于是 LLVM 用一套 **手写/生成的 classof + 三个工具函数**。

### 4.2 具体实现（概念模型）

每种具体类型提供：

```cpp
static bool classof(const Operation *op) {
  return op->getName().getStringRef() == "my.add";  // 示意
}
```

工具：

```cpp
template <typename T>
bool isa(const From &val) { return T::classof(/*...*/); }

template <typename T>
T dyn_cast(const From &val) {
  return isa<T>(val) ? T(/*...*/) : nullptr/空;
}

template <typename T>
T cast(const From &val) {
  assert(isa<T>(val));
  return T(/*...*/);
}
```

使用：

```cpp
if (auto add = dyn_cast<AddOp>(op)) {
  Value lhs = add.getLhs();
  // ...
}

if (isa<FuncOp>(op)) { /* 只判断 */ }

AddOp a = cast<AddOp>(op);  // 你 100% 确定时；错了就 assert 挂
```

### 4.3 效果是什么

| API | 语义 | 失败时 |
|---|---|---|
| `isa<T>` | 是不是 T | `false` |
| `dyn_cast<T>` | 尝试变成 T | `nullptr` / 空 |
| `cast<T>` | 断言就是 T | **崩**（调试期抓逻辑错误） |

效果：在巨型 Op 层次里 **又快又清晰** 的分派，且与 TableGen 生成的 `classof` 一致。

### 4.4 一般哪里会使用

- 几乎所有 Pass / Pattern 的第一行  
- `TypeSwitch<Operation *>(op).Case<AddOp>(...).Case<MulOp>(...)`  
- Verifier、fold、打印器  

口诀：**不确定用 `dyn_cast`；确定到可以赌命才用 `cast`。**

**和 Visitor 的连接：** Visitor / TypeSwitch 内部本质上还是一串「按类型分发」，底层仍依赖这套 RTTI。

---

## 5. CRTP：编译期多态

### 5.1 为什么要用

Pass 框架想表达：

```text
通用流程：setup → runOnOperation → teardown
差异点：每个 Pass 的 runOnOperation 不同
```

虚函数可以，但：

- 热路径多一层间接  
- 有时还要把类型信息留到编译期做优化  

CRTP（Curiously Recurring Template Pattern）让 **基类模板参数就是派生类自己**，从而在编译期调用到派生实现。

### 5.2 具体实现

```cpp
template <typename Derived>
class PassBase {
public:
  void run() {
    // 编译期就知道 Derived，可内联
    static_cast<Derived *>(this)->runOnOperation();
  }
};

class EliminateIdentityPass : public PassBase<EliminateIdentityPass> {
public:
  void runOnOperation() {
    // 真正逻辑
  }
};
```

派生类继承 `PassBase<自己>`——看起来怪，所以叫「奇异递归」。

MLIR 里 `PassWrapper<ConcretePass, Pass>` 一类就是这种味道：你写 Concrete，基类帮你挂到 Pass 机制上。

### 5.3 效果是什么

| | 虚函数 | CRTP |
|---|---|---|
| 绑定时机 | 运行时 vtable | **编译期** |
| 内联 | 难 | 容易 |
| 灵活性 | 可随意指针多态 | 类型在模板里钉死 |
| 代价 | 间接跳转 | 代码膨胀（每个 Derived 一份） |

### 5.4 一般哪里会使用

- Pass / Analysis 的样板基类  
- 一些 iterator / trait 工具  
- 任何「框架定流程、用户填钩子、又想要速度」的地方  

**和 TableGen 的连接：** TableGen 生成 Pass 登记代码；你的 Pass 类常再套一层 CRTP wrapper——生成 + 静态多态一起减样板。

---

## 6. Visitor：把「遍历」和「处理」拆开

### 6.1 为什么要用

Op 种类极多，朴素写法：

```cpp
if (isa<A>(op)) ...
else if (isa<B>(op)) ...
else if (isa<C>(op)) ...
// 爆炸，且遍历逻辑和处理逻辑缠在一起
```

Visitor 经典意图：

- **遍历结构**的人只负责走到每个节点  
- **处理节点**的人按类型 overload `visit`  

两者可独立扩展（在经典 GoF 里通过 `accept`/`visit` 双分派；MLIR 里常简化成 walk + 分发）。

---

### 6.2 完整例子：迷你表达式 IR + 两种 Visitor

下面是一份 **可单独理解的完整 C++ 例子**（教学向，不依赖 MLIR）。  
场景：表达式树只有三种节点——常量、加法、取负。我们要做两件完全不同的事：

1. **打印**成字符串  
2. **求值**成 `int`  

若不做 Visitor，每种新操作都要改所有节点类（加虚函数）。Visitor 把「新操作」收成一个新 Visitor 类。

#### （1）节点与 Visitor 接口

```cpp
#include <iostream>
#include <memory>
#include <string>

// ---- 前置声明 ----
struct ConstExpr;
struct AddExpr;
struct NegExpr;

// Visitor：每种具体节点一个 visit 重载
struct ExprVisitor {
  virtual ~ExprVisitor() = default;
  virtual void visit(ConstExpr &e) = 0;
  virtual void visit(AddExpr &e) = 0;
  virtual void visit(NegExpr &e) = 0;
};

// 表达式基类：只负责 accept（把「我是谁」交给 visitor）
struct Expr {
  virtual ~Expr() = default;
  virtual void accept(ExprVisitor &v) = 0;
};

struct ConstExpr : Expr {
  int value;
  explicit ConstExpr(int v) : value(v) {}
  void accept(ExprVisitor &v) override { v.visit(*this); }  // 双分派第 2 下
};

struct AddExpr : Expr {
  std::unique_ptr<Expr> lhs, rhs;
  AddExpr(std::unique_ptr<Expr> l, std::unique_ptr<Expr> r)
      : lhs(std::move(l)), rhs(std::move(r)) {}
  void accept(ExprVisitor &v) override { v.visit(*this); }
};

struct NegExpr : Expr {
  std::unique_ptr<Expr> inner;
  explicit NegExpr(std::unique_ptr<Expr> e) : inner(std::move(e)) {}
  void accept(ExprVisitor &v) override { v.visit(*this); }
};
```

**双分派在干什么（读代码时跟一眼）：**

```text
expr->accept(visitor)
  → 虚调用进 ConstExpr::accept / AddExpr::accept / …
  → 里面写 v.visit(*this)
  → 此时 *this 已是静态类型 ConstExpr&，于是命中 visit(ConstExpr&)
```

第一下靠「节点虚表」选中正确的 `accept`；第二下靠「visit 重载」选中正确的处理函数。

#### （2）PrintVisitor：只负责「怎么打印」

```cpp
struct PrintVisitor : ExprVisitor {
  std::string out;

  void visit(ConstExpr &e) override {
    out += std::to_string(e.value);
  }

  void visit(AddExpr &e) override {
    out += "(";
    e.lhs->accept(*this);   // 递归：遍历交给 accept，处理仍是本 visitor
    out += " + ";
    e.rhs->accept(*this);
    out += ")";
  }

  void visit(NegExpr &e) override {
    out += "-(";
    e.inner->accept(*this);
    out += ")";
  }
};
```

#### （3）EvalVisitor：只负责「怎么求值」

```cpp
struct EvalVisitor : ExprVisitor {
  int result = 0;

  void visit(ConstExpr &e) override {
    result = e.value;
  }

  void visit(AddExpr &e) override {
    e.lhs->accept(*this);
    int a = result;
    e.rhs->accept(*this);
    int b = result;
    result = a + b;
  }

  void visit(NegExpr &e) override {
    e.inner->accept(*this);
    result = -result;
  }
};
```

#### （4）拼一棵树跑起来

```cpp
int main() {
  // 树： -(1 + 2)    即 -3
  auto expr = std::make_unique<NegExpr>(
      std::make_unique<AddExpr>(
          std::make_unique<ConstExpr>(1),
          std::make_unique<ConstExpr>(2)));

  PrintVisitor printer;
  expr->accept(printer);
  std::cout << printer.out << "\n";   // 期望：-(1 + 2)

  EvalVisitor eval;
  expr->accept(eval);
  std::cout << eval.result << "\n";   // 期望：-3
}
```

#### （5）这个例子说明了什么

| 角色 | 谁扮演 | 改它当什么变了 |
|---|---|---|
| 结构 / 遍历 | `Expr` 树 + `accept` 递归 | 新增节点类型时要改 Visitor 接口（经典 Visitor 的代价） |
| 操作 A | `PrintVisitor` | **只加一个类**，不用改 Const/Add/Neg |
| 操作 B | `EvalVisitor` | 同上 |

对照编译器：`Const/Add/Neg` ≈ 各种 `Op`；`PrintVisitor` ≈ dump；`EvalVisitor` ≈ 常量折叠/解释执行。

---

### 6.3 同一问题的 MLIR 风格写法（walk + 分发）

真实 MLIR 很少手写整套 GoF `accept`，因为 `Operation` 已经能 **walk**，类型分发用 `dyn_cast` / `TypeSwitch`。

> 排版说明：Markdown 会把「方括号 + 圆括号」当成链接，所以不要把 lambda 直接粘在 `walk(` 后面写成一行；下面改成先定义 `visitOp`，再 `root->walk(visitOp)`。
>
> C++ 里等价写法仍是：`root->walk( 带捕获的 lambda )`。

```cpp
// 教学示意：统计模块里 AddOp 个数，并打印 ConstOp 的值
// （类型名按你的 Dialect 替换；逻辑完整）

struct CountAndDump {
  int addCount = 0;

  void run(Operation *root) {
    // 捕获列表与参数分行写：避免 Markdown 把 ]( 当成链接；C++ 完全合法
    auto visitOp = [&]
    (Operation *op) {
      // —— 遍历：walk 已经帮你走到每个 op ——
      // —— 处理：按类型分发（Visitor 的 visit 重载）——
      if (auto add = dyn_cast<AddOp>(op)) {
        (void)add;
        ++addCount;
        return;  // 只结束本次 lambda，不结束 run()；walk 继续下一个 op
      }
      if (auto c = dyn_cast<ConstOp>(op)) {
        llvm::errs() << "const=" << c.getValue() << "\n";
        return;
      }
      // 其它类型：默认忽略
    };
    root->walk(visitOp);
  }
};
```

仍可能被部分预览器误伤的两行，用文字描述捕获即可：lambda 捕获列表为「仅 `&`」，参数为 `Operation *op`。

```cpp
// 等价的 TypeSwitch 写法（更像「一组 visit 重载」）：
void dumpOne(Operation *op) {
  llvm::TypeSwitch<Operation *>(op)
      .Case<AddOp>([](AddOp add) {
        llvm::errs() << "saw add\n";
      })
      .Case<ConstOp>([](ConstOp c) {
        llvm::errs() << "const=" << c.getValue() << "\n";
      })
      .Default([](Operation *) {});
}
```

调用关系：

```text
定义 visitOp = lambda(捕获 &, 参数 Operation*)
然后 root->walk(visitOp)
```

和 GoF 例子的对应：

| GoF | MLIR 日常 |
|---|---|
| `expr->accept(v)` 递归遍历 | `root->walk(visitOp)` |
| `v.visit(AddExpr&)` | `dyn_cast<AddOp>` / `.Case<AddOp>` |
| 新加一种「操作」= 新 Visitor 类 | 新加一个 Pass / 一个 lambda 管道 |
| 节点上写 `accept` | 一般 **不用** 你手写；框架已有 |

**（3）PatternRewrite**  
「匹配到某种局部形状 → 改写」——可以看成 **声明式的、可组合的局部 Visitor**（见 [04](./04-mlir-pass-patterns.md)）。Lab1 的消除 identity 就是这种。

---

### 6.4 效果是什么

- 结构清晰：先想「怎么走完图」，再想「碰到 Add 干什么」  
- 新加一种 **分析/打印/变换** 时，处理逻辑有固定挂载点  
- 和 Greedy Pattern 结合后，优化可插拔  

### 6.5 一般哪里会使用

- 整模块统计、校验  
- Canonicalize / 自定义消除 Pass（Lab1）  
- 调试打印「所有 ConvOp」  

**和 RTTI 的连接：** Visitor 的每个分支仍然靠虚表（GoF）或 `isa`/`dyn_cast`/`Case<T>`（MLIR）。

---

## 7. TableGen：用声明生成 C++ 样板

### 7.1 为什么要用

一个 Dialect 可能有几十上百个 Op。若手写每个 Op 的：

- C++ 类  
- `getLhs()` 访问器  
- 解析 / 打印  
- `classof`（给 isa 用）  
- traits、文档字符串…  

会重复到无法维护，且易与 `.mlir` 文本格式不一致。

TableGen：用 **领域特定的声明**（`.td`）描述 Op，**生成**上述样板。

### 7.2 具体实现（工作流）

```text
你写 my_ops.td
        │  TableGen + MLIR 的 Op 生成后端
        ▼
  MyOps.h.inc / MyOps.cpp.inc 等
        │  #include 进你的 Dialect
        ▼
  得到 AddOp 类、parse/print、classof…
你只手写：verify / fold / canonicalize pattern
```

`.td` 示意：

```td
def MyAddOp : Op<"my.add", [Pure]> {
  let summary = "integer add";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
}
```

### 7.3 效果是什么

| 手写全部 | TableGen |
|---|---|
| 易漏 `classof` → isa 错乱 | 生成与声明一致 |
| 改 operand 名要改很多处 | 改 `.td` 再生成 |
| 新人难入门 | 声明即文档 |

代价：要学 `.td` 语法与生成管线；调试时要会看「这是生成代码」。

### 7.4 一般哪里会使用

- 定义 Dialect / Op / Pass 选项  
- LLVM 里大量后端、Option、Intrinsics 也是 TableGen  

昇腾课日常 **ATC 转 om 不必写 TableGen**；但 Lab1 若自定义 Op、或读开源 dialect，就会碰到。

**和 RTTI / CRTP 的连接：**  
TableGen 生成 `classof` → `isa/dyn_cast` 才认识你的 Op；Pass 登记也可能生成，再配合 CRTP wrapper。

---

## 8. 串起来：写 Lab1「消冗余」时它们如何同时出现

目标：删掉 `y = identity(x)`，用 `x` 替换所有对 `y` 的使用。

```text
1. TableGen（可选）定义 IdentityOp，生成 classof / accessors
2. 你的 Pass 类用 CRTP/PassWrapper 挂进 PassManager
3. run 里 walk 或 GreedyPattern：
     auto id = dyn_cast<IdentityOp>(op);   // RTTI
     if (!id) return failure();
     Value in = id.getInput();
     // 可能用 SmallVector 收集 users（很多时）
     rewriter.replaceAllUsesWith(id.getResult(), in);
     rewriter.eraseOp(id);
4. 函数边界上用 ArrayRef/StringRef 传只读信息
5. 分析辅助节点若很多，可放进 Bump 池（简单 Lab 未必需要）
```

读代码时也可以反过来：**看见 SmallVector → 想「这里在攒小列表」；看见 dyn_cast → 想「这里在分类型」**。

---

## 9. 和昇腾课的关系（再钉一次）

| 场景 | 这些知识 |
|---|---|
| 只会 ATC / 量化 / 上板 | 可先浏览；面试常问概念 |
| Lab1、读 MLIR 源码 | **必须啃透本讲** |
| 以后写自定义算子编译器 | TableGen + Pass + 容器/池日常用 |

建议顺序：本讲 → [02](./02-compiler-overview.md) → [03](./03-mlir-dialect-op-type.md) → [04](./04-mlir-pass-patterns.md) → [Lab1](./labs/lab1-redundant-op-pass.md)。

---

## 10. 练习（逼自己连起来）

1. 用自己的话画 `SmallVector<T,4>` 在 size=3 与 size=10 时内存在哪。  
2. 为什么 `StringRef` 作函数**返回值**容易翻车？如何改成安全？  
3. Bump 分配的对象为什么常常不调用析构？这限制你往池里放什么？  
4. 给出 `cast` 用错会怎样、`dyn_cast` 用错会怎样。  
5. 用一张流程图描述：CRTP Pass + walk + dyn_cast + SmallVector 在一次消除 Pass 里的调用顺序。  
6. TableGen 若漏生成 `classof`，你会在运行期看到什么症状？

---

## 11. 大纲覆盖

| 原课条目 | 章节 |
|---|---|
| SmallVector | §1 |
| StringRef / ArrayRef | §2 |
| BumpPtrAllocator | §3 |
| isa / cast / dyn_cast | §4 |
| CRTP | §5 |
| Visitor | §6 |
| TableGen | §7 |
| 综合串联 | §0、§8 |
