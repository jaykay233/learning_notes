# 07 · 编译器工具链架构与工作流（Transform / Deploy）

> 原课：Transform 与 Deploy 工具链内幕。

---

## 0. 总故事

[02](./02-compiler-overview.md) 的前后端，在工程上拆成两条可交付链路：

```text
Transform（转换）：框架模型 → 可加载的硬件模型
Deploy（部署）：  加载 → 喂数 → 执行 → 业务后处理
```

昇腾：Transform≈**ATC**；Deploy≈**AscendCL**（或更高层引擎）。

---

## 1. Transform——为什么单独存在

### 1.1 问题

训练 checkpoint 不能直接在 NPU 上「打开即跑」：缺调度、缺内存规划、缺 kernel 绑定。

### 1.2 怎样工作

```text
ONNX/TF/…
  →（可选）量化校准
  → ATC：Parser + GE（准备/拆分/优化/编译）
  → *.om
  →（可选）AOE 调优再产出
```

### 1.3 效果与用在哪

效果：把「慢的、复杂的编译」留在上线前；设备上只加载。  
用在哪：Lab2/3；所有正式部署。

---

## 2. Deploy——为什么不能止于 om

### 2.1 问题

om 不会自动读摄像头、不会做 letterbox、不会做 NMS 或 tokenizer。

### 2.2 怎样工作

```text
应用进程
  AscendCL：Init → LoadModel → 分配 Device 输入输出
  → 拷入数据 → Execute → 取回结果
  → CPU 后处理 / 业务逻辑
```

### 2.3 效果与用在哪

效果：模型与应用解耦；同一 om 可嵌进不同服务。  
用在哪：Lab2 冒烟、Lab5/6 业务。

---

## 3. 工具地图（每个为什么存在）

| 工具 | 解决什么问题 | 典型使用者 |
|---|---|---|
| ATC | 离线编译出 om | 编译/部署工程师 |
| GE | ATC 内部图引擎 | 你主要通过日志感知 |
| AOE | 自动搜更好切分/算子策略 | 性能优化阶段 |
| AscendCL | 应用加载执行 | 业务开发 |
| msame | 快速验证 om | 调试 |
| 量化工具 | PTQ 配置 | 精度工程师 |

---

## 4. 一次完整闭环（效果检验）

```text
导出 ONNX → ATC → msame 冒烟 → 写入 AscendCL 服务 → Profiling →（可选 AOE）→ 再 ATC
```

细拆：[08](./08-frontend-parser.md)→[09](./09-backend-ge.md)→[10](./10-memory-schedule-aoe.md)

---

## 5. 练习

1. Transform 与 Deploy 的产物交接物是什么？  
2. 为何调试时常用 msame 而不是直接改大服务？  
3. AOE 属于 Transform 还是 Deploy？为什么？

### 参考答案

**1. Transform 与 Deploy 的产物交接物**

交接物是 **`*.om`（离线模型）**。

- Transform（ATC / 可选 AOE）：框架模型（如 ONNX）→ 编译、调度、绑 kernel → 产出 om  
- Deploy（AscendCL / msame / 业务服务）：**加载这份 om** → 分配 Device 缓冲 → 喂数 → Execute → 取回结果  

一句话：Transform 交付「能上板的模型文件」；Deploy 消费它，不再重做图编译（除非你再跑一轮 ATC）。

**2. 为何调试常用 msame，而不是直接改大服务**

msame 是 **最小 Deploy**：只验证「这份 om + 这份输入能否跑通、输出是否合理」。

大服务里还有预处理、后处理、多线程、业务逻辑、配置/鉴权等；出问题很难分清是：

- om / ATC 编错了，还是  
- letterbox、归一化、NMS、batch 拼装写错了  

先用 msame 冒烟，把变量收束到「模型本身」；确认 om 正确后再嵌进 AscendCL 服务，排错成本低一个数量级。Lab2 的路径也是这个习惯。

**3. AOE 属于 Transform，还是 Deploy？为什么？**

**属于 Transform（转换/调优侧），不是线上 Deploy 的主路径。**

AOE 做的是：自动搜索更好的切分、算子实现、融合等策略，**再产出（或改写）更优的 om**。这仍是「上线前把模型编译好」的工作。

Deploy 侧只应：**加载已调优的 om → 执行**。AOE 本身不负责读摄像头、做业务后处理；若把它塞进每次请求路径，会把「慢且重的搜索」拖到线上，违背 AOT「编译留在上线前」的分工。

（可选闭环：Profiling 发现问题 → AOE → 再 ATC → 新 om → 再 Deploy 验证。）
