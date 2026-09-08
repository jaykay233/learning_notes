# 09 · Backend：GE 优化与编译（原 TpuDialect / lowering）

> 原课：TpuDialect、TopToTpu、conversion、Pass。  
> 昇腾：**GE 拆分 / 优化 / 编译 → `.om`**。

---

## 1. 为什么需要 Backend

Frontend 给出「优化后的计算意图」，但仍缺：

- 这个 MatMul 用哪份 kernel  
- 中间 buffer 在哪段地址  
- 哪些 op 进 AI Core，哪些进 AI CPU  
- 最终如何打包成设备可加载物  

→ Backend = **硬件相关的 lowering + 资源编排 + 产出**。

---

## 2. 五步流水（怎样实现——概念）

```text
1. Parser           （偏前端，见 08）
2. 图准备           原图优化、Infershape
3. 图拆分           按引擎切子图
4. 图优化           融合、选实现、编译成 kernel 元数据；再整图优化
5. 图编译           内存、stream、task → 写 *.om
```

### 每步为什么存在

| 步 | 为什么 | 效果 |
|---|---|---|
| Infershape | 没 shape 无法分配/选 kernel | 静态规划成为可能 |
| 拆分 | 不同引擎能力不同 | 各优化各的 |
| 子图优化 | UB 融合、匹配算子库 | 吞吐↑ |
| 整图优化 | 跨子图再收一遍 | 全局更优 |
| 编译出 om | 运行时不能再做重编译（AOT） | 部署稳定 |

原课 **TopToTpu** ≈ 步骤 3–5 的「降到硬件方言」；**TpuDialect** ≈ 硬件侧 op 表达。

---

## 3. `.om` 里大概有什么（效果）

把它想成「已经排好工的施工包」：

- 权重（可能已按硬件 layout 重排）  
- 算子执行序列 / task  
- 内存规划结果  
- 输入输出描述  

**不是**训练用的 state_dict；也不是可读的 `.mlir` 教材文件。

---

## 4. 排错时如何用这张图

| 症状 | 更可能卡在 |
|---|---|
| unsupported / 解析失败 | Parser（08） |
| shape 推断失败 | 图准备 |
| 某 op 无实现 | 图优化/算子库 |
| 生成 om 后跑崩 | 运行时输入/内存，或编译假设被破坏 |

---

## 5. 练习

1. 默写五步，标出最像「算子融合」的一步。  
2. 为什么说 om 是 AOT 产物？  
3. TopToTpu 与 GE 图优化的类比一句话。

下一讲：[10](./10-memory-schedule-aoe.md)
