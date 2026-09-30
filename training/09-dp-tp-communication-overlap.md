# DP 与 TP 的通信—计算重叠有什么区别

## 本次讲解位置

- 章节：`training` / Megatron-LM 分布式训练与通信优化
- 小节：Megatron-LM 02《张量并行》Part 4，4.3「TP 通信与计算重叠」
- 知识点：DP 梯度通信为什么通常能和反向传播重叠，以及它与 TP 层内重叠的区别
- 上次：TP + SP 前向中，All-Gather / Reduce-Scatter 如何转换序列布局
- 下次：结合具体配置观察通信耗时、bucket 大小及 overlap 的实际收益

## 为什么现在讲这个

“DP 是层间重叠，TP 是层内重叠”容易被记成死规则。需要弄清重叠双方分别是什么：DP 在等待某些参数梯度跨数据并行副本归约时，autograd 还可以继续计算其他层的梯度；TP 则尝试让同一层里的 GEMM 分块计算与该层所需的 All-Gather / Reduce-Scatter 并行推进。它们利用的是不同的依赖关系，因此调度窗口也不同。

## 心智模型

把一次反向传播想成从模型顶端往底端倒着走：

```text
反向计算方向：L3 → L2 → L1
梯度通信：    grad(L3) ────────────────► DP ranks
                       grad(L2) ───────► DP ranks
                                  grad(L1) ─► DP ranks
```

某层的梯度一旦算好，就可以开始传；同时 GPU 继续计算更靠前层的反向。通信和计算不必等整个 backward 结束后才依次进行。

## DP：对不同数据副本的同一参数梯度做同步

假设有 4 个 DP ranks。每个 rank 持有相同模型参数，但处理不同样本：

```text
DP rank 0: batch 0 → 本地参数梯度 g0
DP rank 1: batch 1 → 本地参数梯度 g1
DP rank 2: batch 2 → 本地参数梯度 g2
DP rank 3: batch 3 → 本地参数梯度 g3
```

对同一参数，各 rank 必须聚合梯度（通常做 sum 或按约定取平均），才能保持副本的参数更新一致。普通 DDP 常用 All-Reduce；采用分片优化器时也可能使用 Reduce-Scatter 等通信。

反向传播从后往前产生各层梯度。框架可以把梯度放进 buckets：某个 bucket 所含梯度都 ready 后，立即异步发起 DP collective，而不是等所有层都反向完成。

### 三层的时间线示例

令前向为 `L1 → L2 → L3`，反向为 `L3 → L2 → L1`：

```text
时间向右：
计算流：  backward(L3)   backward(L2)   backward(L1)   optimizer
通信流：  [同步 L3 梯度] [同步 L2 梯度] [同步 L1 梯度] [必要时等待]
          └─与 L2 计算重叠┘ └─与 L1 计算重叠┘
```

拆成事件看：

1. `L3` 反向得到梯度后，相关 DP bucket ready，发起异步 All-Reduce。
2. All-Reduce 在通信通道上推进时，GPU 继续算 `L2` 的反向。
3. `L2` 的梯度 ready 后，再发起它所在 bucket 的归约；通信可与 `L1` 反向重叠。
4. optimizer 更新前，所需梯度通信必须全部完成；若有通信没赶上计算，就在同步点等待。

这就是常说的“DP 梯度通信与反向传播重叠”。所谓“层间”，是典型情形下**一层已 ready 的梯度通信**与**其他尚未完成层的反向计算**并发，不表示只能严格一层对一层。实际 bucket 可能包含多个层的参数；一个 bucket 的通信也可能与同一层剩余计算或多个后续反向算子重叠。

## TP：同一层内的通信和矩阵乘分块重叠

TP 把一个层的矩阵/激活按张量维度分给多个 ranks。某些 TP 路径在该层内需要 All-Gather 或 Reduce-Scatter；若实现有通信—计算 overlap，可把输入、输出拆成 chunks，让通信和 GEMM 流水执行：

```text
chunk 0 的数据到达 → GEMM chunk 0 开始
                       同时传输 chunk 1
chunk 1 到达       → GEMM chunk 1 开始
                       同时传输 chunk 2
```

例如 All-Gather 的数据分块到齐后，相应 GEMM chunk 可以先计算，不必总等整个张量传完；或局部 GEMM 产生一块结果后，就开始该块对应的归约/分发。依赖顺序仍必须满足：消费者不能使用尚未到达的数据，归约也不能早于对应的 partial 结果。

TP overlap 不是“TP 通信天然和 GEMM 同时发生”。不做分块、异步调度或专门 overlap 实现时，常见执行仍是整段通信结束后再做 GEMM，或 GEMM 结束后再通信。

## 一张表对比

| 对比项 | DP overlap | TP overlap |
|---|---|---|
| 通信内容 | 同一参数在不同数据副本上的梯度聚合 | 张量分片间的数据交换/归约，如 All-Gather、Reduce-Scatter |
| 典型通信对象 | 梯度 buckets | 激活、矩阵乘输入/输出分片或其 partial |
| 常见重叠双方 | 已 ready 的梯度通信 + 其余层的 backward 计算 | 同一层的通信 chunks + GEMM chunks |
| 可利用的依赖窗口 | 反向按层逐步产出梯度 | 一个层的矩阵乘/通信可按块流水 |
| 结束前的约束 | optimizer 更新前梯度同步完成 | 每个计算块使用数据前，该块的通信完成；依赖的归约结果就绪 |

## 哪些情况下看不到收益

- **DP**：梯度直到 backward 末尾才 ready、bucket 太大导致启动很晚、通信比剩余计算更慢，或使用了同步等待，都可能导致尾部仍有明显等待。
- **TP**：GEMM chunk 太小、通信启动开销大、计算与通信争用带宽/SM、或没有真正的异步流水实现，都可能让 overlap 收益很低甚至变慢。
- overlap 只能隐藏通信的一部分，不能减少必须传输的数据量本身；它的收益取决于通信时长与可重叠计算时长。

## 常见误区

1. **“DP 的梯度通信一定发生在下一层之间。”** 不一定。bucket 可跨层，hook 触发时机和 kernel 粒度决定实际重叠窗口；“层间”是便于理解的典型描述。
2. **“DP collective 完成后才开始反向。”** 通常可以异步启动，反向继续执行；但在 optimizer 等同步点必须确保梯度归约已完成。
3. **“TP 通信 overlap 就是 DP 梯度 all-reduce overlap。”** 两者通信对象和依赖都不同：DP 聚合同一参数在不同样本副本上的梯度；TP 交换一个层所需的张量分片/partial。
4. **“标了 overlap 就能把通信完全藏起来。”** 只有当可并行计算足以覆盖通信且资源争用不严重时，才能隐藏大部分通信。

## 自测

1. 为什么 DP 能在反向期间启动梯度 All-Reduce？——因为反向按层逐步产生梯度，某个 bucket ready 后可以异步通信，autograd 还能继续算其他层。
2. DP 的 “层间” 典型是哪个通信和哪个计算重叠？——较靠后层已就绪的梯度通信，与较靠前层尚未完成的反向计算。
3. TP overlap 的通信和计算通常属于什么范围？——同一层内，分块的 All-Gather / Reduce-Scatter 与 GEMM chunks 流水执行。
4. 为什么 overlap 不保证完全没有通信等待？——通信可能比剩余计算慢，或被启动开销、资源争用限制；optimizer 前 DP 梯度仍必须同步完成。

## 已经覆盖 / 下一知识点

- [x] DP 梯度 buckets 的异步归约如何与 backward 重叠；TP 层内通信—GEMM overlap 的区别。
- 下一知识点：用 Megatron / PyTorch 的 profiler trace 识别真实通信区间、bucket 边界和未被隐藏的尾部等待。
