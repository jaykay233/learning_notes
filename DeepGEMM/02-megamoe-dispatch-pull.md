# DeepGEMM MegaMoE：workspace、slot、metadata 与 remote pull

## 本次讲解位置

```text
章节：DeepGEMM MegaMoE 执行路径
小节：Dispatch 元数据发布与输入 pull
知识点：workspace 中的计数器、slot offset、源 token metadata、跨 rank pull
上次：DeepGEMM 项目定位与普通 GEMM 的流程边界
下次：FP8 的 x_sf、UTCCP 4×32 scale 布局及 A/SFA、B/SFB warp
PTX：本节是系统数据流；原子计数、全局同步、NVLink symmetric buffer、TMA copy 各自承担不同职责
```

### 为什么现在讲这个

GEMM 只知道“第几行 M”并不知道这一行最初来自哪个 rank、哪个 token、哪个 top-k 分支。MoE dispatch 必须把动态路由变成一个可以并发写入、可以跨 rank 找回源数据的索引系统。理解 workspace 和 slot 之后，像“slot 12 从哪里来”就能按计数和分配顺序一步步算出来，而不是把 12 当成某个硬件固定编号。

## 1. Workspace 是什么

在这个 kernel 流程里，workspace 是 GPU kernel 之间/CTA 之间用来协作的全局内存区域，存放计数、同步状态、metadata 或临时调度信息。它不是“显存里任意一块临时 tensor”的泛称：每个字段都需有明确的生产者、消费者、索引空间和发布顺序。

可先按用途分成三类：

| 类别 | 示例 | 回答的问题 |
|---|---|---|
| 分配计数 | 每个 expert 的 send count | 这个 token 应分到该 expert 的哪个连续槽位？ |
| 来源 metadata | `src_token_topk_idx` 一类编码 | 这个槽位对应原始哪个 token、哪个 top-k？ |
| 同步状态 | receive count、ready/count/barrier 状态 | 上游写完了吗？下游可以读取了吗？ |

workspace 是协议的一部分；单独有一个计数值并不代表它自动保证数据可见，仍需正确 atomic ordering、grid/rank 同步和发布顺序。

## 2. 从 expert count 到 slot offset

dispatch 会统计 token-topk 路由到各 expert 的数量。多个 SM 并发处理时，单个 SM 得到本地的 `expert_token_count`。接着用 atomic add 把本地数量加到该 expert 的全局 send count。atomic add 返回的旧值就是该 SM 获得的连续区间起点。

公式：

```text
base_offset = atomic_add(global_expert_send_count, local_count)
本 SM 给本地第 i 条记录分配的 slot = base_offset + i
```

具体例子：某 expert 的全局计数原来是 10；本 SM 有 3 条该 expert 记录。atomic add 将全局计数更新为 13，并把旧值 10 返回给本 SM：

| 本地记录顺序 `i` | 得到的 slot |
|---:|---:|
| 0 | 10 |
| 1 | 11 |
| 2 | 12 |

所以 **slot=12 是“旧计数起点 10 + 本 SM 内局部序号 2”**。如果原计数或 atomic 执行次序不同，slot 当然可能不是 12。slot 是 expert 发送/接收池中的位置，不是原 token index，也不天然等于 rank id。

多 SM 之间 atomic add 的线性化顺序可能不同；不应假设 SM0 永远拿第一段。正确性来自每个 SM 用 atomic add 返回的 base，并把自己的 metadata 写到相同分配结果。

## 3. metadata 把 slot 连回原始 token

slot 只回答“写进池的第几个位置”，还需要 metadata 回答“这条路由来自谁”。dispatch 扫描有效的 token-topk 项，为每个目标槽发布源 token 信息，逻辑上至少包含：

```text
原 token_idx + topk_idx
```

某些路径还需 rank/目标位置等信息。实现可能把 token 与 top-k 编码到紧凑整数中；学习时先把它看成一个 tuple，不要在不了解位宽和编码函数前自己假设 bit field。

示例：

```text
slot 12 的 metadata -> (src_rank=1, src_token=7, src_topk=0)
```

那么后续 pull 虽然从接收池的 slot 12 读取，却能依据 metadata 找到 rank 1 上 token 7 的输入，并知道它是这个 token 的第 0 个 top-k 分支。

## 4. 为什么要全局同步和发布 receive count

本地 SM 写完 metadata 不等于所有消费者都可以读：另一个 SM 或 GPU rank 可能还没看到写入。dispatch 因此有全局同步/跨 rank barrier 边界。同步完成后，再把每个 expert 的 send count 汇总/发布为消费者可见的 receive count，并计算接收总数供 scheduler 确定工作量。

顺序关系可以画成：

```text
所有 producer 写 metadata
       ↓
grid / rank 级同步，确保写入到达可见边界
       ↓
发布 expert_recv_count 与总接收量
       ↓
scheduler / pull 读取 metadata 和 payload
```

计数发布不能早于相应数据可见。否则消费者可能看到“有 100 条可读”，但 metadata 或 token payload 只写完了 90 条。

## 5. Remote pull：由来源 metadata 找回 token

接收端按 local expert 的接收数量处理槽位，按某个调度策略（例如 round-robin 地遍历源 rank/源 slot）找到该槽对应的 metadata，再还原源 rank、源 token 与 top-k。随后从远端 symmetric buffer 拉取：

- FP8 token 激活数据；
- 对应激活的 scale factors；

并把它们写进本地按 expert 连续组织的输入池。到这一步，后面的 GEMM 不需要逐行跨 rank 找输入，可以像普通矩阵乘那样从本地池取 M 行。

要区分三套位置：

| 位置 | 意义 |
|---|---|
| 原始 `token_idx` | 用户输入 batch 中的 token 身份 |
| `(expert, slot)` | dispatch 为 routed token 分配的专家池位置 |
| 本地 L1 pool row | pull 后接收端 GEMM 的连续 M 行位置 |

它们可以对应同一个 routed token，但数值不必相等。

## 6. TMA 分 chunk 拉取与 SF 搬运

payload 大时，dispatch warp 不需要用 lane-by-lane 标量 load/store 复制每个 token 字节。路径可以使用 TMA 把远端 token chunk 搬到 warpgroup/CTA 的 shared-memory send buffer，再 TMA store 到本地 L1 token pool；每个搬运 stage 有 mbarrier 等机制表示数据完成。与此同时，线程按对应布局复制 scale factor，并把 M/token 索引改写成消费者需要的 UTCCP-friendly layout。

```text
源 rank symmetric buffer
  ├─ FP8 token chunk --TMA--> SMEM send buffer --TMA--> 本地 L1 token pool
  └─ x_sf ----------------线程按索引变换复制-----------> 本地 L1 SF pool
```

两路数据必须保持同一 routed token 对齐：token 行变换错一格，或者 scale 还留在旧行号，数值会错误，即使 token 字节本身搬运正确。

## 7. 常见错误与症状

| 错误理解/实现 | 实际风险 | 可见症状 |
|---|---|---|
| 把 slot 当作 token_idx | 丢失路由池与原 batch 索引的区分 | 输出写错 token，或 pull 读错源行 |
| 用本地 count 当全局 offset | 不同 SM 分配区间重叠 | metadata 覆盖、接收数量对不上 |
| 假设 atomic add 的顺序固定 | 代码依赖未承诺的 SM 执行次序 | 并发或不同 launch 下数据错位 |
| receive count 先于 metadata/payload 发布 | 消费者提前读取未完成项 | 偶发读到旧 metadata/未初始化 payload |
| 只拉 FP8 token，不同步搬其 scale | FP8 值与 scale 行错配 | 反量化幅值异常、误差随 token 变化 |
| 把远端 slot、源 token 与本地 pool row 混成一个索引 | 读写使用错地址空间 | 越界或 token/expert 串位 |

## 自测（含答案）

1. **atomic add 返回的旧值是什么？**  答：本 SM 获得的 slot 连续区间起点。
2. **全局计数旧值为 10，本 SM 分到 3 个 token-topk 项，slot 范围是什么？**  答：10、11、12。
3. **slot 能否直接当原始 token_idx？**  答：不能；slot 是专家池位置，需 metadata 映射回源 token/top-k。
4. **为什么等所有 metadata 写完再发布 receive count？**  答：避免消费者根据新计数读取尚未完成的 metadata/payload。
5. **pull 时为何 token 和 scale 必须一起保持行对齐？**  答：scale 是对应 token FP8 激活的量化信息，错行会导致反量化错误。
