# verbs 程序走读：从一个完整程序看所有资源的依赖顺序

```text
本次讲解位置
章节：communication / RDMA
小节：L1/L2 的实操（Q29、Q30、Q31、Q33、Q34、Q44）
知识点：资源依赖顺序、三个 modify_qp 各填什么、sq_sig_all 与 SIGNALED、RNR 与 flush 的真实序列
上次：GPUDirect 退化（37）——同样是"看起来跑通了但不是你以为的路径"
下次：39 · 注册成本账本（`reg_mr` 那一步在存储路径里为什么会变成瓶颈）```

## 为什么现在讲这个

前面五章都在回答"为什么"：[33](./33-rdma-six-layer-spine.md) 给了路径和成本，
[34](./34-rdma-connection-state-ledger.md) 给了连接状态账本，
[35](./35-rdma-congestion-control-loop.md) 给了拥塞闭环，
[36](./36-rdma-troubleshooting-decision-tree.md) 给了排障树，
[37](./37-rdma-gpudirect-degradation.md) 给了路径判定。

但 [32 章](./32-rdma-interview-questions.md) 的 Q29 是**手写一遍初始化流程**。
这一题用要点是答不出来的——面试官要看你写 `ibv_modify_qp` 的 attr 填什么。
而现场最常出的错，恰好都在这一题里：

```text
qp_access_flags 填在 RTR 而不是 INIT  → EINVAL
CQ 深度按单边设                       → 高负载静默丢 CQE
sq_sig_all=0 却忘了设 SIGNALED        → hang 在 poll，网卡一切正常
对端没预投 recv WR                    → RNR，而不是"连不上"
```

这些都是**资源和顺序**的问题，不是理论问题。所以这一章换一种讲法：
先给你一个能编译的完整程序，再逐段解释每个 `ibv_*` 到底建立了什么依赖，
最后用一个可运行的状态机模拟器把四种典型事故的真实序列跑出来。

## 执行边界（先说清楚，因为这决定你怎么验证）

| 产物 | 能否在本机运行 | 验证方式 |
| --- | --- | --- |
| [`38_minimal_rc.c`](./38_minimal_rc.c) | **不能**（macOS 无 rdma-core） | `clang -fsyntax-only` 语法/类型检查通过 |
| [`38_stub/`](./38_stub) 桩头文件 | — | 只用于上一条的编译检查，不可链接 |
| [`38_verbs_emulator.py`](./38_verbs_emulator.py) | **能** | `python3` 直接跑，输出真实 trace |

真实运行需要在有 RNIC 的 Linux 上：

```bash
# 依赖
sudo apt install librdmacm-dev libibverbs-dev   # 或 dnf install rdma-core-devel

# 编译
gcc -O2 -Wall -Wextra -o minimal_rc 38_minimal_rc.c -libverbs -lrdmacm

# 两个终端
./minimal_rc server 7471
./minimal_rc client 192.168.1.10 7471
```

期望输出（服务器端）：

```text
[server] 监听 7471
[server] 收到连接请求
[server] 连接建立
[ok] 资源就绪 qpn=341
[server] CQE wr_id=0 opcode=128 status=IBV_WC_SUCCESS
[server] 收到 64 字节：hello rdma from client pid=12345
```

**下面所有"跑出来的 trace"都来自模拟器，不是真实网卡。** 模拟器保证"状态与字段的
因果关系"和真实 verbs 一致，不保证任何时间数字。

---

## 心智模型：资源是一棵树，不是一张清单

大多数人把 verbs 初始化背成一个线性列表。它其实是**有依赖的树**：

```text
device (ibv_open_device, rdma_cm 帮你做)
  └── pd                     隔离容器，所有内存和 QP 的命名空间
       ├── mr                注册内存：拿 lkey / rkey
       │    └── (sge 引用 lkey)
       └── qp                SQ + RQ，必须绑定 cq
            ├── sq  ──────────┐
            └── rq            ├── cq   完成队列，深度必须 >= sq + rq
                              ┘
```

三条从这棵树上直接读出来的规则：

1. **MR 必须和 QP 同 PD**。跨 PD 的 lkey 直接 `LOC_PROT_ERR`。
2. **CQ 深度必须是 send + recv 之和**。取小了不会报错，只会在高负载下丢完成。
3. **MR 的生命周期必须覆盖所有引用它的 WR**。`ibv_dereg_mr` 之后还有在途 WR，
   就是野指针 DMA。

再把"状态"叠上去，就是完整的约束图：

```text
RESET ──→ INIT ──→ RTR ──→ RTS
                           │
              任何不可恢复错误 ↓
                          ERROR ──(必须先回 RESET)──→ INIT ...
```

**四个状态是四组字段的开关**，不是四个名词。每一级只允许填它那一段的字段：

| 迁移 | 必须填 | 此时能做 |
| --- | --- | --- |
| → INIT | `port_num`、`pkey_index`、`qp_access_flags` | 什么都不能 |
| → RTR | `dest_qp_num`、`path_mtu`、`rq_psn`、`max_dest_rd_atomic`、`min_rnr_timer`、`ah_attr` | 能收，不能发 |
| → RTS | `sq_psn`、`timeout`、`retry_cnt`、`rnr_retry`、`max_rd_atomic` | 能收能发 |
| ERROR | 只读 | 挂起 WR 全部 flush |

`qp_access_flags` 在 INIT 是本节最容易踩的坑：它控制的是"**对端**能用什么方式访问你
注册的内存"，逻辑上像是 QP 的属性，但规范把它归在 INIT。

## 逐段走读

程序的骨架在 `build_resources()`。按调用顺序看每一条在建立什么。

### 第一段：PD 与 MR

```c
res->pd = ibv_alloc_pd(id->verbs);

res->mr = ibv_reg_mr(res->pd, res->buf, BUFFER_LEN,
                     IBV_ACCESS_LOCAL_WRITE |
                     IBV_ACCESS_REMOTE_WRITE |
                     IBV_ACCESS_REMOTE_READ);
```

三个 access flag 各自对应一件事，缺了会以不同方式失败：

| flag | 缺了会怎样 |
| --- | --- |
| `LOCAL_WRITE` | 本地 `post_recv` 的缓冲不可写 → 收不到数据 |
| `REMOTE_WRITE` | 对端 `WRITE` 进来 → `REM_ACCESS_ERR` |
| `REMOTE_READ` | 对端 `READ` 出去 → `REM_ACCESS_ERR` |

注意 `REMOTE_WRITE`/`REMOTE_READ` 在本程序里其实用不到（只做 SEND），
但保留它们是有意的：你可以把它改成单边写来对比语义，而不用重注册。

### 第二段：CQ 与 QP

```c
res->cq = ibv_create_cq(id->verbs, 16, NULL, NULL, 0);

qp_attr.send_cq    = res->cq;
qp_attr.recv_cq    = res->cq;
qp_attr.qp_type    = IBV_QPT_RC;
qp_attr.cap.max_send_wr  = 8;
qp_attr.cap.max_recv_wr  = 8;
qp_attr.cap.max_send_sge = 1;
qp_attr.cap.max_recv_sge = 1;
qp_attr.sq_sig_all = 0;            /* 默认不产生 CQE */
```

`16` 不是随便写的：它必须 ≥ `max_send_wr + max_recv_wr` = 8 + 8。
如果这里写 8，程序在低负载下能跑，一上并发就在 `ibv_poll_cq` 里丢掉完成事件——
因为网卡没有 CQE 可写时会**静默丢弃**，不会报错。这是最典型的"压测才炸"配置。

`sq_sig_all = 0` 是一个刻意的选择，为了演示第四段的规则：

```text
sq_sig_all = 1  → 每笔 WR 都产生 CQE（简单，但每笔都有 CQE 开销）
sq_sig_all = 0  → 默认不产生，必须逐笔设 IBV_SEND_SIGNALED
```

### 第三段：三个 modify_qp

```c
/* INIT：只填"身份"和权限，此时不能收发 */
attr.qp_state        = IBV_QPS_INIT;
attr.pkey_index      = 0;
attr.port_num        = id->port_num;
attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                       IBV_ACCESS_REMOTE_READ;
ibv_modify_qp(res->qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                              IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);

/* RTR：填"对端是谁、路怎么走"，此时能收不能发 */
attr.qp_state           = IBV_QPS_RTR;
attr.path_mtu           = IBV_MTU_4096;
attr.dest_qp_num        = id->qp->qp_num;
attr.rq_psn             = 0;
attr.max_dest_rd_atomic = 8;
attr.min_rnr_timer      = 12;
attr.ah_attr.is_global  = 1;
attr.ah_attr.port_num   = id->port_num;
attr.ah_attr.grh.hop_limit = 1;
ibv_modify_qp(res->qp, &attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                              IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                              IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);

/* RTS：填重传与超时，此时才能发 */
attr.qp_state      = IBV_QPS_RTS;
attr.timeout       = 14;   /* 4.096us << 14 ≈ 67ms */
attr.retry_cnt     = 7;    /* 7 = 无限重试 */
attr.rnr_retry     = 7;    /* 7 = 无限重试 */
attr.sq_psn        = 0;
attr.max_rd_atomic = 8;
ibv_modify_qp(res->qp, &attr, IBV_QP_STATE | IBV_QP_TIMEOUT |
                              IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                              IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
```

**`attr_mask` 不是可选的冗余参数**：它告诉内核"这次我只想改这些字段"。
mask 与填的字段不一致，就会 `EINVAL`——这就是 Q43 那四类原因里最隐晦的一类。

两个参数值得单独算一遍：

```text
timeout = 14  →  4.096 us × 2^14 = 4.096 × 16384 ≈ 67.1 ms
min_rnr_timer = 12  →  接收端说"我 RNR 了，你等我多久"
retry_cnt = 7 / rnr_retry = 7  →  规范里 7 表示无限
```

生产里 `rnr_retry = 7`（无限）通常**不该用**：对端 recv WR 永远不投的时候，
发送端会永久挂着而不是报错，故障表现为"卡住"而不是"失败"，更难发现。
本程序为了演示贪方便设成 7，正文这里明确标出这个取舍。

### 第四段：投递与轮询

```c
/* 接收侧：必须提前投，否则对端 SEND 会 RNR */
post_recv(res);

/* 发送侧：signaled 才会产生 CQE */
wr.opcode     = IBV_WR_SEND;
wr.send_flags = IBV_SEND_SIGNALED;
post_send(res);

/* 轮询：一定要检查 status */
int n = ibv_poll_cq(res->cq, 1, &wc);
if (n == 1 && wc.status != IBV_WC_SUCCESS) { /* 处理错误 */ }
```

最后这三行是 Q44 的答案。绝大多数示例代码写成：

```c
while (ibv_poll_cq(cq, 1, &wc) == 0) { /* spin */ }
/* 到这里就当成功了 */
```

**`n == 1` 只说明拿到一条 CQE，不说明它成功。** 把这个判断省掉，
所有错误完成都会被静默吞掉，最后表现成"数据不对但没报错"。

---

## 模拟器：四种事故的真实序列

上面说的都是"应该怎样"。[`38_verbs_emulator.py`](./38_verbs_emulator.py)（364 行）
把 QP 状态机和队列建成一个语义模型，直接跑：

```bash
python3 communication/38_verbs_emulator.py
```

### 场景 A：正常路径

```text
初始状态： RTS / RTS
B 预投 recv WR：3 个
  send wr_id=0 -> OK，A 的 CQ 有 1 条
  send wr_id=1 -> OK，A 的 CQ 有 2 条
  send wr_id=2 -> OK，A 的 CQ 有 3 条
  A poll 到 wr_id=0 status=IBV_WC_SUCCESS
  ...
  B poll 到 wr_id=100 status=IBV_WC_SUCCESS
```

注意 **SEND 是双边的**：一次成功在发送端产生 1 条 CQE，在接收端也产生 1 条
（用掉的那个 recv WR 的 `wr_id`）。这就是 [33 章](./33-rdma-six-layer-spine.md)
L2 那张表第一行的具体后果——**同一笔消息，两个 CQ 各有一条完成**。

### 场景 B：modify_qp 填错

```text
  ✗ INIT: EINVAL，缺字段 ['qp_access_flags']
  跳过 RTR 直接 RTS：
  ✗ RTS: 不能从 RESET 直接到 RTS，必须逐级（RESET→INIT→RTR→RTS）
```

两种错，一个错误码。这就是 Q43 为什么难答：`EINVAL` 不区分原因，
你必须逐个字段排除。

### 场景 C：对端没投 recv WR

```text
  B 的 RQ 长度：0
  send -> RNR_EXC，A 拿到 IBV_WC_RNR_RETRY_EXC_ERR
  修法一：补投 recv WR
  send -> OK
```

**这里有个关键对照**：应用看到的是"hang 在等完成"，而拿到的是 RNR 错误。
排除方向完全不同——不是"网络不通"，是"对端应用没投缓冲"。
[36 章](./36-rdma-troubleshooting-decision-tree.md) 把它排在场景一的第 1 假设，
就是因为它比"链路故障"常见得多。

### 场景 D：一个错误把 QP 打进 ERROR

这是本章最重要的一段输出：

```text
  A 的 SQ 里有 5 笔
  wr_id=1 -> REM_ACCESS，QP 进入 ERROR
  连带 flush 3 笔未完成的 WR

  CQE 序列（注意根因排在最前）：
    wr_id=0  IBV_WC_SUCCESS
    wr_id=1  IBV_WC_REM_ACCESS_ERR      ← 根因
    wr_id=2  IBV_WC_WR_FLUSH_ERR        ← 连带
    wr_id=3  IBV_WC_WR_FLUSH_ERR        ← 连带
    wr_id=4  IBV_WC_WR_FLUSH_ERR        ← 连带
```

**3 笔被 flush，1 笔是根因，1 笔已成功。** 排障时如果只看见后面三条 flush，
会去查 wr_id=2/3/4 的参数——全是合法的 WR，永远查不出来。
[36 章](./36-rdma-troubleshooting-decision-tree.md) 那条"往前找第一个非 flush 错误"
就是从这条序列来的。

顺带一个容易忽略的点：**flush 的 WR 数会变**。这里是 3，因为在途的是 2/3/4；
如果 SQ 里还排着 20 笔，就是 20 条 flush。所以"CQE 数量和 WR 数量对不上"
本身不是异常，是 ERROR 状态的正常表现。

### 场景 E：sq_sig_all=0 且没设 SIGNALED

```text
  发了 4 笔、全部未 signal，A 的 CQ 长度 = 0
  -> 应用会 hang 在 poll 里，但网卡和连接都是好的

  把最后一笔设成 signaled：
  poll 到 wr_id=4 status=IBV_WC_SUCCESS
  -> 它同时也证明前 4 笔已经完成（batching 的常用手段）
```

这一段解释了为什么生产代码常用"批量 + 最后一笔 signal"：
既省 CQE，又能在需要时得到同步点。代价是**中间那些 WR 失败你不会知道**——
它们可能已经落进 ERROR 的 flush 里，而你的同步点只报最后一笔的状态。

---

## 编译检查：怎么在没有网卡的机器上验证

macOS 上没有任何 rdma 头文件，但**语法和类型仍然可以验证**。
[`38_stub/`](./38_stub) 提供了一套只声明接口、不含实现的桩头文件：

```bash
cd ~/Downloads/learning_notes
clang -fsyntax-only -Wall -Wextra -I communication/38_stub \
      communication/38_minimal_rc.c
```

输出（本机实际执行）：

```text
SYNTAX OK
```

这条检查能证明什么、不能证明什么，要分清：

| 能证明 | 不能证明 |
| --- | --- |
| 每个 `ibv_*` 的参数数量和类型对得上 | 真实头文件的字段名一致（桩是按本文用法写的） |
| `attr_mask` 与填的字段在类型上兼容 | 运行时的 `EINVAL`、`ENOMEM` |
| 没有明显的空指针/未初始化使用 | 网卡上的实际行为 |
| 控制流完整（所有分支都有返回） | 性能 |

**所以它只是"形状检查"**，真正的验证必须在有 RNIC 的 Linux 上跑。
这也是本节开头那张执行边界表存在的原因：不把静态检查冒充成运行验证。

## 常见错法，以及症状

| 错法 | 症状 |
| --- | --- |
| CQ 深度按单边设（`max_send_wr`） | 低负载正常，压测时 `poll` 丢完成，表现为"消息偶尔丢" |
| 只判断 `poll > 0` 不看 `wc.status` | 错误被静默吞掉，最后是"数据不对但不报错" |
| `qp_access_flags` 填在 RTR | `modify_qp` 返回 `EINVAL`，看不出原因 |
| `attr_mask` 与字段不一致 | 同上，`EINVAL` |
| 接收端启动晚于发送端 | 发送端 `RNR_RETRY_EXC_ERR`，不是"连不上" |
| `rnr_retry=7`（无限）用在生产 | 对端永不投缓冲时永久挂起，而不是快速失败 |
| `ibv_dereg_mr` 之后还有在途 WR | 野指针 DMA，表现为随机数据损坏或内核报错 |
| MR 注册在 A 的 PD、QP 在 B 的 PD | `LOC_PROT_ERR`，报错指向 MR 但其实不是 MR 的问题 |
| 多线程共用一个 QP 却不加锁 | WR 顺序错乱，`wr_id` 与数据错配，偶发 |
| `sq_sig_all=0` 忘记 SIGNALED | 应用 hang，网卡/连接全正常（场景 E） |

倒数第四条（跨 PD）特别值得记：报错是 `LOC_PROT_ERR`，字面意思是"本地保护错误"，
很容易去查 MR 的 access flag，但真正的原因是 **QP 和 MR 不在同一个 PD**。

## 模型边界

模拟器是语义模型，明确没有建模：

- **时间**：没有 PCIe、网络、流水线延迟，所以不能拿它估延迟
- **真实重传**：RNR 是直接判定的，没有 go-back-N（见 [33 章](./33-rdma-six-layer-spine.md)）
- **多 WR 并发与乱序完成**：CQ 是 FIFO，真实网卡可能乱序
- **资源上限**：没有 `max_send_wr` 超限后的 `ENOMEM`
- **多线程**：没有建模 [32 章 Q34](./32-rdma-interview-questions.md) 的竞争

真机验证顺序建议：先单机 loopback 跑通场景 A，再故意不投 recv 复现场景 C，
最后人为把 rkey 改错复现场景 D。**三个都复现过，才算真的会写 verbs 程序。**

---

## 自测

1. CQ 深度为什么必须是 send + recv 之和？取小了会怎样？
2. `qp_access_flags` 在哪个状态填？填晚了报什么错？
3. 一次成功的 SEND，涉及几个 CQ、几条 CQE？WRITE 呢？
4. 场景 D 里为什么是 3 笔 flush 而不是 2 笔或 4 笔？
5. `attr_mask` 的作用是什么？mask 与字段不一致会怎样？
6. 为什么 `rnr_retry=7` 在生产里不该用？
7. 只看 `ibv_poll_cq` 返回值不看 `wc.status`，会导致什么现象？

<details>
<summary>答案</summary>

1. 因为 SQ 和 RQ 共用同一个 CQ，每条完成都要占一个 CQE 槽位。
   取小了网卡没有 CQE 可写时**静默丢弃**，表现为低负载正常、压测偶发丢消息。
2. **INIT**。它控制对端能用什么方式访问你的 MR，但规范把字段归在 INIT。
   填到 RTR 就 `EINVAL`（[38 章正文第三段](#第三段三个-modify_qp)）。
3. SEND：**2 个 CQ、2 条 CQE**（发送端一条、接收端消耗的 recv WR 一条）。
   WRITE：**只有发送端 1 条**，除非用 `WRITE_WITH_IMM` 才给对端一条。
4. 因为在途的是 wr_id=2/3/4 三笔：wr_id=0 已完成、wr_id=1 是根因。
   flush 数量 = ERROR 发生时 SQ 里剩余的 WR 数。
5. 声明"本次只改这些字段"。mask 与填的字段不一致（少填或多填）都会 `EINVAL`。
6. `rnr_retry=7` 表示无限重试。对端永远不投 recv WR 时，发送端会永久挂起，
   故障表现成"卡住"而不是"失败"，排查成本高得多。应设有限值快速失败。
7. 所有错误完成被当成成功忽略。表现为"数据不对但不报错"、或 `wr_id` 与数据错配，
   排查时会误以为是内存/协议问题。

</details>

---

## 一页速记

```text
资源树（依赖关系，不是清单）:
  pd ── mr ─── lkey (SGE 引用)
   └── qp ── sq/rq ── cq
  规则: MR 与 QP 同 PD；CQ 深度 >= send + recv；MR 活得比所有 WR 长

状态机（四组字段的开关）:
  INIT: port_num, pkey_index, qp_access_flags
  RTR : dest_qp_num, path_mtu, rq_psn, max_dest_rd_atomic,
        min_rnr_timer, ah_attr
  RTS : sq_psn, timeout, retry_cnt, rnr_retry, max_rd_atomic
  ERROR: 只读，挂起 WR 全部 flush
  attr_mask 必须与填的字段一致，否则 EINVAL

超时换算:
  timeout = 4.096us << value      value=14 -> 67.1 ms
  retry_cnt / rnr_retry = 7 表示无限（生产慎用）

完成语义:
  SEND        2 条 CQE（发送 + 接收）
  WRITE       1 条（只有发送端）
  sq_sig_all=0 -> 每笔要自己设 SIGNALED，否则永远等不到

三个必查:
  poll 到 CQE != 成功（必须查 wc.status）
  一次错误把 QP 打进 ERROR -> 后面的 WR 全是 WR_FLUSH_ERR
  根因 = 第一个非 flush 的错误

验证边界:
  本机只能 clang -fsyntax-only（形状检查）
  真机三步：loopback 通 -> 不投 recv 复现 RNR -> 改错 rkey 复现 flush
```

---

## 附录 A：完整程序

[`38_minimal_rc.c`](./38_minimal_rc.c) 共 465 行，含服务器与客户端两个角色、完整的资源
建立与销毁、事件循环、错误检查。正文没有逐行贴全，因为它的价值在于**调用顺序**
而不是每行代码；完整源码就在章节旁边，并且被 `clang -fsyntax-only` 检查过。

读它的建议顺序：

```text
build_resources()      资源树 + 三个 modify_qp（正文逐段讲过）
post_recv() / post_send()   投递，注意 signaled
wait_completion()      轮询，注意 wc.status 检查
on_established()       服务器投 recv、客户端填 payload
run_server() / run_client()  事件循环的差异
```

## 附录 B：模拟器

[`38_verbs_emulator.py`](./38_verbs_emulator.py) 的核心是 `QP` 类和 `deliver()`：

```python
class QP:
    REQUIRED = {
        QPState.INIT: ("port_num", "pkey_index", "qp_access_flags"),
        QPState.RTR: ("dest_qp_num", "path_mtu", "rq_psn",
                      "max_dest_rd_atomic", "min_rnr_timer", "ah_attr"),
        QPState.RTS: ("sq_psn", "timeout", "retry_cnt", "rnr_retry",
                      "max_rd_atomic"),
    }

    def modify_qp(self, target, **attrs):
        err = transition_error(target, self.state)
        if err:
            raise ValueError(f"{target.value}: {err}")
        missing = [a for a in self.REQUIRED[target] if a not in attrs]
        if missing:
            raise ValueError(f"{target.value}: EINVAL，缺字段 {missing}")
        ...

    def flush_all(self):
        """QP 进 ERROR 后，SQ/RQ 里剩余的 WR 全部以 WR_FLUSH_ERR 完成。"""
        for e in self.sq:
            if e.signaled:
                self.cq.append(CQE(e.wr_id, Status.WR_FLUSH, e.opcode))
        ...
```

`REQUIRED` 这张表就是"四组字段的开关"的可执行版本——它让"填错就 EINVAL"
从一个口头结论变成可以复现的输出。
