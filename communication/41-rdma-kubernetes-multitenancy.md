# Kubernetes 里的 RDMA：三层隔离边界与八个必查项

```text
本次讲解位置
章节：communication / RDMA
小节：L5 工程（Q52、Q54 的实操版）
知识点：VF / PD / P_Key 三层边界、memlock 与 IPC_LOCK、GID index 选型、Multus 与设备插件、容量上限
上次：集合通信与 rail 拓扑（40）——拓扑能不能用，取决于流量形态
下次：回到 L3 把可靠性剩余机制补齐（PSN 窗口 / 超时与重传的定量），或 NVMe-oF 全链路```

## 为什么现在讲这个

[32 章 Q52/Q54](./32-rdma-interview-questions.md) 是两道"运维题"：
一道问容器里怎么把 RDMA 设备给进去，一道问多租户怎么隔离。
它们看起来和前面 8 章的物理、语义、拥塞无关，其实问的是**同一个问题**：

> 你在 [34 章](./34-rdma-connection-state-ledger.md) 里学的那些状态对象，
> 哪些能当隔离边界用？哪些只是命名空间？

答案分三层，而绝大多数线上事故是因为**把命名空间当成了硬件边界**：

```text
VF       硬件边界  —— PCIe 功能级，独立的 QP 空间、独立的 GID
PD       命名空间  —— 只在同一进程/同一驱动实例内做隔离，不是墙
P_Key    逻辑分区  —— 需要 SM/交换机配合，RoCE 下通常不用
```

这一章把这三层讲透，再给一份**可执行的检查表**（脚本
[`41_k8s_rdma_validator.py`](./41_k8s_rdma_validator.py)），
每一项检查都对应一个具体的故障症状。

---

## 心智模型：三层边界，越往下越"软"

```text
┌─────────────────────────────────────────────────────────┐
│ VF（SR-IOV Virtual Function）      真正的墙              │
│   · 独立 PCIe 功能、独立 QP 资源池、独立 GID             │
│   · 一个 VF 被 pod A 独占，pod B 拿不到它的 /dev 节点    │
│   · 但共享物理口的带宽 —— 隔离是"资源可见性"，不是"速率" │
├─────────────────────────────────────────────────────────┤
│ PD（Protection Domain）            命名空间              │
│   · 只在同一 uverbs 设备内给 MR/QP 分组                  │
│   · 防止"用错 lkey"，不防止"故意越权"                    │
│   · 同 VF 内两个进程若共享 PD 或泄漏 rkey，就能互读内存   │
├─────────────────────────────────────────────────────────┤
│ P_Key                              逻辑分区              │
│   · 二层分区，需要 SM（IB）或交换机配置（RoCE 少用）      │
│   · 默认 0xffff 是 full membership，谁都能通              │
└─────────────────────────────────────────────────────────┘
```

**Q54 的正确答案就是这棵树**：`PD` 是命名空间，`VF` 才是边界。
只说 PD 会被追问"那我拿到 rkey 呢"，只说 VF 会被追问"那同一 pod 里两个进程呢"。

### 为什么 PD 不是墙

PD 的设计目标是**防止意外**，不是**防止恶意**：

```text
PD 能防：
  QP(A) 用 MR(B) 的 lkey  ->  LOC_PROT_ERR     ← 手滑写错变量
  MR 跨 PD 被引用          ->  注册/改 QP 时失败

PD 不能防：
  同 PD 内两个进程互读            <- 它们本来就共享 lkey
  rkey 通过日志/共享内存泄漏      <- rkey 就是一个 32 位凭据
  内核模块/DMA 直接访问物理地址   <- 不在 PD 的管辖范围
```

所以多租户的正确做法是：**一个租户一个 VF**，PD 只当作代码正确性的护栏。

---

## 检查表：八个必查项

脚本对一份配置跑八条规则，每条对应一个真实症状：

```bash
cd ~/Downloads/learning_notes
python3 communication/41_k8s_rdma_validator.py
```

| # | 检查项 | 配错后的症状 |
| --- | --- | --- |
| 1 | `memlock=unlimited` | `ibv_reg_mr` 返回 `ENOMEM`，小 MR 侥幸成功、大页必失败 |
| 2 | `capabilities: IPC_LOCK` | `mlock`/大页分配 `EPERM`，报错和网络无关 |
| 3 | `/dev/infiniband` 已挂载 | `ibv_get_device_list` 返回 0，"no RDMA devices found" |
| 4 | Multus 挂了 RDMA 设备网络 | pod 只有 veth，走内核栈，RDMA 根本不可用 |
| 5 | GID index 对应的类型 | 对端 RoCEv2 而本端选了 v1 地址 → 握手失败或静默退回 TCP |
| 6 | VF 数 ≥ pod 数 | 多出来的 pod 卡 `Pending`（`Insufficient rdma/...`） |
| 7 | 多租户各占独立 VF | 共 VF 时 PD 挡不住 rkey 泄漏 |
| 8 | pkey 与对端一致 | 非 `0xffff` 时对端不通，且报错不指向 pkey |

### 第 1、2 项：容器权限

这两项是**新集群第一次跑 RDMA 时最常见的失败**，而且症状完全不像网络问题。

```yaml
securityContext:
  capabilities:
    add: ["IPC_LOCK"]        # 允许 mlock / 锁页
  # 有些发行版还需要 privileged 或 specific device cgroup 规则
```

`memlock` 的默认值通常是 **64 KiB**（容器运行时默认），而 RDMA 注册内存需要
`RLIMIT_MEMLOCK` 至少覆盖你要注册的总量。这就是为什么：

```text
用 4 KB 的小 MR 做 smoke test  -> 通过
换成 1 MB 的真实 buffer        -> ENOMEM
```

**这个"小消息能跑、大消息报 ENOMEM"的形状，是 memlock 的指纹。**
[36 章](./36-rdma-troubleshooting-decision-tree.md) 的三层证据模型里，它属于"第一层就出不去"——
包都没发，所以对端计数器全是 0。

### 第 5 项：GID index 是最隐蔽的一项

RoCE 的每个端口有一张 GID 表（IPv6 格式的地址表），每一项有类型：

```text
$ show_gids
DEV     PORT  INDEX  GID                                     IPv4            VER  DEV
mlx5_0  1     0      fe80:0000:0000:0000:...:0001            fe80:...:1      v1   eth0
mlx5_0  1     1      fe80:0000:0000:0000:...:0001            fe80:...:1      v2   eth0
mlx5_0  1     2      0000:0000:0000:0000:...:0001            192.168.1.10    v1   eth0
mlx5_0  1     3      0000:0000:0000:0000:...:0001            192.168.1.10    v2   eth0
```

**同一个 IP 会出现两次：一次 v1、一次 v2。** `index 0` 通常是 link-local 的 v1 地址。

如果你的代码写死 `gid_index = 0`：

```text
对端也是 v1        -> 能通（但很多新驱动已禁用 v1）
对端只有 v2        -> 握手失败；某些实现会静默回退到 TCP，带宽掉两个量级
```

回退到 TCP 是最坏的情况：**连接是通的、带宽是有的、只是慢**，所有监控都正常。
这和 [37 章](./37-rdma-gpudirect-degradation.md) GPUDirect 退化到 bounce buffer
是同一类病：**没有错误，只有性能**。

正确做法不是写死索引，而是按类型选择：

```python
# 伪代码：从 gid 表里挑类型匹配的那一项
for idx, gid in enumerate(gid_table):
    if gid.type == "RoCE v2" and gid.is_ipv4_mapped():
        bind_gid_index(idx)
        break
```

### 第 4 项：pod 的网络是怎么接进来的

RDMA 需要**设备节点**和**网络命名空间内的接口**同时到位，两者来源不同：

```text
设备节点 /dev/infiniband/*      <- 来自 device plugin（volume mount）
网络接口 + IP + 路由            <- 来自 CNI（Multus 的 sriov/macvlan 插件）
```

只做前者：应用能打开设备，但 GID 表里是宿主的地址，连到错误的网络。
只做后者：容器有 VF 的 netdev，但没有 `/dev/infiniband/uverbs*`，打不开设备。

**两者都要。** 这是 Q52 的完整答案，也是很多"按文档配了一半"的故障来源。

### 第 6 项：VF 是硬上限，不会降级

```text
VF/口  口数   可用 pod
   8     1        8
  32     2       64
```

超过上限的 pod **直接 Pending**，不会自动降级到"共享物理口"。
所以扩容时必须同时改两个地方：节点上的 VF 数（需要重启/重配 SR-IOV），
和 device plugin 暴露的资源数。

## 脚本实际输出

### 全对时的样子

```text
场景：单租户训练 pod（配对了）
  ok   memlock=unlimited
  ok   capabilities: IPC_LOCK
  ok   /dev/infiniband 已挂载
  ok   Multus + RDMA 设备网络
  ok   gid_index=3 (RoCE v2)
  ok   8 VF >= 1 pod
  ok   单租户
  ok   pkey=0xffff（full membership）

结论：0 FAIL / 0 WARN / 8 PASS  => 配置自洽
```

### 忘了 memlock

```text
 FAIL  memlock=64KiB
        -> ibv_reg_mr 返回 ENOMEM；小 MR 可能侥幸成功，一大页就失败
结论：1 FAIL / 0 WARN / 7 PASS  => 这个配置跑不起来，先修 FAIL
```

### GID index 选到 v1

```text
 FAIL  gid_index=0 是 RoCE v1
        -> 对端是 RoCEv2 时握手失败或静默回退到 TCP；ibv_modify_qp 可能返回 EINVAL
```

### 多租户共用一个 VF（Q54 的反例）

```text
 FAIL  4 租户共用一个 VF
        -> PD 只在进程内做命名空间隔离，不是硬件边界：拿到 rkey 就能读写同 VF 内其他进程的内存
```

### hostNetwork 图省事

```text
 warn  hostNetwork=true
        -> 能跑通，但 pod 与宿主共享 netns：GID/路由互相污染，且失去网络隔离
 warn  hostNetwork 模式
        -> 1 个 pod 抢同一块物理口：没有 VF 隔离，QoS/限速只能靠交换机
```

`hostNetwork` 值得单独说一句：它**能跑**，所以常被当成"配好了"。
代价是三件事同时失去——网络命名空间隔离、GID 选择的独立性、以及一台机器只能跑一个
RDMA pod。单机 demo 完全没问题，上生产就是债。

## 常见错法，以及症状

| 错法 | 症状 |
| --- | --- |
| `memlock` 用默认 64 KiB | 小 MR 通过、大 buffer `ENOMEM` |
| 只挂设备不做 CNI | 能 open 设备，但 GID 是宿主的，连错网络 |
| 只做 CNI 不挂设备 | 有网卡但 `ibv_get_device_list` 返回 0 |
| 写死 `gid_index=0` | v1/v2 不匹配 → 握手失败或静默回退 TCP（慢但"正常"） |
| 用 PD 当多租户边界 | 同 VF 内拿到 rkey 就能跨进程读写 |
| VF 超卖 | pod `Pending`，`Insufficient rdma/...` |
| hostNetwork 当标准方案 | 单机正常，多 pod 互相污染 GID/路由 |
| 以为 VF 隔离了带宽 | VF 只隔离资源可见性，物理口带宽仍共享，需要 QoS |
| 忽略 pkey 与对端不一致 | 对端不通，错误信息不指向 pkey |

最后一条值得展开：`pkey` 不是 `0xffff` 时，两端必须一致，而**错误表现是"连不上"**，
和链路故障无法区分。如果集群里有人为了"隔离"随手改了 pkey，排查会非常痛苦。
所以在 RoCE 部署里，默认就不要用 pkey 做隔离，用 VF。

## 和前面章节的接点

```text
34 章 连接状态账本   -> 每个 VF 的 QP 上下文是独立资源池，VF 数决定连接上限
36 章 排障树         -> memlock / 设备缺失属于"第一层就出不去"，对端计数器全 0
37 章 GPUDirect      -> 同样是「没有错误，只有性能」的退化，判定方法一致
40 章 拓扑           -> pod 落在哪条 rail 上，由 CNI 与 device plugin 共同决定
```

最后一条容易被忽略：**K8s 调度器不知道 rail 拓扑。**
如果你的 pod 被随机调度到不同 rail，[40 章](./40-rdma-collective-topology.md) 里
那个"87.5% 跨 spine"的问题会在你完全没改代码的情况下出现。
生产上要么用 topology-aware 调度，要么接受这个损失。

## 模型边界

- 脚本只做**规则判定**，不连集群、不读真实节点状态
- `gid_index` 的合法性依赖具体驱动版本，脚本只能判断"类型是否匹配"
- 没有建模 CNI 的具体实现差异（sriov / macvlan / ipvlan / host-device）
- 没有建模 VF 的带宽 QoS（`max_tx_rate`），这部分要在交换机/驱动侧配
- 没有建模多网卡多 rail 的路由选择，那需要看 `ip route` 与 GID 表的组合

---

## 自测

1. 三层隔离边界分别是什么？哪一层是真正的墙？
2. 为什么 `PD` 不能当多租户边界？它能防住什么、防不住什么？
3. 容器里 `ibv_reg_mr` 返回 `ENOMEM`，最可能的原因是什么？为什么"小 buffer 能过"？
4. 同一个 IP 在 GID 表里出现两次，为什么？选错会有什么后果？
5. 只挂 `/dev/infiniband` 不做 CNI，或者反过来，分别会怎样？
6. VF 数量不够时，多出来的 pod 会怎样？会降级到共享物理口吗？
7. 为什么说 K8s 调度器会让 rail 拓扑的收益消失？

<details>
<summary>答案</summary>

1. `VF`（硬件，PCIe 功能级）、`PD`（命名空间）、`P_Key`（二层逻辑分区）。
   **只有 VF 是墙**，PD 和 P_Key 都不是硬件边界。
2. PD 只能防"意外用错 lkey"（`LOC_PROT_ERR`），防不住恶意：
   同 PD 内进程本来就共享 lkey，rkey 泄漏后可以直接访问。
   多租户必须一租户一 VF。
3. `memlock`（`RLIMIT_MEMLOCK`）是默认 64 KiB。
   小 MR 在这个额度内所以通过；大 buffer 超过额度就 `ENOMEM`。
   "小能跑大不能跑"是它的指纹。
4. 因为 RoCE 同时注册 v1 和 v2 两个 GID 条目（同一个 IP）。
   选到 v1 而对端只有 v2 → 握手失败，或静默回退到 TCP（慢但"正常"，最难查）。
5. 只挂设备：能 open 设备，但 GID 表是宿主的，连到错误的网络。
   只做 CNI：有 netdev，但没有 `/dev/infiniband/uverbs*`，`ibv_get_device_list` 返回 0。
   **两者必须同时到位。**
6. `Pending`，报 `Insufficient rdma/...`。**不会降级**到共享物理口，VF 数是硬上限。
7. 因为调度器只看 CPU/内存等资源，不知道 rail 编号。
   pod 被随机放到不同 rail 后，[40 章](./40-rdma-collective-topology.md) 里
   "同 rail 概率 = 8/EP" 的优势直接丢失，All-to-All 会出现 87.5% 跨 spine 的情况。

</details>

---

## 一页速记

```text
三层边界（越往下越软）:
  VF      硬件墙：独立PCIe功能、独立QP池、独立GID     <- 多租户只能靠它
  PD      命名空间：防手滑用错 lkey，不防恶意
  P_Key   二层分区：需 SM/交换机配合，RoCE 默认别用

八个必查项:
  1 memlock=unlimited          错 -> reg_mr ENOMEM（小能跑大不能跑）
  2 capabilities: IPC_LOCK     错 -> mlock EPERM
  3 /dev/infiniband 挂载       错 -> 0 devices
  4 Multus RDMA 网络           错 -> 只有 veth
  5 gid_index 类型匹配 v2      错 -> 握手失败或静默回退 TCP
  6 VF >= pod                  错 -> Pending（不降级）
  7 多租户独立 VF              错 -> rkey 泄漏即越权
  8 pkey 一致                  错 -> 连不上，报错不指向 pkey

设备节点 与 网络接口 来源不同:
  /dev/infiniband/*   <- device plugin（volume mount）
  netdev + IP + 路由  <- CNI（Multus + sriov/macvlan）
  两者缺一不可

容量:
  VF 数是硬上限，超了 Pending；VF 隔离的是资源可见性，不是带宽
```

## 附：校验器怎么改

[`41_k8s_rdma_validator.py`](./41_k8s_rdma_validator.py) 的扩展点在两处：

```python
# 1) 先在 Cfg 里加字段：  mtu: int = 4096
# 2) 再加一条检查：接收 Cfg、返回 Result
def check_mtu(c: Cfg) -> Result:
    if c.mtu != 4096:
        return Result(Level.WARN, f"mtu={c.mtu}", "小 MTU 会让大消息拆更多包")
    return Result(Level.OK, "mtu=4096", "")

# 3) 注册进列表，report() 会自动渲染
CHECKS = [check_memlock, ..., check_mtu]
```

每个 `Result` 必须带 `symptom`——这是这份脚本的设计约束：
**没有症状描述的检查项等于没写**，因为它不能帮你从现象反推原因。
