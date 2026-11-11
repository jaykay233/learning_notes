# GPUDirect 的两条权限：退化成 bounce 时发生了什么

```text
本次讲解位置
章节：communication / RDMA
小节：L5 GPUDirect 退化判定（33 章只写了"ACS/IOMMU/注册/驱动"，Q40/Q41）
知识点：两条独立的 P2P 权限、四条路径、退化判定方法、带宽与延迟代价
上次：排障判定树（36）——"带宽正常但不对"的第二种形态
下次：38 · verbs 程序走读（把"路径"落到可编译的调用顺序上）```

## 为什么现在讲这个

[33 章](./33-rdma-six-layer-spine.md) 对 Q41 的答案是四个条件：拓扑支持 P2P、ACS 关闭、
显存已注册、驱动版本匹配。这四句都对，但合起来**不能回答现场问题**：

> 我这套机器，到底在不在 GDR 路径上？

更麻烦的是退化方式：**它不报错。** 注册照样成功、`ibv_reg_mr` 返回 0、
`ib_write_bw` 照样跑完，只是带宽掉一半、延迟多几微秒。
所以团队经常在"以为在用 GDR"的状态下跑了几个月。

这一章要把三件事钉死：GDR 其实需要**两条独立的权限**、退化后**具体多付多少**、
以及**怎么在五分钟内测出自己在哪条路径上**。

---

## 心智模型：P2P 是两条权限，不是一个开关

这是最容易搞混、也最能体现理解深度的地方。回顾 [21 章](./21-gpu-initiated-rdma-submission-dc-qp-sharing.md)
讲的 UAR 映射：NIC 能 DMA 读写 GPU 内存，和 GPU 能写 NIC 的 BAR，是**两个方向**的事。

```text
方向 A：NIC → GPU 显存
        网卡通过 PCIe P2P 直接 DMA 写显存
        需要：P2P 通路 + 显存注册成 peer memory（nvidia-peermem）
        用途：数据面（WRITE 落地、READ 源）

方向 B：GPU → NIC BAR
        GPU 线程 MMIO 写 UAR doorbell
        需要：把 NIC BAR 映射进 GPU 地址空间（PeerMappingOverride=1）
        用途：GPU-initiated 提交（21 章第 7 节）
```

两个方向可以独立失败，于是有四种组合：

| A: NIC→GPU | B: GPU→NIC | 结果 | 谁在提交 |
| --- | --- | --- | --- |
| 通 | 通 | 完整 GPU-initiated GDR | SM 直接写 doorbell |
| 通 | 不通 | **数据面 GDR，控制面后退** | CPU-assisted doorbell |
| 不通 | 通 | 数据仍走 bounce | SM 提交但数据绕主机 |
| 不通 | 不通 | 完全退化 | CPU proxy + bounce |

**关键结论：`ib_write_bw --use_cuda` 跑得快，只证明方向 A 通。**
它完全不证明方向 B——那要靠 `cuMemHostRegister` 能不能注册 BAR page 来验证。
两件事混谈，就会出现"带宽很好但 issue 延迟很奇怪"的矛盾现象。

---

## 四条路径与判定规则

[`37_gpudirect_path.py`](./37_gpudirect_path.py) 把判定写成规则：

```python
def p2p_ok(self):
    if not self.p2p_supported:
        return False, "平台/驱动不支持 P2P"
    if self.nvlink:
        return True, "NVLink 直连网卡，绕开 PCIe ACS"
    if self.acs:
        return False, "ACS 开启，P2P 被 IOMMU 拦在 switch 边界"
    if not self.same_switch:
        return False, "跨 PCIe switch 且未做 P2P 定向"
    return True, "同一 PCIe switch，ACS 关闭"
```

四条判定依据，按优先级：

| 条件 | 后果 | 为什么 |
| --- | --- | --- |
| 平台/驱动不支持 P2P | 一定退化 | 内核没编 `CONFIG_PCI_P2PDMA` 或厂商驱动没开 |
| NVLink 直连网卡 | 通 | 绕开 PCIe 与 ACS，物理上就不经过 switch |
| ACS 开启 | 不通 | IOMMU 要在 switch 边界做隔离，P2P 被拦 |
| 跨 switch 且无 P2P 定向 | 不通 | 上行流量默认回到 root complex |

ACS 这条值得单独说：**它的默认配置经常是"开启"**，因为服务器 BIOS 优先考虑隔离性。
所以"买了同一 switch 下的 GPU 和网卡"并不等于 P2P 能通——还要看 ACS 和
（部分平台的）`pcie_acs_override`。

## 退化后多付多少

脚本的账本（Gen4 x16，200 Gbps 网卡）：

```text
                        拓扑   P2P              路径     有效 Gbps    延迟 us
 GPU/NIC 同 switch, ACS off     通       GDR (P2P)        200      0.0
  GPU/NIC 同 switch, ACS on    不通   bounce buffer        100      3.0
 GPU/NIC 跨 switch, ACS off    不通   bounce buffer        100      3.0
               NVLink 直连网卡     通      NVLink→NIC        400      0.0
                 驱动不支持 P2P    不通   bounce buffer        100      3.0
```

**带宽正好掉一半**，机制很直白：数据要过两段 PCIe（NIC→host 内存、host 内存→显存），
而每一段都要吃满同一条链路的带宽。

延迟多约 **3 us**——和 [33 章](./33-rdma-six-layer-spine.md) 里
"一条消息的固定成本 1.5 us" 是同一量级，**所以小消息场景里退化是延迟问题，不是带宽问题**：

```text
1 KB 消息，固定成本 1.5 us：
  GDR    : 1.5 us + 82 ns 线上  ≈ 1.58 us
  bounce : 1.5 us + 3 us + 82 ns ≈ 4.58 us   ← 2.9 倍
```

而 1 GB 的 AllReduce 里，退化体现为纯带宽：

```text
传 1024 MB：
  GDR       20.5 ms
  bounce    41.0 ms      (同 switch，ACS on)
  bounce    81.9 ms      (跨 switch，两段都吃满)

Ring AllReduce，1024 MB × 8 rank：
  每 rank 发送 2×(N-1)/N × size = 1.75 × 1024 MB = 1792 MB
  GDR 400 Gbps  : 35.8 ms
  bounce 200 Gbps: 71.7 ms      ← 退化损失 35.8 ms
```

`1.75` 这个系数值得记住：ring AllReduce 每个 rank 要发 `2(N-1)/N` 倍数据量，
N 越大越接近 2。**所以通信时间是数据量的 1.75~2 倍除以带宽**——
这也是"带宽退化 2x 直接让训练 step time 翻倍"的原因。

---

## 五分钟判定：怎么确认自己在哪条路径上

按顺序做，每步只看一个信号：

```bash
# 1. 拓扑：GPU 和网卡在哪个 PCIe 层级
nvidia-smi topo -m
#   看 GPU 与 mlx5 那一格：PIX/PXB = 同 switch；SYS = 跨 NUMA / 跨 socket

# 2. ACS 是否开启（关键）
sudo lspci -vvv -s <上游 bridge BDF> | grep -i acs
#   AcsCtl: Enabled / Disabled

# 3. 显存是否注册给网卡
lsmod | grep -E 'nvidia_peermem|nv_peer_mem'
dmesg | grep -i peer_memory

# 4. 数据面是否真在 GDR：GPU buffer 与 host buffer 的带宽对比
ib_write_bw -d mlx5_0 --use_cuda=0 -s 1048576    # host 内存
ib_write_bw -d mlx5_0 --use_cuda=1 -s 1048576    # 显存
#   两者接近 -> 在 GDR 路径；后者明显低 -> 已退化
```

判读规则：

| 观察 | 结论 |
| --- | --- |
| `nvidia-smi topo` 显示 SYS，且带宽只有一半 | 跨 socket，基本确定 bounce |
| ACS Enabled，且 `--use_cuda` 带宽减半 | ACS 拦掉了 P2P |
| `nvidia_peermem` 没加载 | 显存没注册，一定 bounce |
| GPU/host 带宽接近但 issue 延迟高 | 方向 A 通、方向 B 不通（CPU-assisted doorbell） |
| 两者都正常但 p99 抖 | 不是 GDR 问题，去看 [36 章](./36-rdma-troubleshooting-decision-tree.md) 的 pause 占比 |

最后一行是本章和 36 章的接口：**GDR 退化是"稳态变慢"，PFC 反压是"抖动"**，
两者的监控形态完全不同，别互相甩锅。

---

## 常见错法，以及症状

| 错法 | 症状 |
| --- | --- |
| 用 `--use_cuda` 带宽证明 GPU-initiated 可用 | 数据面 GDR 正常，但 doorbell 仍走 CPU，issue 延迟高 |
| 以为"同 switch"就够了 | 带宽稳定减半，CPU 占用升高（memcpy 线程跑满） |
| 只查驱动版本，不查 `nvidia_peermem` | 注册成功但走 bounce，`dmesg` 里有 peer memory 报错被忽略 |
| 退化后去调 `max_inline_data` / QP 数量 | 参数越调越乱，因为瓶颈在路径不在参数 |
| 打开 ACS override 却没验证 | 参数生效了但内核策略没生效，仍走 bounce |
| 把 bounce 的 3 us 当成网络延迟 | 去查交换机和 PFC，查不到任何异常 |

第三条最隐蔽：`ibv_reg_mr` 在 GPU 指针上**也能成功**（走的是注册而不是 P2P），
所以"注册成功"完全不能作为 GDR 生效的证据。必须看 `nvidia_peermem` 与实测带宽。

## 模型边界

- **带宽按"两段 PCIe 各吃满"估**：真实值取决于 DMA 引擎并发与 TLP 效率，可能好于 50%
- **没有建模 ACS 的部分放行**：某些平台可以只允许同 switch 内的 P2P，此时同 switch 仍通
- **延迟 3 us 是量级**：实际取决于 memcpy 调度与 NUMA 位置
- **没有建模单边 READ 的退化**：READ 退化路径与 WRITE 不同（源在远端）

所以这张表只能用来判断"量级差多少、该不该查"，具体数值必须实测。

---

## 自测

1. GDR 需要哪两条独立权限？各自由什么决定？
2. `ib_write_bw --use_cuda=1` 带宽和 host 内存一样，能否说明 GPU 可以直接写 doorbell？
3. ACS 为什么会让 P2P 失效？它的默认状态通常是什么？
4. 1 KB 消息 + bounce，延迟大概变多少倍？为什么大消息里退化体现为带宽？
5. ring AllReduce 的 1.75 系数怎么来的？8 rank、1 GB 时每 rank 发多少？
6. `ibv_reg_mr` 在 GPU 指针上成功，能证明 GDR 生效吗？

<details>
<summary>答案</summary>

1. ① **NIC→GPU 显存**（P2P 通路 + `nvidia_peermem` 注册），用于数据面；
   ② **GPU→NIC BAR**（`cuMemHostRegister` + `PeerMappingOverride=1`），用于 SM 写 doorbell。
   两条可以独立失败。
2. **不能。** 那只证明方向 A 通。方向 B 要看 BAR 能不能映射进 GPU 地址空间；
   不通时数据面仍是 GDR，但提交退化成 CPU-assisted doorbell，表现为 issue 延迟高。
3. ACS 是 IOMMU 在 PCIe switch 边界做隔离的机制，P2P 流量要穿过边界就被拦。
   **默认通常是开启**（BIOS 优先隔离性），所以"同 switch"不等于 P2P 可用。
4. 固定成本 1.5 us → 4.58 us，约 **2.9 倍**。因为 3 us 的 bounce 延迟是每消息固定成本；
   消息大时它被摊薄，退化就只剩"带宽减半"这一项。
5. 每个 rank 需发送和接收各 `(N-1)/N` 份数据，合计 `2(N-1)/N`。
   N=8 时 = 1.75，1 GB × 1.75 = **1792 MB**。
6. **不能。** 注册成功只说明内核接受了这个地址，不代表走了 P2P。
   必须看 `nvidia_peermem` 加载状态 + 实测 GPU/host 带宽对比。

</details>

---

## 一页速记

```text
两条权限（可独立失败）:
  A NIC -> GPU 显存    P2P 通路 + nvidia_peermem     数据面
  B GPU -> NIC BAR     BAR 映射 + PeerMappingOverride 提交面

四种组合:
  A通B通  完整 GPU-initiated GDR
  A通B不通 数据 GDR，doorbell 走 CPU（issue 变慢）
  A不通B通 提交快，数据 bounce
  都不通   CPU proxy + bounce

判定规则:
  驱动不支持 P2P -> 必退化
  NVLink 直连网卡 -> 通（绕开 ACS）
  ACS 开启       -> 不通（默认常开）
  跨 switch 无定向 -> 不通

代价:
  带宽减半（两段 PCIe 各吃满）
  延迟 +3 us/消息  -> 1 KB 消息变 2.9 倍
  ring AllReduce 每 rank 发 2(N-1)/N 倍，N=8 时 1.75 倍

五分钟测法:
  nvidia-smi topo -m  ->  PIX/PXB 还是 SYS
  lspci -vvv | grep acs
  lsmod | grep peermem
  ib_write_bw --use_cuda=0/1 对比带宽

一句话:
  注册成功 != GDR 生效；--use_cuda 带宽快 != GPU 能写 doorbell
```
