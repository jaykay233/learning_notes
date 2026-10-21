# DeepInfer 的 Zero-Overhead Asynchronous Scheduling

> 根据 DeepInfer 源码仓库 `774cc244b87cc0eb8d4b94ab9f5de2ab22049f01` 的 [部署 QA](https://github.com/deepseek-ai/DeepInfer/blob/774cc244b87cc0eb8d4b94ab9f5de2ab22049f01/doc/inference_deployment_qa.md#L266-L274)、[`EngineArgs`](https://github.com/deepseek-ai/DeepInfer/blob/774cc244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/engine/arg_utils.py#L48-L60) 与 [`Worker.execute_model_step()`](https://github.com/deepseek-ai/DeepInfer/blob/774cc244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/worker/worker.py#L89-L189) 整理。

## 本次讲解位置

```text
章节：models/deepinfer
小节：Asynchronous scheduling (zero-overhead batch scheduler)
知识点：把 CPU schedule / post-process 与 GPU forward / sample 的关键路径重叠
上次：SWA KV 状态分离与 bounded replay
下次：沿源码与 profiler 验证 schedule / forward / sample 的实际重叠
PTX：不适用
```

## 为什么现在讲这个

刚才讨论的 bounded replay 优化的是 prefill 时缺失 SWA 状态的重算量；它解决不了 decode 每步的 CPU 调度空档。Decode 中 CPU 必须根据上一轮结果维护请求状态、安排下一轮 batch，而 GPU 同时需要持续执行 forward 和 sample。若 CPU 等 GPU 结果后才开始 post-process，再开始下一轮 schedule，GPU 会暴露等待。理解 ZOS 的流水线，才能判断它减少的是哪段空档，也避免把它误解为 CUDA Graph、设备端 kernel launch，或真正零成本的 CPU 调度。

## 1. 心智模型：把“等结果再处理”移出 GPU 关键路径

普通同步心智模型是：

```text
CPU: schedule(N) ─────────────── wait sample(N) ─ post-process(N) ─ schedule(N+1)
GPU:              forward(N) ─ sample(N)                          空档
```

问题是 `post-process(N)` 与 `schedule(N+1)` 都是 CPU 工作；如果必须等 `sample(N)` 完成后才开始，它们可能表现为 GPU 空档。

ZOS 让 CPU 尽早提交当前步的 forward，并把 decode 当前步的 post-process 延后到下一轮；GPU 执行当前 forward/sample 时，CPU 可安排下一步。下一轮调用里先提交新 forward，再处理上一步结果：

```text
                    step N                         step N+1
CPU: schedule(N) -> launch forward(N) -> schedule(N+1) -> launch forward(N+1)
                                      post-process(N) ─────┘
GPU:                 forward(N) -> sample(N) -> forward(N+1) -> sample(N+1)
```

更精确地说，`Worker.execute_model_step()` 的顺序是：

1. 先调用 `execute_main_model()`，提交当前步 forward。
2. 再对 `prev_step_state` 调用 `_post_process_step()`；它等待上一步的 `sample_end_evt`，然后做 CPU post-process。
3. 对当前步执行 sample，并记录当前 sample 的完成事件。
4. decode 且 ZOS 开启时，不立即处理当前状态，而是保存为 `prev_step_state`，留给下一次调用处理。

这样，当前 GPU forward 有机会与 CPU 处理上一步输出重叠。实际能重叠多少取决于 GPU 队列、同步依赖、batch、其他 CPU 工作与设备占用；不是每个工作负载都能完全消除空档。

## 2. 源码里的实际执行顺序

在 [`worker.py`](https://github.com/deepseek-ai/DeepInfer/blob/774cc244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/worker/worker.py#L113-L189) 的 `execute_model_step()` 中，顺序可以逐项对应：

1. 对当前 step 调用 `execute_main_model()`，提交 forward。
2. 调用 `_post_process_step(self.prev_step_state)` 处理前一轮状态。
3. 对当前 step 调用 `execute_sample()`，记录 `sample_end_evt`。
4. prompt 或 ZOS 关闭时，当前状态立即 post-process；ZOS decode 则把 `curr_step_state` 保存为 `prev_step_state`。

`_post_process_step()` 在 [`worker.py`](https://github.com/deepseek-ai/DeepInfer/blob/774cc244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/worker/worker.py#L247-L272) 先同步 `sample_end_evt`，再执行模型与 engine 的 CPU post-process。ZOS 没有取消这个依赖，而是把它放到下一轮当前 forward 提交之后，争取让设备工作先排上队。

## 3. `zero-overhead` 的 “zero” 是什么

这里的 zero-overhead 是优化目标/调度器名称，意指尽可能让调度成本被设备执行覆盖，不是 CPU 工作或 launch 成本在物理上等于零。

| 项目 | ZOS 做了什么 | 没有承诺什么 |
|---|---|---|
| CPU schedule | 让下一轮 batch 准备更早进行 | scheduler 不再消耗 CPU 时间 |
| post-process | 延后 decode 上一步处理，争取与下一步 forward 重叠 | 永远不等待 GPU sample 事件 |
| GPU forward/sample | 主机继续按运行时逻辑提交工作 | kernel launch 延迟归零或改为设备端 launch |
| CUDA Graph | 可与原执行路径共同存在 | ZOS 本身就是 CUDA Graph |
| 总体效果 | 减少关键路径上暴露的 host 调度空档 | 所有模型、负载都必然达到完全无空档 |

类比为流水线：零空档不等于装配动作免费，而是尽量在前一件产品加工时完成下一件的准备。

## 4. DeepInfer 里开关是否打开

源码中的开关是：

```text
DEEPINFER_ZERO_OVERHEAD_SCHEDULING=1  # 开
DEEPINFER_ZERO_OVERHEAD_SCHEDULING=0  # 关
```

`EngineArgs.__post_init__()` 使用 `os.environ.setdefault()` 设置默认值，所以它只在变量尚未设置时写入。DeepInfer [README](https://github.com/deepseek-ai/DeepInfer/blob/774cc244b87cc0eb8d4b94ab9f5de2ab22049f01/README.md#L25-L27) 也将异步调度标成默认开启：

| `EngineType` | 默认值 |
|---|---:|
| `NORMAL` | 开 `1` |
| `COMPLETION` | 开 `1` |
| `PROMPT` | 关 `0` |

默认 engine type 是 `NORMAL`；显式在进程启动前设置环境变量，会覆盖上述默认。例如：

```bash
# 关闭 ZOS；在启动推理进程前设置
DEEPINFER_ZERO_OVERHEAD_SCHEDULING=0 python examples/benchmark.py --model_name /path/to/compatible-v41-hf

# 显式开启 ZOS
DEEPINFER_ZERO_OVERHEAD_SCHEDULING=1 python examples/benchmark.py --model_name /path/to/compatible-v41-hf
```

命令需要在满足 DeepInfer 硬件和依赖要求的仓库环境里运行。启动日志包含 `zero_overhead_scheduling=...` 配置字段，可用于检查最终生效值；不能只看 shell 是否执行过 export。

在 DeepInfer 已安装/可导入的环境中，可用下面的完整小脚本核对三个 `EngineType` 的默认策略：

```bash
python - <<'PY'
from deepinfer.engine.engine_type import EngineType

for engine_type in EngineType:
    enabled = engine_type.default_zero_overhead_scheduling
    print(f"{engine_type.name}: {int(enabled)}")
PY
```

预期输出：

```text
NORMAL: 1
PROMPT: 0
COMPLETION: 1
```

该脚本检查的是各类型默认值；要验证环境变量覆盖，还需在创建 `EngineArgs` 之前显式设置变量，并检查启动日志。源码的 `setdefault()` 位置见 [`arg_utils.py`](https://github.com/deepseek-ai/DeepInfer/blob/774cc244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/engine/arg_utils.py#L48-L55)，engine 默认映射见 [`engine_type.py`](https://github.com/deepseek-ai/DeepInfer/blob/774cc244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/engine/engine_type.py#L11-L17)。

## 5. 按 token step 跟踪一次执行

假设 decode 处于稳定阶段，步骤编号从 `N` 开始：

| 时刻 | CPU | GPU | 同步/状态 |
|---|---|---|---|
| A | `schedule(N)`，生成本轮 metadata | 正在执行或等待前一轮工作 | scheduler 产出本轮 batch |
| B | 调用 worker，提交 `forward(N)` | 开始 `forward(N)` | host 发起的执行，不是设备端 launch |
| C | 处理 `N-1` 的输出；等待其 sample event 后更新请求状态 | `forward(N)` 可与 CPU 处理尝试重叠 | 只等待上一轮的事件 |
| D | 提交 `sample(N)` 并保存 `N` 的 step state | 执行 `sample(N)` | 当前 decode 结果先不做完整 CPU post-process |
| E | 回到 engine，开始下一轮 `schedule(N+1)` | `forward/sample(N)` 可能仍在设备队列中 | CPU 读写滞后字段需走 ZOS metadata 修正路径 |
| F | 提交 `forward(N+1)` 后处理 `N` | 执行 `forward(N+1)` | `N` 的 host post-process 与下一轮设备工作争取重叠 |

在深度/置信度调度等路径里，CPU 可能暂时拿不到上一轮设备侧结果；DeepInfer 用 ZOS metadata 路径修正滞后长度等字段。这是流水线化需要付出的状态管理复杂度，不是简单地把一次 CPU 调用删掉。

## 6. 与 CUDA Graph、kernel launch 的边界

| 机制 | 解决的问题 | 提交/调度主体 |
|---|---|---|
| CUDA Graph | 将一组已知 GPU 操作捕获并降低重复的 host 提交开销 | 通常仍由 CPU replay graph |
| DeepInfer ZOS | 让 CPU schedule/post-process 与 GPU forward/sample 在 step 间重叠 | CPU engine/worker 配合异步 GPU 执行 |
| Device-side launch | GPU 设备端发起后续工作 | GPU/device runtime，需专门的设备端 launch 支持 |

因此，开启 ZOS 后仍然存在 host 侧模型执行提交、kernel launch、CUDA stream/event 和必要的同步。它主要隐藏可重叠的 CPU 调度/后处理时间；不等于 CPU 不再向 GPU 下发算子，也不等于“把所有层打包成一个 graph 后设备端自行运行”。

## 7. 如何验证是否真的有收益

1. 先确认同一套模型、请求数、上下文长度、CUDA Graph 配置和采样设置，只改变 `DEEPINFER_ZERO_OVERHEAD_SCHEDULING`。
2. 观察启动日志中的开关值，并确保稳态计时排除冷启动和 profiler teardown。
3. 看 GPU timeline：检查 forward/sample 之间的空档是否缩短、CPU post-process 是否落在 GPU 工作期间。
4. 同时看端到端吞吐和单请求 TPOT；吞吐上升不自动表示 TPOT 也下降。
5. 再读 CPU timer 中的 `schedule`、`worker.post_process.sync`、`worker.post_process.do_process`，确认瓶颈有没有移到别处。

## 常见误解

| 误解 | 正确理解 | 可观察症状 |
|---|---|---|
| “zero-overhead 表示调度成本为零” | 调度成本仍在，只是试图从关键路径上隐藏 | CPU timer 仍能测到 schedule/post-process 时间 |
| “ZOS 是 CUDA Graph” | 一个是 CPU/GPU step 间流水，另一个是 GPU 工作捕获/重放 | 即使关闭 graph，ZOS 概念仍成立；反之亦然 |
| “开 ZOS 后不用 CPU 发 kernel” | worker 仍由 host 执行模型并提交 GPU 工作 | timeline 仍显示 host API 与 kernel launch |
| “把 post-process 延后就不会同步” | 下一轮仍等保存的 sample event，只是改变等待位置以争取重叠 | 负载不合适时仍能看到 event wait 或 GPU bubble |
| “默认开关对所有 EngineType 都相同” | NORMAL/COMPLETION 默认开，PROMPT 默认关，显式 env 可覆盖 | 启动配置与预期不一致时先查 EngineType 和 env |

## 自测题

1. DeepInfer ZOS 主要想藏住哪类开销？
   - 答：CPU batch scheduling 与前一步 post-process 等可能暴露在 GPU 关键路径上的 host 工作。
2. `zero-overhead` 是否意味着 CPU 不再做 schedule？
   - 答：不是；仍执行，只是争取与 GPU 工作重叠。
3. 默认 `NORMAL` engine 下开关值是什么？怎样显式关闭？
   - 答：默认开启；启动进程前设 `DEEPINFER_ZERO_OVERHEAD_SCHEDULING=0`。
4. ZOS 与 CUDA Graph 能否同时存在？
   - 答：可以；它们解决的不是同一个层面的问题。
5. 为什么 ZOS 可能没有吞吐收益？
   - 答：若 CPU 工作很短、GPU 已饱和、依赖导致无法并行，或瓶颈在 HBM/计算/同步等其他位置，可隐藏的 host 空档有限。

## 本节小结

```text
传统路径：sample 完 -> host post-process -> 再安排下一轮
ZOS 路径：下一轮 forward 先提交 -> 处理上一轮结果 -> 下一轮 sample
目标：把 CPU 工作移到 GPU 工作期间
边界：不是零 CPU 成本，不是 CUDA Graph，不是 device-side launch
```
