# DeepEP：`elect_one_sync` + `__shfl_sync(..., 0)` 正确性陷阱

> 承接 [11](./11-deepep-v1-intranode-combine-and-warp-roles.md) §5。读 DeepEP / 自写 warp 协作代码时，常把「选中一个 thread」和「从 lane 0 广播」当成同一件事——**不是**。本文专门钉死这个坑：算不算 bug、和合法「其它 lane 写 0」怎么区分、怎么写才稳。
>
> 代码：`csrc/kernels/legacy/utils.cuh`（V1）、`deep_ep/include/deep_ep/common/ptx.cuh`（V2）；对照合法用法见 `csrc/kernels/legacy/internode_ll.cu`。

```text
本次讲解位置
章节：communication / DeepEP
小节：warp 选举与广播契约
知识点：elect.sync 不保证 lane 0；shfl 源 lane 必须与写值者一致
上次：LL recv hook / 「0 SM」（18）
下次：V2 ElasticBuffer / hybrid dispatch，或 SourceMeta 位图
PTX：elect.sync；shfl.sync
```

## 为什么现在讲这个

走读 DeepEP 时会反复看到两种「只让一个 lane 干活」的写法：

1. `if (elect_one_sync()) { … }` —— 发 TMA、init mbarrier、打 printf
2. `lane_id == 0 ? atomic… : 0;` 再 `__shfl_sync(..., 0)` —— 抢槽后广播 `slot_idx`

若把 (1) 的结果再用 (2) 的方式广播（`shfl` 固定从 lane 0 取），在 SM90 的 `elect.sync` 语义下是**正确性错误**：选中者未必是 lane 0，lane 0 寄存器里往往还是 **0 / 未写过的值**，全 warp 就会拿到错的 0。

这不是「写了 0 就不对」，而是 **写值的 lane ≠ shuffle 的源 lane**。

---

## 1. 心智模型

```text
elect.sync(mask)
  契约：mask 内恰好一个 thread 的 predicate = true
  不保证：被选中的一定是 lane 0（规范只谈确定性选举，不谈「最低编号」）

__shfl_sync(mask, var, srcLane)
  契约：每个参加的 thread 读到的是「srcLane 那一格上的 var」
  不关心：谁当初「算」出了这个值；只读源 lane 寄存器里现在是什么
```

因此：

| 步骤 | 正确配对 | 危险配对 |
|---|---|---|
| 谁写值 | 固定 `lane_id == 0`，或记下 `elected_lane` | `if (elect_one_sync()) x = …` |
| 怎么广播 | `shfl(..., 0)` 或 `shfl(..., elected_lane)` | 仍 `shfl(..., 0)` |

副作用（TMA store、mbarrier init）可以只 `elect` 一次、**不 shfl 取值**——那种用法本身没问题。

---

## 2. DeepEP 里的 `elect_one_sync`

V1（`utils.cuh`）：

```cpp
__device__ __forceinline__ uint32_t elect_one_sync() {
#ifndef DISABLE_SM90_FEATURES
    uint32_t pred = 0;
    asm volatile(
        "{\n"
        ".reg .b32 %%rx;\n"
        ".reg .pred %%px;\n"
        "      elect.sync %%rx|%%px, %1;\n"
        "@%%px mov.s32 %0, 1;\n"
        "}\n"
        : "+r"(pred)
        : "r"(0xffffffff));
    return pred;
#else
    return get_lane_id() == 0;   // 非 SM90：显式 lane 0，与 shfl(...,0) 对齐
#endif
}
```

V2（`ptx.cuh`）同构：SM90 走 `elect.sync`，否则 `get_lane_idx() == 0`。

要点：

- **非 SM90 / `DISABLE_SM90_FEATURES`**：恒等价于 lane 0 → 再 `shfl(..., 0)` 安全。
- **SM90**：实测常常选出 lane 0，但 **PTX 不保证**；依赖「碰巧是 0」= 脆弱正确性。

---

## 3. 危险写法（bug 模式）

```cpp
int x = 0;                         // 或未初始化
if (elect_one_sync())
    x = /* 真实结果，例如 atomic / 计数 */;
x = __shfl_sync(0xffffffff, x, 0); // 假定 lane 0 有值
```

| 选举结果 | lane 0 上的 `x` | 全 warp 看到的值 |
|---|---|---|
| 选中 lane 0（常见） | 真值 | 碰巧对 |
| 选中 lane `k≠0` | 仍是 **0** | **全员广播 0** → 错槽 / 错地址 / 错计数 |

症状特点：偶发、和架构/驱动/编译选项相关；在「大家测 SM90 且常选 0」的环境里很容易被掩盖。

**算不算 bug？**

- 对这种**配对本身**：是正确性 bug（规范不允许你假设 elect==lane0）。
- 对**当前 DeepEP 树**：大量 `elect_one_sync` 用于 TMA / mbarrier / signal / timeout printf，只做副作用、不随后 `shfl(..., 0)` 取回那个值；扫描时未必能找到一处「正在炸」的活 call site。更准确说：这是**已确认的坑 / 潜伏风险**；一旦有人复制「elect 写值 + shfl 0」就会成真 bug。

---

## 4. 合法对照：故意写 0，但源就是 lane 0

LL dispatch（`internode_ll.cu`）抢槽：

```cpp
int slot_idx = lane_id == 0 ? atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1) : 0;
slot_idx = __shfl_sync(0xffffffff, slot_idx, 0);
```

| lane | 写什么 | shuffle 源 |
|---|---|---|
| 0 | `atomicAdd` 真值 | **就是这里** |
| 其它 | 占位 `0` | 不参与「提供结果」 |

其它 lane 的 0 **不会**被广播出去，因为 `srcLane` 固定为 0，而 0 号线程写的是 atomic 结果。
这和「elect 写真值、lane 0 仍是 0」完全相反。

---

## 5. 怎么写才稳

1. **要广播标量**：固定 `lane_id == 0` 计算，再 `shfl(..., 0)`（LL 模式）；或
2. **`elect` 之后**：用 ballot / 返回的 elected lane id，再 `shfl(..., elected_lane)`；或把结果先写入 smem/`__shared__` 再 `__syncwarp` 读。
3. **只做副作用**（TMA、mbarrier、单次 printf）：单独 `elect_one_sync()` 即可，**不要**再 `shfl` 那个临时寄存器。
4. Code review 关键字：搜 `elect_one_sync` 后几行是否出现 `__shfl_sync(..., 0)` 且中间变量只在 `if (elect…)` 里赋值。

---

## 6. 完整可运行验证（CPU 模拟 warp）

把 32 lane 寄存器与「选举 + shfl from 0」缩成纯 Python，不依赖 GPU。预期：当选举非 0 时，危险配对得到全 0；合法 `lane0` 配对始终得到真值。

```python
#!/usr/bin/env python3
"""Simulate elect+shfl lane0 hazard vs lane0-writer pattern."""

from __future__ import annotations


WARP = 32


def shfl_sync(regs: list[int], src_lane: int) -> list[int]:
    src = regs[src_lane]
    return [src for _ in regs]


def dangerous_elect_then_shfl0(true_value: int, elected_lane: int) -> list[int]:
    """if (elect) x = true; x = shfl(x, 0)"""
    regs = [0] * WARP
    regs[elected_lane] = true_value
    return shfl_sync(regs, 0)


def safe_lane0_then_shfl0(true_value: int) -> list[int]:
    """lane0 ? true : 0; shfl(..., 0)"""
    regs = [0] * WARP
    regs[0] = true_value
    return shfl_sync(regs, 0)


def safe_elect_then_shfl_elected(true_value: int, elected_lane: int) -> list[int]:
    """if (elect) x = true; x = shfl(x, elected)"""
    regs = [0] * WARP
    regs[elected_lane] = true_value
    return shfl_sync(regs, elected_lane)


def main() -> None:
    true_value = 42

    # Case A: elect happens to be lane 0 — dangerous pattern looks OK
    a = dangerous_elect_then_shfl0(true_value, elected_lane=0)
    assert a == [42] * WARP, a

    # Case B: elect is lane 7 — dangerous pattern broadcasts 0
    b = dangerous_elect_then_shfl0(true_value, elected_lane=7)
    assert b == [0] * WARP, b
    print(f"dangerous elect=7 -> broadcast {b[0]} (BUG)")

    # Case C: LL-style lane0 writer — always OK
    c = safe_lane0_then_shfl0(true_value)
    assert c == [42] * WARP, c
    print(f"safe lane0 writer -> broadcast {c[0]}")

    # Case D: elect + shfl(elected) — always OK
    d = safe_elect_then_shfl_elected(true_value, elected_lane=7)
    assert d == [42] * WARP, d
    print(f"safe elect=7 shfl(7) -> broadcast {d[0]}")

    print("all checks passed")


if __name__ == "__main__":
    main()
```

运行与期望输出：

```bash
python3 communication/19_elect_shfl_sim.py   # 或把上文另存为该文件后执行
```

```text
dangerous elect=7 -> broadcast 0 (BUG)
safe lane0 writer -> broadcast 42
safe elect=7 shfl(7) -> broadcast 42
all checks passed
```

数值走读（Case B）：

| lane | elect 后 `regs[i]` | `shfl(..., 0)` 读到 |
|---|---|---|
| 0 | `0`（没人写） | `0` ← 全员拿到这个 |
| 7 | `42`（被选中写入） | `0` |
| 其它 | `0` | `0` |

---

## 7. 常见误判与症状差

| 误判 | 实际 | 症状/对照 |
|---|---|---|
| 「其它 thread 写了 0 就是 bug」 | LL 抢槽就是故意写 0 | 看 `srcLane` 是不是写真值的那个 |
| 「SM90 上 elect 总是 0，所以没事」 | 实务常见 ≠ 规范保证 | 偶发错槽、难复现 |
| 「elect 都危险」 | 只做副作用、不 shfl 取值时安全 | TMA/mbarrier 路径正常 |
| 「和 `__syncwarp` 搞混」 | syncwarp 不传标量；shfl 才传 | sync 后读 smem ≠ shfl from 0 |

---

## 8. 自测题与答案

1. **`elect.sync` 保证选中 lane 0 吗？**
   答：不保证；只保证 mask 内恰好选中一个。

2. **危险配对的一句话定义？**
   答：只有 `elect` 选中的 lane 写了真值，却用 `__shfl_sync(..., 0)` 从 lane 0 广播。

3. **LL `slot_idx` 为何「其它 lane 写 0」却安全？**
   答：真值写在 lane 0，`shfl` 源也是 0；其它 lane 的 0 不会被当作结果。

4. **当前 DeepEP 里大量 `elect_one_sync` 为何多数没事？**
   答：用于 TMA/mbarrier 等副作用，不随后从 lane 0 shfl 那个临时值。

5. **两种稳妥修法？**
   答：固定 lane 0 写再 `shfl(..., 0)`；或 `shfl(..., elected_lane)` / 经 smem + syncwarp。

---

## 9. 学习进度

- [x] 机内 combine 笔记里点到的 elect/shfl 风险（11 §5）
- [x] 专门记录：契约、危险/合法对照、CPU 模拟、review 清单
- [ ] V2 ElasticBuffer / hybrid（可选下一课）

### 下一知识点

DeepEP V2：`ElasticBuffer`、scaleup/scaleout、`dispatch.cuh` vs `hybrid_dispatch.cuh`；或回头补 `SourceMeta` 位图。

### 相关链接

- 短注出处：[11 §5](./11-deepep-v1-intranode-combine-and-warp-roles.md)
- V1 实现：`csrc/kernels/legacy/utils.cuh` → `elect_one_sync`
- V2 实现：`deep_ep/include/deep_ep/common/ptx.cuh` → `elect_one_sync`
- 合法抢槽：`csrc/kernels/legacy/internode_ll.cu`（`lane_id == 0 ? atomicAdd : 0` + `shfl(..., 0)`）
