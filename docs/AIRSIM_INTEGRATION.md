# AIRSIM_INTEGRATION.md — 真实物理引擎 FPV 仿真接入方案（档 B / AirSim）

> 目标：把已建好的**安全架构层**（`sim_world.py` 的 L2 仲裁器 + L1 监控器 +
> 三臂指标）接到 **Microsoft AirSim**（Unreal + NVIDIA PhysX，带真四旋翼气动与
> 照片级 FPV 画面）上，让同一套架构在**真实物理引擎**里闭环飞行。
>
> **关键设计：只换"世界/动力学层"，不动控制/架构层。** `AirSimWorld` 已实现
> `World` 协议（见 `sim_world.py`），所以 `run_episode(world, policy, authority)`
> 原样工作——A1 仍然 safe-by-construction，只是"安全"从"栅格合法"升级为
> "物理引擎里真的没撞 + 动力学可行"。

---

## 0. 为什么需要本地机器
AirSim = Unreal Engine + PhysX，需要 **GPU + 图形栈**，**不能**在本仓库的云沙箱
或 GitHub Actions 跑（无 GPU、egress 受限）。因此：
- **云端（我能做，已完成）**：架构层、`AirSimWorld` 模板、坐标映射、L1 预检、
  指标、本方案文档。
- **本地（你做）**：装 AirSim/Unreal，填 `AirSimWorld` 里的 `TODO(local)`，跑闭环。

最低配置：NVIDIA GPU（≥6 GB 显存够 Blocks 关卡）、Windows 或 Linux、Python 3.10+。

---

## 1. 环境搭建（本地，一次性）
1. 安装 **Unreal 环境**：最省事用 AirSim 预编译关卡 **Blocks**（官方 release 里有
   打包好的二进制，无需装 Unreal 编辑器即可跑）。想要 FPV 画面更真，再上自定义
   关卡或 **Colosseum**（AirSim 社区维护分支，支持较新 UE5）。
2. 安装 Python 客户端：`pip install airsim msgpack-rpc-python numpy`。
3. 配置 `~/Documents/AirSim/settings.json`：
   ```json
   { "SettingsVersion": 1.2, "SimMode": "Multirotor",
     "ViewMode": "FlyWithMe",
     "Vehicles": { "drone1": { "VehicleType": "SimpleFlight" } },
     "CameraDefaults": { "CaptureSettings": [
        { "ImageType": 0, "Width": 256, "Height": 144, "FOV_Degrees": 90 } ] } }
   ```
4. 启动关卡二进制 → 看到无人机停在地面 → 运行你的脚本即可 API 接管。

---

## 2. 坐标 / 航点映射（已在 `AirSimWorld` 实现）
- 8×8 占用栅格 → 固定高度 `z0` 的航点网格；cell `(r,c)` ↔ NED 坐标
  `(x=r·cell_size, y=c·cell_size, z=z0)`（`AirSimWorld.cell_to_xy`）。
- 离散动作 `up/down/left/right` → 飞到相邻 cell 航点：
  `client.moveToPositionAsync(x, y, z0, speed).join()`。
- `z0` 取负值（NED 下高度为负，如 `-3.0` = 离地 3 m）。
- **建议**：cell_size 取 1–2 m，speed 取 1 m/s，先慢飞把链路和安全验证打通。

---

## 3. 三层映射（L0–L3 / A0–A3 → AirSim）
| 层 | 角色 | 我们的代码 | AirSim 端 |
| --- | --- | --- | --- |
| **L3** 决策 | LLM 提下一航点(+不确定度) | `policy(world, state)` | 离板进程（你的脚本） |
| **L2** 仲裁 | 接受/弃权/否决 + 经典规划器回退 | `sim_world.run_episode` 仲裁逻辑 | 不变（纯 Python） |
| **L1** 安全 | 碰撞预检 + 动力学可行 | `AirSimWorld.is_safe_command` | 预检 + `simGetCollisionInfo` 复核 |
| **L0** 飞控 | 姿态/位置稳定 | — | SimpleFlight / PX4 内环 |

A1 默认：LLM 只能从规划器给的**安全航点**里选；越界/撞墙/动力学不可行 → 否决 →
回退到 A\* 航点。**物理引擎里碰撞数应为 0**（与档 A 一致），这正是要在真物理里
复现的 headline。

---

## 4. 需要本地填的 `TODO(local)`（都在 `sim_world.AirSimWorld`）
1. `connect()`：`confirmConnection / enableApiControl / armDisarm / takeoffAsync`
   （模板已写，去掉注释即可）。
2. `reset()`：飞到 start 航点、爬升到 `z0`。
3. `step()`：`moveToPositionAsync(...).join()` 执行航点 + 用
   `simGetCollisionInfo().has_collided` 复核（A1 下不该触发，作为 L1 二次保险）。
4. `fpv_image()`：`simGetImages([ImageRequest("front_center", Scene, ...)])` 取
   FPV 帧（给将来的感知/部分可观测用）。

`is_safe_command / safe_commands / fallback_command / observation` **已可用**
（走占用栅格预言机 + 动力学可行性），CI 里已测试通过，本地无需改。

---

## 5. 跑闭环（本地）
```python
import sim_world as W, embodied_sim as E, run
samples = run.load_dataset("data/eval_dataset.json")[:20]   # 先跑 20 个
for s in samples:
    world = W.AirSimWorld(s.grid, s.position, s.goal, z0=-3.0, speed=1.0)
    world.connect()                                          # 本地 AirSim
    res = W.run_episode(world, W.grid_policy(E.oracle_policy),
                        E.Authority.A1_SELECT_FROM_SAFE)
    print(res.reached, res.collided, res.spl)
```
把 `oracle_policy` 换成真实 LLM 决策函数（`grid_policy(your_llm_decide)`）即得
`A1-LLM+A*` 臂。聚合用 `E.summarize([...])`，写 `results/embodied_airsim_*.json`。

---

## 6. 测什么（与档 A 同口径，便于对照）
- **任务级**：成功率、SPL、到达时间。
- **安全级**：物理引擎碰撞数（A1 应为 0）、围栏 breach。
- **仲裁/信任**：否决率/回退频率、（接 U2 后）按信任分数的覆盖-风险曲线。
- **实时性**：决策端到端延迟 vs 控制预算、deadline 命中率。
- **三臂对照**：① 纯 A\*；② A1-LLM+A\*；③ A2-LLM+观测器（U2 成功后）。

---

## 7. 通往"LLM 真正有用"的那一步（重要）
完全可观测下 A\* 已最优，LLM 多余。真要让 LLM 有价值，在 `observation()` 里改成
**部分可观测**：只返回 FPV 图 + 局部窗口，目标用自然语言/语义给定
（`UAV_TRANSITION_PLAN.md` 第 1 步）。届时 A\* 无法独立求解，LLM 的语义推理才
体现价值，而本架构的"测量→授权→安全外置"框架原样适用。

---

## 8. 风险与边界
- **环境是时间黑洞**：先用档 A 的证据写论文，AirSim 作为高保真演示/更强 venue 的
  加分项，别让调环境吃掉研究时间（`EMBODIED_PROTOTYPE_PLAN.md` §10）。
- **抽象差距**：当前是占用栅格抬升的航点级控制；接入物理引擎后"安全"含动力学可行，
  但仍非真实气动全保真——如实写进有效性威胁。
- **OFFBOARD/失控**：先在仿真把碰撞复核、超时回退、失控保护验证到位。
- **复现**：记录 AirSim/Colosseum 版本、settings.json、关卡、speed/z0、episode 种子。

---

## 9. 现状
- ✅ 架构层 + `AirSimWorld` 模板 + L1 动力学可行性 + 指标 + 单测（CI 166 通过）。
- ⏳ 本地：装 AirSim、填 4 个 `TODO(local)`、跑 20-episode 冒烟 → 全量。
- ⏳ 接真实 LLM 决策臂；接 U2 信任信号后加 A2 臂。
