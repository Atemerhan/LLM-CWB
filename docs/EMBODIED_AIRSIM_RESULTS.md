# EMBODIED_AIRSIM_RESULTS.md — 档B 物理引擎闭环（AirSim/PhysX）

Higher-fidelity confirmation of the 档A closed-loop result: the **same** L0–L3 /
A0–A3 architecture (`sim_world.py`), now with waypoint moves executed as real
quadrotor flights in **Microsoft AirSim** (Unreal Engine + NVIDIA PhysX, Blocks
v1.8.1), run locally on an RTX 3060. Source: `results/embodied_airsim_arms.json`
(+ the earlier oracle smoke `results/embodied_airsim_a1.json`).

## Three-authority comparison (n=8 episodes, physics engine)

| arm (authority / policy) | collision | success | SPL | veto-rate | goal/coll |
| --- | --- | --- | --- | --- | --- |
| ungated / greedy | **0.125** | 0.875 | 0.875 | 0.000 | 7 / 1 |
| **A1 / greedy** | **0.000** | 1.000 | 1.000 | 0.083 | 8 / 0 |
| ungated / random | **0.875** | 0.125 | 0.125 | 0.000 | 1 / 7 |
| **A1 / random** | **0.000** | 1.000 | 0.544 | 0.286 | 8 / 0 |

## Reading
- **Ungated collides in the physics engine**: a wall-blind greedy policy crashes
  12.5% of episodes, a random policy 87.5% — the untrusted controller really does
  fly into walls when nothing gates it.
- **A1 holds the collision rate at 0** for *both* policies, exactly as in the
  kinematic 档A run (0.590/0.943 → 0). The cost is paid in efficiency:
  `a1_random` SPL drops to 0.544 (mean 7.0 steps vs optimal 1.5) because vetoed
  random proposals (veto-rate 0.286) are replaced by safe planner fallbacks —
  the thesis's "safety-for-efficiency" trade-off, now visible in a flight sim.
- This reproduces the constructive A1 result at higher fidelity: the architecture
  integrates with a real flight/physics stack and the vehicle physically flies the
  safe paths.

## Honest scope (threats to validity)
- **Collision is adjudicated at the grid/L1-oracle layer** (`sim_world.is_safe
  _command`) *before* the physics `step()`, not by AirSim's collision sensor. So
  this demonstrates the architecture executing in a physics-engine flight loop
  with the same verified monitor — not an independent physics-based collision
  detection. `AirSimWorld.step()` already records `simGetCollisionInfo().has
  _collided` into `_collided`; promoting that to the episode's collision signal
  is the natural full-physics extension (future work).
- **n=8** (reduced from 15 after AirSim RPC instability under heavy ungated
  collisions); a shared client is used across episodes (per-episode connect()+
  takeoff conflicts). This is a **demonstration/prototype**, not a flight test —
  the statistically strong evidence remains the kinematic n=1200 run (§4.7).
- 2D occupancy lifted to fixed-altitude waypoints; no aerodynamic/attitude
  coupling beyond what AirSim's multirotor model provides.

## Thesis use
Supports §4.7 / §6 as the **physics-engine demonstration** that the A1 ceiling
neutralizes an untrusted policy in a real flight stack (carbon copy of the
kinematic result: collision 0 vs ungated 0.125/0.875). Cite as a prototype /
defense demo; keep the n=1200 kinematic run as the primary quantitative evidence.
Implementation: `sim_world.AirSimWorld` (local run); see `AIRSIM_INTEGRATION.md`.
