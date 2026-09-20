"""Pluggable World layer — decouple the dynamics/collision backend from the
L2 arbiter, L1 monitor, and metrics so a real physics engine (AirSim / PX4+
Gazebo) drops in under the *same* safety architecture as the lightweight grid.

The control/architecture layer is backend-agnostic: `run_episode(world, policy,
authority)` here mirrors `embodied_sim.run_episode` but talks only to the
`World` protocol. Two backends are provided:

  - `GridWorld`  — the frozen-grid kinematic backend (reuses `embodied_sim` /
    the BFS oracle), now with an explicit **kinematic-feasibility** check so a
    "safe command" means collision-free *and* dynamically admissible (a turn
    within limits), not merely grid-legal (UAV_TRANSITION_PLAN §4.4). With the
    default permissive limits it reduces exactly to `embodied_sim`.
  - `AirSimWorld` — a fill-in template for Microsoft AirSim (Unreal + PhysX),
    runnable **only on a local machine** with AirSim + an Unreal environment
    (`import airsim` is lazy, so this module still imports in CI). See
    `docs/AIRSIM_INTEGRATION.md`.

Policies stay injectable; grid policies are adapted via `grid_policy`. The
frozen v0 baseline is never touched — this is an additive layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, runtime_checkable

import embodied_sim as E
import run

Coord = tuple[int, int]
Command = str           # a discrete move today; richer (vector) for physics later
State = Any             # backend-defined opaque state
WorldPolicy = Callable[["World", State], Optional[Command]]

_OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


@dataclass(frozen=True)
class KinematicLimits:
    """Admissible-maneuver model (backend-neutral, discrete heading).

    Defaults are fully permissive so `GridWorld` matches `embodied_sim`.
    `allow_reverse=False` forbids an instantaneous 180° flip; `max_turn_deg<90`
    forbids any turn (straight-only). Physics backends map these to real
    yaw-rate / acceleration limits.
    """

    allow_reverse: bool = True
    max_turn_deg: float = 180.0

    def feasible(self, heading: Optional[str], command: Optional[str]) -> bool:
        if command is None:
            return False
        if heading is None:               # first step: any direction admissible
            return True
        if command == heading:
            return True
        if command == _OPPOSITE[heading]:                     # 180° reversal
            return self.allow_reverse and self.max_turn_deg >= 180.0
        return self.max_turn_deg >= 90.0                      # 90° turn


@runtime_checkable
class World(Protocol):
    """Backend contract the arbiter/monitor/metrics depend on."""

    name: str
    optimal_len: int

    def reset(self) -> State: ...
    def at_goal(self, state: State) -> bool: ...
    def is_safe_command(self, state: State, command: Optional[Command]) -> bool: ...
    def safe_commands(self, state: State) -> list[Command]: ...
    def fallback_command(self, state: State) -> Optional[Command]: ...
    def step(self, state: State, command: Command) -> State: ...
    def observation(self, state: State) -> Any: ...


# --- grid backend ---------------------------------------------------------------

class GridWorld:
    """Frozen-grid kinematic backend. State = (position, heading)."""

    def __init__(self, grid: run.Grid, start: Coord, goal: Coord,
                 limits: Optional[KinematicLimits] = None) -> None:
        self.grid = grid
        self.start = start
        self.goal = goal
        self.dist = run.distance_field(grid, goal)
        d = self.dist[start[0]][start[1]]
        self.optimal_len = d if d is not None else 0
        self.limits = limits or KinematicLimits()
        self.name = "grid_kinematic"

    def reset(self) -> State:
        return (self.start, None)

    def at_goal(self, state: State) -> bool:
        return state[0] == self.goal

    def _grid_safe(self, pos: "Coord", command: Optional[str]) -> bool:
        return E.is_safe(self.grid, pos, command, self.dist)

    def is_safe_command(self, state: State, command: Optional[str]) -> bool:
        pos, heading = state
        return (self._grid_safe(pos, command)
                and self.limits.feasible(heading, command))

    def safe_commands(self, state: State) -> list[Command]:
        return [m for m in run.DIRECTIONS if self.is_safe_command(state, m)]

    def fallback_command(self, state: State) -> Optional[Command]:
        pos, heading = state
        for m in run.correct_moves(self.grid, pos, self.goal, self.dist):
            if self.is_safe_command(state, m):
                return m                                  # feasible optimal move
        sc = self.safe_commands(state)
        return sc[0] if sc else None

    def step(self, state: State, command: Command) -> State:
        pos, _ = state
        return (E.move_dest(pos, command), command)

    def observation(self, state: State) -> Any:
        return (self.grid, state[0], self.goal)


# --- generic closed-loop driver (backend-agnostic) -----------------------------

def run_episode(world: World, policy: WorldPolicy, authority: E.Authority,
                step_cap: Optional[int] = None) -> E.EpisodeResult:
    """Run one episode against any `World` under the given authority."""
    state = world.reset()
    optimal_len = world.optimal_len
    if step_cap is None:
        step_cap = max(4 * max(optimal_len, 1), 64)

    steps = n_veto = n_fallback = n_props = n_safe = 0
    collided = reached = False
    termination = "step_cap"

    while steps < step_cap:
        if world.at_goal(state):
            reached, termination = True, "goal"
            break
        proposed = policy(world, state)
        proposed_safe = world.is_safe_command(state, proposed)
        if proposed is not None:                      # count proposals uniformly
            n_props += 1
            if proposed_safe:
                n_safe += 1

        if authority is E.Authority.PLANNER_ONLY:
            executed, source, vetoed = world.fallback_command(state), "planner", False
        elif authority is E.Authority.UNGATED:
            executed, source, vetoed = proposed, "policy", False
        elif proposed_safe:                            # A1: accept safe proposal
            executed, source, vetoed = proposed, "policy", False
        else:                                          # A1: veto -> planner fallback
            executed, source, vetoed = world.fallback_command(state), "fallback", True

        if vetoed:
            n_veto += 1
        if source == "fallback":
            n_fallback += 1

        if executed is None:
            termination = "stuck"
            break
        if not world.is_safe_command(state, executed):     # only reachable UNGATED
            collided, termination = True, "collision"
            steps += 1
            break
        state = world.step(state, executed)
        steps += 1

    if world.at_goal(state) and not reached:
        reached, termination = True, "goal"

    return E.EpisodeResult(
        reached=reached, collided=collided, termination=termination, steps=steps,
        optimal_len=optimal_len, path_len=steps, n_veto=n_veto,
        n_fallback=n_fallback, n_proposals=n_props, n_safe_proposals=n_safe,
    )


def grid_policy(fn: E.Policy) -> WorldPolicy:
    """Adapt a grid policy `fn(grid,pos,goal)->move` to the World interface."""
    def policy(world: World, state: State) -> Optional[Command]:
        grid, pos, goal = world.observation(state)
        return fn(grid, pos, goal)
    return policy


def grid_worlds_from_dataset(path: str = run.DEFAULT_DATASET_PATH,
                             limit: Optional[int] = None,
                             limits: Optional[KinematicLimits] = None) -> list[GridWorld]:
    samples = run.load_dataset(path)
    if limit is not None:
        samples = samples[:limit]
    return [GridWorld(s.grid, s.position, s.goal, limits) for s in samples]


def run_arm(worlds: list[World], policy: WorldPolicy,
            authority: E.Authority) -> list[E.EpisodeResult]:
    return [run_episode(w, policy, authority) for w in worlds]


# --- AirSim backend template (local-only; lazy import) -------------------------

class AirSimWorld:
    """Microsoft AirSim (Unreal + PhysX) backend — same World protocol.

    Runs ONLY on a local machine with AirSim + an Unreal environment. The
    `import airsim` is deferred to `connect()` so this module still imports in
    CI (no GPU/Unreal). Because it satisfies `World`, the L2 arbiter, L1 monitor,
    and metrics in `run_episode` work unchanged — only the dynamics/collision
    backend changes. Fill in the TODOs locally; see `docs/AIRSIM_INTEGRATION.md`.

    Mapping (occupancy grid -> flight): each grid cell <-> a waypoint at fixed
    altitude z0; a discrete move -> a `moveToPositionAsync` to the neighbour
    waypoint; L1 collision uses `simGetCollisionInfo`; the FPV view comes from
    `simGetImages`. Kinematic feasibility maps `KinematicLimits` to yaw-rate /
    speed caps.
    """

    def __init__(self, grid: run.Grid, start: Coord, goal: Coord,
                 z0: float = -3.0, cell_size: float = 1.0,
                 speed: float = 1.0,
                 limits: Optional[KinematicLimits] = None) -> None:
        self.grid, self.start, self.goal = grid, start, goal
        self.z0, self.cell_size, self.speed = z0, cell_size, speed
        self.dist = run.distance_field(grid, goal)
        d = self.dist[start[0]][start[1]]
        self.optimal_len = d if d is not None else 0
        self.limits = limits or KinematicLimits()
        self.name = "airsim_physx"
        self._client = None

    # -- world<->flight coordinate mapping --
    def cell_to_xy(self, cell: Coord) -> tuple[float, float]:
        r, c = cell
        return (r * self.cell_size, c * self.cell_size)        # NED: x=row, y=col

    def connect(self):  # call locally before run_episode
        import airsim  # noqa: lazy — local-only dependency
        self._client = airsim.MultirotorClient()
        self._client.confirmConnection()
        self._client.enableApiControl(True)
        self._client.armDisarm(True)
        self._client.takeoffAsync().join()
        return self._client

    # -- World protocol --
    def reset(self) -> State:
        self._collided = False
        if self._client is not None:
            x, y = self.cell_to_xy(self.start)
            self._client.moveToPositionAsync(x, y, self.z0, self.speed).join()
        return (self.start, None)

    def at_goal(self, state: State) -> bool:
        return state[0] == self.goal

    def is_safe_command(self, state: State, command: Optional[str]) -> bool:
        # Pre-check against the SAME occupancy oracle (cheap, deterministic) AND
        # kinematic feasibility — the L1 monitor that makes A1 safe-by-construction.
        pos, heading = state
        return (E.is_safe(self.grid, pos, command, self.dist)
                and self.limits.feasible(heading, command))

    def safe_commands(self, state: State) -> list[Command]:
        return [m for m in run.DIRECTIONS if self.is_safe_command(state, m)]

    def fallback_command(self, state: State) -> Optional[Command]:
        pos, _ = state
        for m in run.correct_moves(self.grid, pos, self.goal, self.dist):
            if self.is_safe_command(state, m):
                return m
        sc = self.safe_commands(state)
        return sc[0] if sc else None

    def step(self, state: State, command: Command) -> State:
        # Execute the waypoint move in the physics engine and read back collision.
        nxt = E.move_dest(state[0], command)
        if self._client is not None:
            x, y = self.cell_to_xy(nxt)
            self._client.moveToPositionAsync(x, y, self.z0, self.speed).join()
            if self._client.simGetCollisionInfo().has_collided:
                self._collided = True  # L1 secondary check; arbiter prevents this under A1
        return (nxt, command)

    def fpv_image(self):
        if self._client is None:
            raise NotImplementedError("FPV capture requires a local AirSim client")
        import airsim  # lazy — local-only dependency
        return self._client.simGetImages([airsim.ImageRequest(
            "front_center", airsim.ImageType.Scene, False, False)])

    def observation(self, state: State) -> Any:
        # Today: full occupancy (matches the grid study). For partial observability
        # (UAV_TRANSITION_PLAN step 1) return a local window + the FPV image here.
        return (self.grid, state[0], self.goal)


def main() -> None:
    """Smoke: GridWorld through the pluggable layer reproduces the safety result."""
    worlds = grid_worlds_from_dataset(limit=200)
    arms = {
        "planner_only": (grid_policy(E.oracle_policy), E.Authority.PLANNER_ONLY),
        "ungated/greedy": (grid_policy(E.greedy_manhattan_policy), E.Authority.UNGATED),
        "A1/greedy": (grid_policy(E.greedy_manhattan_policy), E.Authority.A1_SELECT_FROM_SAFE),
        "A1/random": (grid_policy(E.make_random_policy(0)), E.Authority.A1_SELECT_FROM_SAFE),
    }
    print(f"pluggable-World smoke on {len(worlds)} grid episodes\n")
    print(f"{'arm':<16} {'succ':>5} {'SPL':>5} {'coll':>5} {'veto':>5}")
    for name, (pol, auth) in arms.items():
        s = E.summarize(run_arm(worlds, pol, auth))
        print(f"{name:<16} {s['success_rate']:>5.2f} {s['spl']:>5.2f} "
              f"{s['collision_rate']:>5.2f} {s['veto_rate']:>5.2f}")


if __name__ == "__main__":
    main()
