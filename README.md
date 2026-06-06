# TIL Bomberman — Environment

## Quick Start

`play.py` controls: `W/A/S/D` or arrow keys to move, `B/F` to place a bomb, `SPACE` to stay, `T` to toggle the respawn-timer overlay, `R` to reset, `Q`/`ESC` to quit.

---

## Overview

A two-team [PettingZoo](https://pettingzoo.farama.org/) AEC environment built around bomb placement, base defence, and resource management. Each team controls one agent and one stationary base. Agents explore a maze-like arena, collect resource tokens to fund bombs, and destroy the opposing base. Agents are never removed from the game — instead, taking lethal damage freezes them in place for a configurable number of ticks before they respawn at full health on the same tile.

---

## Overall Structure

```
til_environment/
├── bomberman_env.py   — PettingZoo AECEnv wrapper (public API)
├── dynamics.py        — all mutable game state and phase logic
├── arena.py           — maze generation, tile placement, spawn assignment
├── observation.py     — multi-channel viewcone builders
├── actions.py         — Action enum, ActionMask builder
├── entities/
│   ├── dynamic.py     — Agent, Base, Bomb (mutable entities)
│   ├── static.py      — Mission, Recon, Resource (collectible entities)
│   ├── protocols.py   — structural trait protocols (Defender, Attacker, …)
│   ├── geometry.py    — RadiusAttack blast-cell computation
│   └── registry.py    — EntityRegistry (typed, indexed entity container)
├── renderer.py        — pygame rendering; never mutates game state
├── config.py          — OmegaConf dataclass config hierarchy
├── bomberman_config.yaml — default configuration
├── helpers.py         — shared grid utilities
└── types.py           — Direction, Tile, Wall enums
```

**`bomberman_env.py`** is the entry point for external code. It subclasses `AECEnv`, delegates all logic to `Dynamics`, and exposes the standard `reset / step / observe / render` contract. `basic_env()` and `parallel_basic_env()` are convenience factory functions that wrap the raw env with `FlattenDictWrapper` and `frame_stack_v2`.

**`dynamics.py`** owns the mutable world state: agent positions and health, team resource and bomb counts, bomb timers, tile respawn queues, and the per-round phase pipeline. It composes `ArenaGenerator`, `ArenaState`, `EntityRegistry`, `ActionMask`, and the `Rewards` system.

**`arena.py`** generates the maze (Recursive Backtracking via `mazelib`), marks a fraction of walls as destructible, scatters mission/recon/resource tiles with rotational symmetry across teams, and assigns agent/base starting positions. The wall layout is cached between episodes if the seed does not change. A Perlin-noise + centre-gradient respawn map is generated each episode, giving shorter respawn delays near the map centre.

**`entities/`** contains the entity hierarchy. All entities share a common `Entity` base (unique ID, position, team). Behaviour is declared via structural protocols (`Defender`, `Attacker`, `Vision`, `Item`, `Timed`) rather than deep inheritance, so `Dynamics` can dispatch on traits without checking concrete types. `EntityRegistry` is an indexed `dict[str, Entity]` with typed accessors (`agents(team)`, `bases(team)`, `bombs()`, etc.).

**`observation.py`** builds the multi-channel float32 viewcone tensors that agents receive. Line-of-sight is computed with a vectorised supercover algorithm (numpy batch array indexing; precomputed per viewcone shape via `lru_cache`).

**`renderer.py`** is a stateless pygame drawer. In `"human"` mode it draws to a window; in `"rgb_array"` mode it returns a numpy array. It never reads or writes game state — it only reads the `ArenaState`, `EntityRegistry`, and the current `observations` dict passed to it.

**`config.py`** defines the full config hierarchy as OmegaConf dataclasses. Every tuneable parameter lives here; no scattered constants anywhere else.

---

## Action Space

Each agent has a flat `Discrete(6)` action space. Bases do not act.

| Value | Name        | Effect                                               |
|-------|-------------|------------------------------------------------------|
| 0     | FORWARD     | Move one tile in the agent's current facing direction |
| 1     | BACKWARD    | Move one tile opposite to facing                     |
| 2     | LEFT        | Turn left (rotate facing 90°), no movement           |
| 3     | RIGHT       | Turn right (rotate facing 90°), no movement          |
| 4     | STAY        | No action                                            |
| 5     | PLACE_BOMB  | Place a bomb on the agent's current tile             |

An **action mask** (shape `(6,)`, `uint8`) is included in every observation. Illegal actions are replaced with `STAY` and penalised. Movement is masked when a wall blocks the path (structural or destructible). `PLACE_BOMB` is masked when the team has no bombs available or the agent is frozen.

---

## Observation Space

Each agent receives a `Dict` observation:

| Key              | Space                                     | Description                                      |
|------------------|-------------------------------------------|--------------------------------------------------|
| `agent_viewcone` | `Box(0,1, (H,W,25), float32)`             | Multi-channel viewcone centred on the agent      |
| `base_viewcone`  | `Box(0,1, (B,B,25), float32)`             | Square view centred on the team base             |
| `direction`      | `Discrete(4)`                             | Agent's current facing direction                 |
| `location`       | `Box(0,G, (2,), uint8)`                   | Agent grid position                              |
| `base_location`  | `Box(0,G, (2,), uint8)`                   | Team base grid position                          |
| `health`         | `Box(0, max_health, (1,), float32)`       | Agent HP                                         |
| `frozen_ticks`   | `Discrete(freeze_turns+1)`                | Ticks remaining until respawn (0 = active)       |
| `base_health`    | `Box(0, max_health, (1,), float32)`       | Team base HP                                     |
| `team_resources` | `Box(0, max_resources, (1,), float32)`    | Shared resource pool (bombs cost 1.0 unit each)  |
| `team_bombs`     | `Discrete(max_bombs+1)`                   | Bombs currently available to the team            |
| `step`           | `Discrete(num_iters+1)`                   | Current step index                               |
| `action_mask`    | `Box(0,1, (6,), uint8)`                   | Legal actions this step                          |

### Viewcone channels (25 total)

The viewcone is oriented relative to the agent's facing direction. Channel 0 is a line-of-sight flag; channels 1–4 encode wall edges; channels 5–12 encode tile types and entity presence; channels 13–16 flag destructible wall edges; channels 17–24 encode bomb state and entity health ratios. Index by `ViewChannel.*` enum rather than raw integers.

---

## Game Loop and Phases

The environment follows the PettingZoo AEC protocol: agents act one at a time. Once all agents have submitted an action the round executes in a fixed phase order:

1. **Validate** — illegal actions are replaced with `STAY`; `invalid_action` penalty awarded
2. **Place bombs** — `PLACE_BOMB` actions spawn bombs; `team_bombs[team]` decremented
3. **Movement** — agents move or turn; collectibles are picked up inline
4. **Detonation** — expired bomb timers produce `AttackIntent` lists
5. **Damage** — intents applied to defenders; agents at 0 HP enter frozen state
6. **Upkeep** — bomb timers tick; resource pools accrue; auto-convert resources → bombs; frozen agents tick down and respawn

Termination fires when a team's base is destroyed. Truncation fires at `num_iters` steps. Frozen (dead) agents alone do **not** trigger termination.

---

## Resource and Bomb Economy

Each team maintains a floating-point resource pool (`team_resources`) and an integer bomb count (`team_bombs`). Resources accumulate every tick at `base_resource_rate` as long as the team's base is alive. Picking up a resource tile adds a fixed amount to the pool. When the pool reaches `bomb_cost` (default 1.5), it is decremented and `team_bombs` incremented. Bombs start the episode at `starting_bombs` (default 3).

---

## Configuration

All parameters live in `bomberman_config.yaml` (loaded into an OmegaConf `BombermanConfig`). Key sections:

```yaml
env:
  grid_size: 16
  num_teams: 2
  num_iters: 200
dynamics:
  vision:
    left: 2
    right: 2
    ahead: 4
    behind: 2
entities:
  agent:
    max_health: 60.0
    freeze_turns: 3
  base:
    max_health: 100.0
  bomb:
    timer: 3
    blast_radius: 2
resources:
  bomb_cost: 1.5
  base_resource_rate: 0.1
  starting_bombs: 3
rewards:
  collect_mission: 5.0
  attack_kill: 15.0
  destroy_enemy_base: 50.0
```

Load a custom config:

```python
from til_environment.config import load_config
from til_environment.bomberman_env import Bomberman

env = Bomberman(load_config("my_config.yaml"))
```

Or use the default:

```python
from til_environment.bomberman_env import basic_env

env = basic_env()  # includes FlattenDictWrapper + frame_stack_v2
```

---

## Tests

```bash
uv run pytest til_environment/tests/ -v
```

The test suite covers bomb placement and detonation, destructible wall behaviour, resource/bomb economy, agent freeze and respawn, termination conditions, action masking, and the Perlin respawn map.
