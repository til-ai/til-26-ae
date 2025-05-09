# TIL Bomberman v2 — Architecture

## Overview

The v2 environment refactors the original monolithic `gridworld.py` into a
layered architecture with clear separation of concerns. **No original files
were modified or deleted** — v1 continues to work as-is.

```
┌─────────────────────────────────────────────────────────────┐
│                    gridworld_v2.py                           │  ← Tier A: AECEnv API
│  Bomberman(AECEnv)                                          │
│  Only implements: reset, step, observe,                     │
│  render, close, observation_space, action_space             │
└──────────┬─────────────────┬────────────────────────────────┘
           │                 │
           ▼                 ▼
┌──────────────────────┐  ┌──────────────────┐  ┌────────────────────────┐
│   dynamics.py        │  │  renderer.py     │  │  config.py             │
│                      │  │                  │  │                        │ ← Tier C
│  Dynamics(cfg)       │  │  Renderer        │  │  BombermanConfig       │
│  ├ VisionSystem      │  │  (all pygame     │  │  ├ EnvConfig           │
│  └ (composes arena,  │  │   drawing)       │  │  ├ DynamicsConfig      │
│     actions, obs)    │  │                  │  │  ├ EntitiesConfig      │
└──┬───────┬───────┬───┘  └──────────────────┘  │  ├ RendererConfig      │
   │       │       │                             │  ├ RewardsConfig       │
   ▼       ▼       ▼                             │  └ ObsSpaceConfig      │
┌────────┐┌──────────┐┌──────────────┐           └────────────────────────┘
│arena.py││actions.py││observation.py│
│        ││          ││              │           ← Tier C: Utility systems
│ Arena  ││ Action ││ ViewChannel  │
│ Gen.   ││ BaseAct. ││ 18-ch build  │
│        ││ Masks    ││              │
└────────┘└──────────┘└──────────────┘
   │
   ▼
┌────────────────────┐  ┌──────────────────┐
│   entities/        │  │  types_v2.py     │   ← Tier B: Data / domain model
│   (see ENTITY_     │  │                  │
│   REFACTOR.md)     │  │  RewardEvent     │
│                    │  │  ObservationV2   │
│  Entity (base)     │  │  TeamResult      │
│  ├ Agent           │  │  (+ re-exports   │
│  ├ Base            │  │   from types.py) │
│  ├ Beacon          │  │                  │
│  ├ Scout           │  │                  │
│  ├ Mission         │  │                  │
│  ├ Recon           │  │                  │
│  └ PowerUp         │  │                  │
│                    │  │                  │
│  Geometry:         │  │                  │
│  AttackType        │  │                  │
│  VisionType        │  │                  │
│                    │  │                  │
│  Trait protocols:  │  │                  │
│  Attacker → Agent  │  │                  │
│  Defender → Agent  │  │                  │
│           Base     │  │                  │
│           Beacon   │  │                  │
│  Collector → Base  │  │                  │
│            Beacon  │  │                  │
│  Vision → Agent    │  │                  │
│           Base     │  │                  │
│           Beacon   │  │                  │
│           Scout    │  │                  │
│  Healer → Base     │  │                  │
│           Beacon   │  │                  │
│                    │  │                  │
│  EntityRegistry    │  │                  │
└────────────────────┘  └──────────────────┘
```

---

## Configuration — `config.py` + `default_config.yaml`

**Every tuneable parameter** is declared in structured dataclasses and loaded
from YAML via OmegaConf.  A single `DictConfig` flows top-down:

```
BombermanConfig
├── env           grid_size, num_teams, agents_per_team, num_iters, novice, render_mode
├── dynamics
│   ├── arena     wall_prob, mission_prob, recon_prob, powerup_prob
│   └── vision    left, right, behind, ahead
├── entities
│   ├── agent     health, max_health, attack, defense
│   ├── base      health, max_health, resource_rate, heal_rate, vision_radius (→ vision_type)
│   ├── beacon    vision_radius (→ vision_type), heal_rate, heal_radius, manufacture_cost
│   ├── scout     velocity, vision_radius (→ vision_type), manufacture_cost
│   ├── mission   reward_value, difficulty
│   └── powerup   duration, strength
├── renderer      window_size, debug, render_fps
├── rewards       14 reward-shaping values (collisions, kills, collection, penalties, …)
└── obs_space     max_agent_health, max_base_health, max_base_resources, max_inventory, …
```

| Function | Purpose |
|----------|---------|
| `default_config()` | Fresh `DictConfig` with all dataclass defaults |
| `load_config(path)` | Merge a YAML file onto structured defaults (catches typos) |
| `save_config(cfg, path)` | Serialise config to YAML |
| `viewcone_tuple(cfg)` | Extract `(left, right, behind, ahead)` tuple |

`Bomberman.__init__` accepts a `DictConfig` and passes it to `Dynamics(cfg)`.
Entity creation in `Dynamics.reset()` reads from `cfg.entities.*`.
Rewards in `Bomberman._apply_*_rewards()` read from `cfg.rewards.*`.

---

## Tier A — `gridworld_v2.py` → `Bomberman`

**Responsibility:** PettingZoo AEC contract — nothing else.

| Method | What it does |
|--------|-------------|
| `__init__(cfg)` | Stores config, creates `Dynamics(cfg)` and `Renderer(…)` |
| `reset()` | Seeds RNG, calls `Dynamics.reset()`, sets up agent selectors |
| `step(action)` | Buffers per-agent dict actions, calls `_execute_round()` when all agents have acted |
| `observe(agent)` | Delegates to `Dynamics.observe()`, stamps current step count + action masks |
| `render()` | Delegates to `Renderer.render()` |
| `close()` | Delegates to `Renderer.close()` |
| `observation_space()` | Returns `gymnasium.spaces.Dict` with bounds from `cfg.obs_space` |
| `action_space()` | Returns `Dict({"agent": Discrete(7), "base": Discrete(3)})` |

The class owns **no** movement logic, collision math, or pygame calls.

**Factory functions:**
- `env_v2(cfg)` — wraps `Bomberman` with `FlattenDictWrapper`, `frame_stack_v2`, bounds-check, and order-enforcing wrappers.
- `parallel_env_v2` — parallel conversion of `env_v2`.

---

## Tier B — Entity classes (`entities/`, `types_v2.py`)

> **Note:** `entities.py` has been split into the `entities/` package.
> See `ENTITY_REFACTOR.md` for the full rationale and migration record.
> All names remain importable from `til_environment.entities` via the
> package `__init__.py` re-export.

### Entity hierarchy

All game objects inherit from `Entity` and explicitly declare each protocol they satisfy as a base class.

**Dynamic entities** — persistent mutable state, can be damaged / moved / craft effects:

| Class | Key attributes | Traits |
|-------|---------------|--------|
| **Entity** | `entity_id`, `team`, `position`, `status` | — | Abstract base; owns `alive`, `destroy()`, distance helpers |
| **Agent** | `health`, `max_health`, `attack`, `defense`, `experience`, `direction`, `active_powerups`, `attack_type`, `vision_type` | Attacker, Defender, Vision, Experience | Mobile unit. Collects missions + recon, fights enemies. Uses directional viewcone in `VisionSystem`; `vision_type` adds optional radius-based supplement |
| **Base** | `health`, `max_health`, `resources`, `resource_rate`, `heal_rate`, `heal_radius`, `vision_type` | Defender, Collector, Vision, Healer | Stationary. Absorbs attacks, gathers resources, heals allies, reveals area. Loss condition if destroyed |
| **Beacon** | `health`, `max_health`, `vision_type`, `heal_rate`, `heal_radius`, `manufacture_cost` | Defender, Collector, Vision, Healer | Placed by agents. Can be destroyed; provides vision + healing aura |
| **Scout** | `direction`, `velocity`, `vision_type`, `manufacture_cost` | Vision | Mobile recon unit launched from Base. Destroyed on wall hit |

**Static entities** — spawned once, no mutable game state, consumed on agent contact:

| Class | Key attributes | Notes |
|-------|---------------|-------|
| **Mission** | `reward_value`, `difficulty` | Team-neutral. Reward on collection |
| **Recon** | `reward_value` | Team-neutral. Reward on collection |
| **PowerUp** | `powerup_type`, `duration`, `strength` | Team-neutral. Grants buff. Can co-exist with Mission/Recon on same tile |

### Pluggable geometry types

Attack and vision geometry are decoupled from entity classes via callable
protocols.  Each entity holds a pluggable instance; `Dynamics` never needs to
know which implementation is active.

**Attack types** — `(position, direction) → list[ndarray]`:

| Class | Cells returned | Direction-aware? |
|-------|---------------|------------------|
| `FrontArcAttack` | 5-cell front arc: directly ahead + 2 front diagonals + 2 flanks | Yes |
| `AllAroundAttack` | All 8 Moore-neighbourhood tiles | No |

**Vision types** — `(position, direction) → list[tuple[int,int]]`; also expose `.radius` for `build_radius_view`:

| Class | Cells returned | Direction-aware? |
|-------|---------------|------------------|
| `SquareVision(n)` | `(2n+1)×(2n+1)` square centred on entity | No |
| `SkewVision(n, m)` | Same base square + `m` extra tile-depths in facing direction | Yes |

### Trait protocols

Structural protocols enable polymorphic dispatch without inheritance coupling.
Each concrete entity class **explicitly inherits** the protocols it satisfies.

Because `Attacker`, `Defender`, etc. are *subclasses* of `Protocol` (not
`Protocol` itself), they appear in each implementor’s `__mro__` and are
keyed in `EntityRegistry.type_index`.  This means `registry.defenders()` calls
`_alive_of_type(Defender)` directly — no need to enumerate concrete types.

| Protocol | Key members | Implementors |
|----------|------------|--------------|
| `Attacker` | `attack_type: AttackType`, `attack_power: float`, `get_attack_cells()` | Agent |
| `Defender` | `health`, `receive_damage()` | Agent, Base, Beacon |
| `Collector` | `collect_resources()` | Base, Beacon |
| `Vision` | `vision_type: VisionType`, `get_visible_cells()` | Agent, Base, Beacon, Scout |
| `Healer` | `heal_rate`, `heal_radius` | Base, Beacon |

### EntityRegistry

A flat `dict[str, Entity]` store with four complementary indexes:

- `pos_index` — `list[list[set[str]]]` pre-allocated `grid_size×grid_size`; direct `[x][y]` access, O(1)
- `status_index` — `EntityStatus → set[entity_id]`
- `team_index` — `team → set[entity_id]`
- `type_index` — `type → set[entity_id]`; keyed by **every class in the MRO** except `object`, `Generic`, `Protocol`

Typed accessor methods fall into two groups:
- **Concrete-type accessors**: `agents(team?)`, `bases(team?)`, `beacons(team?)`, `scouts(team?)`, `missions()`, `recons()`, `powerups()`
- **Protocol accessors**: `attackers(team?)`, `defenders(team?)`, `collectors(team?)`, `vision_providers(team?)`, `healers(team?)` — each calls `_alive_of_type(TheProtocol)` directly

A **Query builder** (`registry.query()`) supports composable, lazy multi-criteria
lookups across type, team, status, and position:

```python
registry.query().type(Defender).team(0).status(EntityStatus.ACTIVE).all()
```

### types_v2.py

| Type | Purpose |
|------|---------|
| `RewardEvent` | 16 canonical reward event names |
| `ObservationV2` | TypedDict specifying the full observation dict shape |
| `TeamResult` | IN_PROGRESS / WIN / LOSS / DRAW enum |

---

## Tier C — Utility systems

### `dynamics.py` → `Dynamics`

Owns **all mutable state transitions**.  Accepts a `DictConfig` and extracts
all parameters from it.

| Sub-component | What it does |
|--------------|-------------|
| `VisionSystem` | Line-of-sight (LOS) checks, viewcone construction, team-wide visible area |
| `Dynamics` (main) | 10-phase round execution: movement, combat, crafting, scouting, collection, upkeep, termination |

#### State grid encoding (`_state`)

`_state` is a `(grid_size, grid_size) uint8` array that encodes **walls only**:

```
Bit   │ Meaning                         │ Direction.value
──────┼─────────────────────────────────┼────────────────
  0   │ Wall on RIGHT edge of tile      │ Direction.RIGHT = 0
  1   │ Wall on BOTTOM edge of tile     │ Direction.DOWN  = 1
  2   │ Wall on LEFT edge of tile       │ Direction.LEFT  = 2
  3   │ Wall on TOP edge of tile        │ Direction.UP    = 3
 4–7  │ Always zero
```

The bit index is always `Direction.value` for the corresponding facing direction.
`convert_tile_to_edge` (v1 helper, unmodified) writes walls into bits 4–7;
`ArenaGenerator.generate_walls` applies `wall_grid >>= 4` immediately after to
shift them to bits 0–3.

Wall mutation is provided by `Dynamics.destroy_wall(x, y, direction)` and
`Dynamics.add_wall(x, y, direction)`.  Both methods keep `_state` and
`self.walls` (the LOS edge-set) in sync atomically.

**Static entity presence** (Mission, Recon, PowerUp) is tracked **exclusively**
in the `EntityRegistry`.  There is no redundant tile-type bitmask in `_state`.
This means:
- A tile can hold a Mission **and** a PowerUp simultaneously.
- Picking up an entity is just `entity.destroy()` — no grid mutation needed.
- `_handle_tile_pickup` queries the registry, not the state array.

Entity creation in `reset()` reads from `cfg.entities.agent`, `cfg.entities.base`,
and from `ArenaResult.static_entities` (list of `StaticEntitySpec`).
Craft phase reads from `cfg.entities.scout`, `cfg.entities.beacon`.

Key design rule: **Dynamics never touches pygame.** It only mutates the
numpy state grid + EntityRegistry.

### `arena.py` → `ArenaGenerator`

Generation is split into **two distinct stages** for efficiency and flexibility:

#### Stage 1 — Wall generation (`generate_walls`, cached)

| Method | What it does |
|--------|-------------|
| `generate_walls(rng, seed)` | Runs Prims/DungeonRooms maze, knocks down walls, builds `WallResult`. **Cached** — only rerun when seed changes. |
| `_build_wall_set(wall_grid)` | Converts wall grid to `set[edge]` for O(1) LOS checks |

The **wall grid** is a `uint8` array with bits 0–3 encoding the four wall
directions (aligned with `Direction.value`: RIGHT=0, BOTTOM=1, LEFT=2, TOP=3).
Bits 4–7 are always zero.  `convert_tile_to_edge` (v1 helper) writes into bits
4–7 using the v1 `Wall` enum; `generate_walls` right-shifts by 4 immediately
after to land in bits 0–3.

The cached `WallResult` is reused for consecutive episodes on the same seed,
making `Dynamics.reset()` fast without regenerating the expensive maze.

#### Stage 2 — Episode generation (`generate_episode`, always fresh)

| Method | What it does |
|--------|-------------|
| `generate_episode(rng, seed, …)` | Calls cached `generate_walls` + `generate_static_layer` + spread starts |
| `generate_static_layer(rng, wall_grid)` | Scatters Mission / Recon / PowerUp entities per-tile with configurable probabilities |
| `_spread_starting_positions()` | Places bases and agents with maximum separation |

**Static entity rules** (all independent per non-wall tile):
- `mission_prob` → `"mission"` spec
- `recon_prob` → `"recon"` spec (mutually exclusive with mission in same roll, drawn from a multinomial)
- `powerup_prob` → `"powerup"` spec (second independent Bernoulli roll — can co-exist with a mission or recon)

Returns an `ArenaResult` containing the wall grid, walls set, and a list of
`StaticEntitySpec(kind, position)` that `Dynamics.reset()` uses to instantiate
entities into the `EntityRegistry`.

#### Key data types

| Type | Purpose |
|------|---------|
| `WallResult` | `{ wall_grid, walls, seed }` — cached output of wall generation |
| `StaticEntitySpec` | `{ kind: "mission"/"recon"/"powerup", position }` — one spec per entity to spawn |
| `ArenaResult` | Full episode setup: wall grid + walls + static entity specs + start positions |

### `actions.py` → `Action`, `BaseAction`, masks

Defines the expanded action vocabulary and per-step mask builders:

| Class / Enum | Values |
|-------------|--------|
| `Action` | FORWARD, BACKWARD, LEFT, RIGHT, STAY, ATTACK, ITEM |
| `BaseAction` | DO_NOTHING, CRAFT_SCOUT, CRAFT_BEACON |
| `ActionMask` | Builds per-agent mask (e.g. ATTACK only if adjacent enemy) |
| `BaseActionMask` | Builds per-agent base-action mask (e.g. CRAFT only if resources ≥ cost) |

Actions are grouped into `ActionGroup` categories (MOVEMENT, COMBAT, UTILITY)
for mutual-exclusion validation.

### `observation.py` → multi-channel viewcone builders

Replaces v1's bitwise `uint8` encoding with sparse `float32` tensors (H × W × 18):

| Function | What it builds |
|----------|---------------|
| `build_agent_viewcone()` | Directional H × W × C tensor for an agent's facing-relative view |
| `build_radius_view()` | Circular (2R+1)² × C tensor for bases, beacons, scouts |
| `populate_channels()` | Shared core: stamps wall, tile, and entity data into a view |

18 channels defined by `ViewChannel` IntEnum: VISIBLE, 4× walls, 3× tile types,
2× ally/enemy for agents/bases/beacons/scouts, MISSION_ITEM, POWERUP_ITEM.

### `renderer.py` → `Renderer`

Owns **all pygame drawing**:

- Lazy display initialisation (only creates a window on first `render()` call)
- Draws tiles, walls, grid lines from the numpy state array
- Draws entities (agents with health bars, bases, beacons, scouts, missions, powerups)
  via the EntityRegistry
- Debug panel showing per-agent viewcone observations, rewards, and actions

Key design rule: **Renderer never mutates game state.** It only reads.

---

## Game Flow (one episode)

```
Bomberman.reset(seed)
  └→ Dynamics.reset(seed)
       ├→ ArenaGenerator.generate()      # maze + tiles + starting positions
       └→ populate EntityRegistry        # Agent, Base, Mission from cfg.entities.*

for each step:
  Bomberman.step({"agent": int, "base": int})
    ├→ buffer dict action for current agent
    └→ (on last agent) _execute_round()
         │
         │  Phase 0: Validate actions against masks
         │  Phase 1: Dynamics.resolve_movement_actions()  — move + collisions + pickups + deposits
         │  Phase 2: Dynamics.attack_phase()              — collect attack intents
         │  Phase 3: Dynamics.damage_phase()              — apply damage, destroy at 0 HP
         │  Phase 4: Dynamics.craft_phase()               — base manufactures scouts / beacons
         │  Phase 5: Dynamics.scout_phase()               — advance scouts, destroy on wall hit
         │  Phase 6: Dynamics.collect_phase()             — bases gather passive resources
         │  Phase 7: Dynamics.upkeep()                    — beacon healing, powerup timers, mission processing
         │  Phase 8: Step / idle penalties
         │  Phase 9: Dynamics.check_termination()         — base/agent destruction
         │  Phase 10: Rebuild observations + render
         │
         └→ rewards applied from cfg.rewards.* at each phase
```

---

## File Inventory

| File | Lines | Status | Description |
|------|-------|--------|-------------|
| `gridworld.py` | 863 | **Unchanged** (v1) | Original monolithic environment |
| `types.py` | 206 | **Unchanged** (v1) | Direction, Action, Tile, Wall, Player, RewardNames |
| `helpers.py` | 145 | **Unchanged** | Math/vision helper functions |
| `flatten_dict.py` | 24 | **Unchanged** | PettingZoo wrapper |
| `__init__.py` | 1 | **Unchanged** | Package imports (v1) |
| `config.py` | 305 | **v2** | OmegaConf structured config dataclasses + load/save |
| `default_config.yaml` | — | **v2** | Reference YAML with all defaults |
| `dynamics.py` | 905 | **v2** | Dynamics(cfg), VisionSystem, all game logic |
| `arena.py` | 580 | **v2** | ArenaGenerator, ArenaResult, two-stage generation |
| `actions.py` | 300 | **v2** | Action, BaseAction, ActionMask, BaseActionMask |
| `observation.py` | 380 | **v2** | ViewChannel, multi-channel viewcone builders |
| `renderer.py` | 614 | **v2** | Renderer (pygame) |
| `gridworld_v2.py` | 480 | **v2** | Bomberman(AECEnv) v2 |
| `types_v2.py` | 120 | **v2** | RewardEvent, ObservationV2, TeamResult |
| `test.py` | 271 | **v2** | Interactive keyboard test harness |
| **`entities/` package** | | **v2** | Entity hierarchy split from the original monolithic `entities.py` |
| `entities/__init__.py` | 74 | **v2** | Flat re-export of every public name |
| `entities/base.py` | 89 | **v2** | Entity base class, EntityStatus |
| `entities/protocols.py` | 91 | **v2** | Trait protocols: Attacker, Defender, Collector, Vision, Healer, Product, Experience |
| `entities/geometry.py` | 207 | **v2** | AttackType, VisionType, FrontArcAttack, AllAroundAttack, SquareVision, SkewVision |
| `entities/dynamic.py` | 234 | **v2** | Agent, Base, Beacon, Scout |
| `entities/static.py` | 128 | **v2** | Mission, Recon, PowerUp, PowerUpType, AttackIntent |
| `entities/registry.py` | 348 | **v2** | EntityRegistry + Query builder |
| `ENTITY_REFACTOR.md` | — | **v2** | Completed migration record: monolithic `entities.py` → `entities/` package |

---

## How to Extend

- **New entity type**: Add a dataclass in `entities/dynamic.py` (or `entities/static.py`
  for a collectible), add a config dataclass in `config.py` (+ YAML defaults),
  add a draw method in `renderer.py`, handle its game logic in `dynamics.py`.
  Re-export from `entities/__init__.py`.
- **New attack geometry**: Add a class to `entities/geometry.py` implementing
  `AttackType`; pass it as `attack_type=` when constructing an `Agent`.
- **New vision geometry**: Add a class to `entities/geometry.py` implementing
  `VisionType`; pass it as `vision_type=` when constructing any Vision entity.
- **New trait protocol**: Add to `entities/protocols.py`; inherit it in the
  relevant entity class; add an accessor to `EntityRegistry` in `entities/registry.py`.
- **New reward event**: Add to `RewardsConfig` in `config.py`, add the YAML
  default, wire it in `Bomberman._apply_*_rewards()`.
- **New action**: Extend `Action` or `BaseAction` in `actions.py`, update
  the mask builder, handle in `Dynamics`.
- **New observation channel**: Add to `ViewChannel` in `observation.py`,
  add a populate clause in `populate_channels()`.
- **Alternative renderer**: Implement the same
  `render(state, registry, ...)` interface as `Renderer`.
