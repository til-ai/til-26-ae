# Entity Refactoring — Migration Record

> **Status: COMPLETE.**  The `entities/` package is live.  All import sites
> use `from til_environment.entities import X` and continue to work unchanged.
> This document is kept as a reference for the rationale and decisions made
> during the migration.

---

## 1. Current Scope of `entities.py`

`entities.py` is currently a single 1,040-line file that owns five distinct
conceptual layers, all bundled together:

### 1.1 Primitive enums and value types
- `EntityStatus` — `ACTIVE` / `DESTROYED` lifecycle flag
- `PowerUpType` — catalogue of buff types
- `AttackIntent` — a dataclass recording a queued attack

### 1.2 Pluggable geometry types
- `AttackType` (Protocol) — callable: `(position, direction) → list[ndarray]`
- `_rotate()` — shared rotation helper
- `FrontArcAttack`, `AllAroundAttack` — two concrete attack geometries
- `FRONT_ARC`, `ALL_AROUND` — singleton instances
- `VisionType` (Protocol) — callable: `(position, direction) → list[tuple]`
- `SquareVision(n)`, `SkewVision(n, m)` — two concrete vision geometries

### 1.3 Trait protocols
- `Attacker`, `Defender`, `Collector`, `Vision`, `Healer`
- `Product`, `Experience` — secondary protocols
- All are `@runtime_checkable` subclasses of `Protocol`; because they appear
  in the MRO of their implementors, `EntityRegistry.type_index` can key on them
  directly — no need to spell out concrete implementors at query sites.

### 1.4 Entity classes
- `Entity` — abstract base with `entity_id`, `team`, `position`, `status`, registry
  back-ref, `alive`, `destroy()`, distance helpers
- **Dynamic entities** (have mutable state, live the whole episode, can be damaged,
  moved, or produce effects):
  - `Agent` — Attacker, Defender, Vision, Experience
  - `Base` — Defender, Collector, Vision, Healer
  - `Beacon` — Defender, Collector, Vision, Healer
  - `Scout` — Vision
- **Static entities** (spawned once, consumed on contact, carry no mutable game state):
  - `Mission` — collectible; reward on deposit
  - `Recon` — collectible; immediate reward
  - `PowerUp` — collectible; grants buff

### 1.5 EntityRegistry
- 1,040-line file ends with the entire `EntityRegistry` class (~270 lines)
- Four indexes: `pos_index` (2D grid-of-sets), `status_index`, `team_index`,
  `type_index`
- Protocol-aware typed accessors: `agents()`, `defenders()`, `vision_providers()`, etc.
- `type_index` is keyed by **any class in the MRO** of an entity — including protocol
  subclasses like `Defender` and `Vision` — so `_alive_of_type(Defender)` works
  without enumerating concrete types.

---

## 2. Problems with the Current Structure

| Problem | Detail |
|---------|--------|
| **Monolithic file** | 1,040 lines. Adding a new entity type, a new protocol, or a new geometry means editing the same file as the registry and the enums. |
| **Mixed abstraction levels** | Enums, callables, protocols, dataclasses, and an indexed store all coexist with no layered hierarchy. |
| **Static and dynamic entities are indistinguishable** | At a glance, `Mission` and `Agent` look equally important. Their fundamentally different lifecycles and complexity levels are not reflected in the directory structure. |
| **Geometry types orphaned from their users** | `FrontArcAttack` and `SquareVision` are visually adjacent to `EntityStatus`, which has nothing to do with them. |
| **Hard to navigate** | Finding where `Beacon` is defined requires scrolling past hundreds of lines of unrelated geometry and protocol code. |

---

## 3. Proposed Refactoring

### 3.1 New directory layout

```
til_environment/
    entities/
        __init__.py          ← re-exports every public name; all existing
                               import sites continue to work unchanged
        base.py              ← Entity base class, EntityStatus
        protocols.py         ← Attacker, Defender, Collector, Vision, Healer,
                               Product, Experience (all @runtime_checkable)
        geometry.py          ← AttackType, VisionType, _rotate(),
                               FrontArcAttack, AllAroundAttack, FRONT_ARC,
                               ALL_AROUND, SquareVision, SkewVision
        dynamic.py           ← Agent, Base, Beacon, Scout
                               (all import from protocols + geometry)
        static.py            ← Mission, Recon, PowerUp, PowerUpType,
                               AttackIntent
        registry.py          ← EntityRegistry
```

### 3.2 File-by-file responsibility

#### `entities/base.py`
Defines the root of the entity hierarchy and its only required enum.

- `EntityStatus` (IntEnum)
- `Entity` (dataclass) — `entity_id`, `team`, `position`, `status`, back-ref
  wiring, `__setattr__` position-change hook, `alive`, `destroy()`, distance
  helpers

No imports from any other `entities/` submodule. All other submodules import
from this.

#### `entities/protocols.py`
Pure structural typing — no game logic, no numpy arrays (except type hints).

- `Attacker`, `Defender`, `Collector`, `Vision`, `Healer`
- `Product`, `Experience`

Imports: `Protocol`, `runtime_checkable` from `typing`; `numpy` for type hints
only. No imports from `base.py` or any concrete entity file.

#### `entities/geometry.py`
Pluggable geometry callables — no entity state, no registry.

- `_rotate(offset, direction)`
- `AttackType` (Protocol)
- `FrontArcAttack`, `AllAroundAttack`, `FRONT_ARC`, `ALL_AROUND`
- `VisionType` (Protocol) with `.radius` property
- `SquareVision(n)`, `SkewVision(n, m)`

Imports: `numpy`, `Protocol` from `typing`. No imports from any other
`entities/` submodule.

#### `entities/static.py`
Team-neutral collectibles that carry no mutable game state. Spawned by
`ArenaGenerator`, consumed on agent contact, then destroyed.

- `PowerUpType` (IntEnum)
- `AttackIntent` (dataclass) — logically lives here as a produced value
- `Mission(Entity)` — `reward_value`, `difficulty`
- `Recon(Entity)` — `reward_value`
- `PowerUp(Entity)` — `powerup_type`, `duration`, `strength`

Imports: `base.py` only. These entities satisfy no trait protocols.

#### `entities/dynamic.py`
Entities with rich mutable state that persist across multiple ticks and
explicitly implement trait protocols.

- `Agent(Entity, Attacker, Defender, Vision, Experience)`
- `Base(Entity, Defender, Collector, Vision, Healer)`
- `Beacon(Entity, Defender, Collector, Vision, Healer)`
- `Scout(Entity, Vision)`

Imports: `base.py`, `protocols.py`, `geometry.py`.

#### `entities/registry.py`
The indexed store. Imports concrete entity types only to type the named
accessors.

- `EntityRegistry` (full implementation)
- All four indexes: `pos_index`, `status_index`, `team_index`, `type_index`
- Primitive methods: `add`, `remove`, `get`, `at`, `clear`
- Protocol-keyed accessors: `defenders()`, `vision_providers()`, etc.

Imports: `base.py`, `static.py`, `dynamic.py`, `protocols.py`.

#### `entities/__init__.py`
Flat re-export of every public name, so all existing `from til_environment.entities import X`
calls continue to work with zero changes at import sites:

```python
from til_environment.entities.base     import Entity, EntityStatus
from til_environment.entities.protocols import (
    Attacker, Defender, Collector, Vision, Healer, Product, Experience,
)
from til_environment.entities.geometry  import (
    AttackType, FrontArcAttack, AllAroundAttack, FRONT_ARC, ALL_AROUND,
    VisionType, SquareVision, SkewVision,
)
from til_environment.entities.static   import (
    PowerUpType, AttackIntent, Mission, Recon, PowerUp,
)
from til_environment.entities.dynamic  import Agent, Base, Beacon, Scout
from til_environment.entities.registry import EntityRegistry

__all__ = [
    "Entity", "EntityStatus",
    "Attacker", "Defender", "Collector", "Vision", "Healer",
    "Product", "Experience",
    "AttackType", "FrontArcAttack", "AllAroundAttack", "FRONT_ARC", "ALL_AROUND",
    "VisionType", "SquareVision", "SkewVision",
    "PowerUpType", "AttackIntent", "Mission", "Recon", "PowerUp",
    "Agent", "Base", "Beacon", "Scout",
    "EntityRegistry",
]
```

No other file in the package needs to change its imports.

### 3.3 Dependency graph

```
geometry.py   protocols.py
     \              \
      \              \
       ──→ dynamic.py ←── base.py ←── static.py
                \                         |
                 ──────────────────────→ registry.py
                                              |
                                         __init__.py  (re-exports all)
```

No circular dependencies. `base.py` → nothing. `protocols.py` → nothing.
`geometry.py` → nothing. `static.py` → `base.py`. `dynamic.py` → `base.py`,
`protocols.py`, `geometry.py`. `registry.py` → all of the above.

---

## 4. What Does Not Change

| Thing | Why unchanged |
|-------|--------------|
| `EntityRegistry` semantics | Same four indexes, same protocol-keyed `type_index` behaviour |
| `type_index` keys protocols | `Defender`, `Vision`, etc. remain valid keys — MRO indexing is unaffected by the file split |
| All import sites outside `entities/` | `__init__.py` re-exports guarantee this |
| `dynamics.py` | No changes needed — still imports `Agent`, `Beacon`, etc. from `til_environment.entities` |
| `renderer.py`, `observation.py` | Same |
| v1 files (`gridworld.py`, `types.py`, etc.) | Untouched by policy |

---

## 5. Migration Steps (Completed)

1. ✅ Created `til_environment/entities/` directory
2. ✅ Wrote `base.py` — `EntityStatus` + `Entity`
3. ✅ Wrote `protocols.py` — all `@runtime_checkable` protocol classes
4. ✅ Wrote `geometry.py` — `_rotate`, both attack types, both vision types, singletons
5. ✅ Wrote `static.py` — `PowerUpType`, `AttackIntent`, `Mission`, `Recon`, `PowerUp`
6. ✅ Wrote `dynamic.py` — `Agent`, `Base`, `Beacon`, `Scout` with updated imports
7. ✅ Wrote `registry.py` — `EntityRegistry` + `Query` builder with updated imports
8. ✅ Wrote `__init__.py` — flat re-exports as shown above
9. ✅ Deleted `til_environment/entities.py`

**Additional change during migration:** `EntityRegistry` gained a composable
`Query` builder (`registry.query().type(X).team(0).all()`) in `registry.py`,
which was not part of the original plan but was added alongside the split.
