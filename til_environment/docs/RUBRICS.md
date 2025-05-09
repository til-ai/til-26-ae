# RL Environment Quality Rubric — TIL Bomberman v2

This document rates the environment against standard criteria for production-ready
multi-agent RL environments.  Each category is scored out of 10 with specific
evidence from the codebase.

---

## Summary Scorecard

| Category | Score | Grade |
|---|---|---|
| 1. Observation Space Design | 6 / 10 | B− |
| 2. Action Space Design | 8 / 10 | B+ |
| 3. Reward Function | 6 / 10 | B− |
| 4. Episode Structure | 8 / 10 | B+ |
| 5. Multi-Agent Design | 7 / 10 | B |
| 6. Reproducibility & Testability | 6 / 10 | B− |
| 7. Code Quality & Extensibility | 9 / 10 | A− |
| 8. Performance | 6 / 10 | B− |
| **Overall** | **50 / 80** | **B** |

---

## 1. Observation Space Design — 6 / 10

### What is done well

- **18-channel float32 spatial tensor** for agent/base viewcones.  Each channel
  is a single semantically meaningful layer (visibility, wall edges, entity
  presence per team).  Competitors can visualise any channel directly; no
  bit-unpacking is needed before model input.
- **`populate_channels()`** is shared between agent directional viewcones and
  base/beacon radius views, so the spatial encoding is consistent.
- **Action mask** is embedded in the observation dict, allowing masking-aware
  policies (PPO + invalid-action masking) without any custom wrapper.
- **Dual spatial views** (`agent_viewcone` + `base_viewcone`) give an agent
  both local egocentric and wider base-centred context in one observation.

### Areas for improvement

- **`direction` is a raw `Discrete(4)` integer.**  Neural networks treat this
  as an ordinal/continuous value (0 < 1 < 2 < 3 ≈ ordered).  A one-hot or
  sine/cosine encoding (`sin(θ)`, `cos(θ)`) would avoid this.

- **`step` is a raw `Discrete(num_iters + 1)` integer.**  A value of 499 is
  meaningless to a network that saw values 0–499 uniformly.  Normalising to
  `step / num_iters ∈ [0, 1]` (a `Box(0, 1, shape=(1,))`) would be better.

- **`location` is absolute coordinates.**  Absolute positions break
  translational symmetry: a policy trained on one map region may not generalise
  to another, even on the same maze layout.  Position relative to the team base
  would be more useful.

- **`health` and `base_health` use raw `Box(0, max_health)` ranges** that span
  four orders of magnitude (`max_agent_health=1000`).  Normalising to `[0, 1]`
  (or at least logarithmically scaling) would make gradient signals more
  consistent.

- **`FlattenDictWrapper` destroys spatial structure.**  After flattening, the
  `(H, W, 18)` viewcone tensor becomes a 1D vector.  Policies must either
  re-apply the correct reshape internally or receive no structural prior at all.
  Providing a pre-flattened _and_ structured variant, or a `gym.spaces.Dict`
  pathway for CNN-based policies, would serve both use cases.

---

## 2. Action Space Design — 8 / 10

### What is done well

- **Decoupled agent and base actions per agent** via `{"agent": Discrete(7),
  "base": Discrete(3)}`.  Every agent simultaneously controls both a mobile unit
  and issues one command to the shared base, reducing policy complexity versus
  a centralised base controller.
- **Action masking** is built into the environment's `observe()` return.
  Illegal actions are intercepted by `Bomberman._execute_round()`, converted to
  `STAY` / `DO_NOTHING`, and penalised — preventing gradient explosion from
  out-of-distribution actions without requiring the policy to learn legality.
- **`Action`** and **`BaseAction`** are clean enums with meaningful names:
  `FORWARD`, `ATTACK`, `ITEM`, `CRAFT_SCOUT`, etc.

### Areas for improvement

- **AEC ordering creates intra-round information asymmetry.**  In PettingZoo's
  AEC model, agents observe the state and act one by one within a single round.
  Later-acting agents see the results of earlier agents' movements; earlier
  agents cannot anticipate later ones.  In a competitive 4-agent scenario this
  can systematically advantage or disadvantage teams based on agent index.
  Consider randomising agent order at the start of each round (currently the
  order is fixed by `AgentSelector`).

- **No continuous or hierarchical action option.**  Discrete-only is correct for
  a grid world, but providing a thin adapter for algorithms that expect a flat
  `Discrete(N)` combined action (rather than the dict) would improve library
  compatibility.

---

## 3. Reward Function — 6 / 10

### What is done well

- **14 configurable reward terms** via `RewardsConfig` in `config.py`.
  Practitioners can tune every signal without touching code.
- **Dense movement feedback**: `agent_collide_wall`, `agent_collide_agent`,
  `step_penalty`, `stationary_penalty` provide immediate gradient for basic
  navigation.
- **Differential step/idle penalties** (`step_penalty` vs. `stationary_penalty`)
  encourage activity without mandating aimless movement.
- **Combat attribution**: `attack_damage * damage_dealt` plus `attack_kill`
  bonus give the attacker proportional credit.  The victim's team receives a
  corresponding `destroyed_team_entity` penalty.
- **Terminal rewards** (`own_base_destroyed`, `destroy_enemy_base`) are
  symmetric and large enough to make base preservation a clear objective.

### Areas for improvement

- **`destroy_enemy_agent` in `RewardsConfig` is defined but never applied.**
  `_apply_combat_rewards` uses `attack_kill` for the attacker but does not
  apply `destroy_enemy_agent`.  This is either dead config or a missing reward
  path — either way it should be resolved.

- **Mission collection reward (`collect_mission`) fires on pickup, not on
  deposit.**  The ARCHITECTURE.md and entity docstrings describe missions as
  a carry-and-deposit mechanic ("agent carries to base for deposit reward"),
  but the current reward fires immediately on tile pickup.  If missions are
  genuinely carry-and-deposit, the reward should trigger on deposit; if they
  are immediate, the documentation should be corrected.

- **No exploration signal.**  On a procedurally generated 16×16 maze, the map
  is initially completely unknown.  Without a curiosity or visit-count bonus,
  agents may learn to camp near the base rather than explore.  A small reward
  for first-visit of each tile (from the agent's perspective) or for reducing
  team fog-of-war would help.

- **Reward scale is not normalised.**  Values range from −50 (`own_base_destroyed`)
  to +50 (`destroy_enemy_base`) with step penalties at −0.01.  Many modern RL
  libraries clip or normalise returns; an unnormalised scale that spans four
  orders of magnitude can destabilise value learning.

- **`truncation` reward defaults to 0.** A draw (both teams survive to `num_iters`)
  is identical to an in-progress step from the reward perspective.  Explicitly
  rewarding survival or adding a resource-differential tie-breaker would make
  the truncation outcome learnable.

---

## 4. Episode Structure — 8 / 10

### What is done well

- **Procedural map generation with seed control** (`ArenaGenerator.generate_walls`
  cached per seed, `generate_episode` fresh each reset).  Same seed → same walls,
  different episodes → different entity scatter.  This is the correct balance for
  curriculum learning (fix layout) vs. generalisation (vary scatter).
- **10-phase round execution** is cleanly phased and documented:
  movement → attack → damage → craft → scout → collect → upkeep → termination.
  Phases are independently testable.
- **Termination vs. truncation** are correctly separated (`terminations` dict
  vs. `truncations` dict), compliant with the PettingZoo AEC spec.
- **`num_iters` is configurable** — short episodes for debugging, long episodes
  for hard tasks.
- **Wall caching** means consecutive same-seed resets are fast (no maze
  re-generation).

### Areas for improvement

- **Debug `print` / `time` statements are live in production code.**
  `gridworld_v2.py:_execute_round` contains multiple `print("--- STEP START ---")`
  and per-phase timing prints.  These pollute stdout during training and slow
  down every step.  They should be replaced with `logging.debug(...)` or removed.

- **`novice` config flag is declared but its effect is not apparent.**
  `EnvConfig.novice: bool = False` exists in the schema but is not referenced in
  `dynamics.py` or `gridworld_v2.py`.  Either wire it or remove it.

---

## 5. Multi-Agent Design — 7 / 10

### What is done well

- **Competitive two-team structure** with symmetric starting conditions
  (`_spread_starting_positions` maximises inter-team separation).
- **Shared team infrastructure** (base, beacons, scouts) creates meaningful
  cooperative sub-tasks within each team.
- **Protocol-based dispatch** (`Defender`, `Vision`, `Healer`) means game phases
  operate on protocols, not concrete types.  Adding a new entity that heals, for
  instance, requires only implementing `Healer` — no changes to the upkeep loop.
- **`EntityRegistry.type_index`** keys on protocol classes (via MRO inclusion),
  making `_alive_of_type(Defender)` correct without manually listing concrete types.

### Areas for improvement

- **Centralised base action creates partial observability mismatch.**  Each
  agent independently submits a `base` action, but only one action per round
  can actually be executed.  There is no clear rule stated in the code for which
  agent's base action "wins" when multiple agents issue conflicting craft commands.
  This should be documented or enforced (e.g. priority queue, voting).

- **Credit assignment for team-level events is coarse.**  When a base is
  destroyed, all surviving agents on the losing team get `own_base_destroyed`.
  Dead agents (already terminated) receive nothing.  The winning team gets
  `destroy_enemy_base` per surviving agent, regardless of individual contribution.
  Shaped intermediate rewards for base defence (e.g. healing the base, blocking
  attacker paths) would reduce this sparseness.

- **`num_teams` beyond 2 is untested territory.**  The config allows `num_teams`
  to be arbitrary, but team-win logic and reward attribution assume binary
  winner/loser semantics.  Multi-team (> 2) play would need explicit changes.

---

## 6. Reproducibility & Testability — 6 / 10

### What is done well

- **Seed-deterministic episodes**: `reset(seed=N)` → `Dynamics.reset(N)` →
  `np_random(N)` → deterministic maze and entity scatter.
- **`test.py`** provides an interactive keyboard harness for manual episode
  inspection.
- **Configuration serialisation** (`save_config`, `load_config`) enables exact
  experiment reproduction from a YAML file.

### Areas for improvement

- **No automated test suite** (`pytest`, `unittest`).  The only test mechanism
  is `test.py` (manual / interactive).  Critical paths — observation shape,
  reward event coverage, registry index consistency — are untested
  programmatically.  A minimal pytest suite covering at least:
  - `reset()` produces valid observation shapes
  - `step()` does not raise on all-STAY actions
  - `entity.destroy()` is correctly reflected in all registry indexes
  - observation space bounds match actual observation values
  would significantly raise confidence.

- **`VisionSystem` has an open TODO** (`dynamics.py:80`: "this seems to be
  poorly used, and even more poorly thought out") indicating a known design
  uncertainty.  LOS is a critical component of partial observability — any
  correctness issues here propagate silently into training.

- **No benchmark / baseline.**  Without a published random-policy baseline or
  reference reward curve, it is hard to tell whether any training run is making
  progress or is stuck.

---

## 7. Code Quality & Extensibility — 9 / 10

### What is done well

- **Strict separation of concerns**: `Bomberman` owns API, `Dynamics` owns state,
  `Renderer` owns display, `EntityRegistry` owns storage.  No cross-layer leakage.
- **`entities/` package split** cleanly layers entity concerns:
  `base → protocols → geometry → dynamic/static → registry → __init__`.
  No circular imports.
- **OmegaConf structured config** catches typos at load time (structured schema
  validation), is serialisable to YAML, and flows top-down from a single root.
- **Protocol-based polymorphism** (`Attacker`, `Defender`, etc.) means new
  entity types can be added with zero changes to game loop code.
- **`Query` builder** on `EntityRegistry` provides composable, lazy multi-criteria
  lookups without bespoke accessor proliferation.
- **`ArenaGenerator` two-stage design** (cached wall gen + fresh scatter) is
  a well-motivated performance trade-off, clearly documented.
- The codebase is consistently docstring'd with design rationale, not just
  parameter names.

### Areas for improvement

- **Debug `print` statements in `gridworld_v2.py`.**  `_execute_round` has 8+
  `print()`/`time.time()` calls.  These are training-time noise and should be
  `logging.debug(...)`.

- **`dynamics.py` docstring** still references `entities.py` (singular file)
  rather than the `entities/` package — minor staleness.

---

## 8. Performance — 6 / 10

### What is done well

- **`EntityRegistry.pos_index`** is a pre-allocated 2D list-of-sets.  Positional
  lookups are O(1) with no hashing overhead.
- **`type_index`** is keyed by every MRO class at add-time (O(K) write, O(1)
  read for any protocol or concrete type).
- **Wall caching** avoids expensive maze re-generation on same-seed resets.
- **`VisionSystem`** computes team-wide LOS once per round and shares results
  across all observation builds.

### Areas for improvement

- **Per-step timing prints confirm bottlenecks.**  The `print("Time taken for
  render phase: …")` output in `_execute_round` shows the render and observation
  phases as notable costs.  Both are currently computed for *all* agents even
  if many are dead; skipping terminated agents' observations would be a quick win.

- **`build_agent_viewcone` and `build_radius_view`** rebuild full numpy arrays
  on every call.  For large grids and many beacons/scouts, this is significant.
  Pre-allocating reusable buffers (write-to rather than allocate-new) would
  reduce GC pressure.

- **`supercover_line` (LOS check)** is called O(agents × visible_cells) per
  round.  At grid_size=16 with 4 agents and wide viewcones this is manageable,
  but at grid_size=32+ with beacons contributing additional radius views, LOS
  becomes the dominant cost.  A pre-computed LOS table keyed on `(from_tile,
  to_tile, wall_set_hash)` would trade memory for speed if scaling is needed.

- **No profiling baseline or benchmark.**  Without a steps/second number at
  the default config, it is impossible to measure regression between changes.

---

## Recommendations by Priority

| Priority | Recommendation | Effort |
|---|---|---|
| **High** | Remove / replace debug `print`/`time` statements in `_execute_round` | Low |
| **High** | Add a minimal `pytest` suite covering observation shape and registry integrity | Medium |
| **High** | Clarify mission collect-on-pickup vs. deposit semantics and align reward + docs | Low |
| **High** | Wire or remove the unused `destroy_enemy_agent` reward term | Low |
| **Medium** | Normalise `direction` (one-hot) and `step` (float ratio) in observations | Low |
| **Medium** | Normalise `health` and `base_health` to [0, 1] in observation space | Low |
| **Medium** | Randomise agent turn order each round to eliminate AEC ordering bias | Low |
| **Medium** | Document (or enforce) base-action conflict resolution when multiple agents craft | Low |
| **Medium** | Resolve or clarify the `VisionSystem` TODO in `dynamics.py` | Medium |
| **Low** | Add exploration shaping (first-visit bonus or fog-of-war reduction reward) | Medium |
| **Low** | Add a steps/second benchmark to catch performance regressions | Low |
| **Low** | Wire the `novice` config flag or remove it | Low |
