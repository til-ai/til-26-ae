# Using the `til_environment` package

This page introduces the basics of using the `til_environment` package to run your AE models.

**Contents**
1. [Using the `til_environment` package](#using-the-til_environment-package)
   1. [Setup](#setup)
   2. [Usage \& Running the environment](#usage--running-the-environment)
      1. [Observation Space](#observation-space)
      2. [Action space](#action-space)
   3. [Configuration](#configuration)
      1. [Key config sections](#key-config-sections)
      2. [Reward shaping](#reward-shaping)
   4. [Use during training](#use-during-training)
      1. [Writing an environment wrapper](#writing-an-environment-wrapper)
   5. [Package structure](#package-structure)
      1. [Key concepts](#key-concepts)

---

## Setup

**Python 3.10 is required.**

Clone the `til-26-ae` repo, then create and activate a virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

Or, if you prefer [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync   # creates .venv and installs all dependencies automatically. goated
```

---

## Usage & Running the environment

You can import the raw environment as such:

```python
from til_environment.bomberman_env import Bomberman
```

This environment follows the [PettingZoo](https://pettingzoo.farama.org/) AEC (Agent-Environment-Cycle) API. Each call to `env.step()` logs an agent's action; a full round executes after the last agent in the cycle submits its action. As such the overall running loop is functionally identical to that of PettingZoo:


```python
from til_environment.bomberman_env import Bomberman

env = Bomberman()

env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # Insert your policy here. or dont. like why dont you throw random at it :)
        action = env.action_space(agent).sample()

    env.step(action)

env.close()
```

> **Note:** Termination should never fire off. In this game, your agents have been (unfortunately) cursed with immortality. Recall agents only freeze after reaching 0 health, then respawn where they died. Enjoy playing the full 200 iterations!

### Observation Space

We've significantly expanded the observation space from last year. This time, we give a series of binary channels (with significant sparsity, i.e many 0s, few 1s, per channel). For a full description of what each channel represents, see [observation.py](til-26-ae/til_environment/observation.py).

| Key              | Shape / type                 | Description                                                                                                |
| ---------------- | ---------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `agent_viewcone` | `float32 [7 × 5 × 25]`       | Viewcone centred on this agent (widths of 4 ahead, 2 behind, 2 to the side, agent at the center inclusive) |
| `base_viewcone`  | `float32 [S × S × 25]`       | Square view centred on this agent's base (S = 2×vision_radius+1)                                           |
| `direction`      | `Discrete(4)`                | Agent's facing direction (0=RIGHT, 1=DOWN, 2=LEFT, 3=UP)                                                   |
| `location`       | `uint8 [2]`                  | Agent's (x, y) grid position                                                                               |
| `base_location`  | `uint8 [2]`                  | Team base (x, y) grid position                                                                             |
| `health`         | `float32 [1]`                | Agent current HP                                                                                           |
| `frozen_ticks`   | `Discrete(freeze_turns+1)`   | Remaining freeze steps (0 = active)                                                                        |
| `base_health`    | `float32 [1]`                | Team base current HP                                                                                       |
| `team_resources` | `float32 [1]`                | Accumulated resource ratio for this agent's team                                                           |
| `team_bombs`     | `Discrete(max_team_bombs+1)` | Bomb stockpile for this agent's team                                                                       |
| `step`           | `Discrete(num_iters+1)`      | Current step index                                                                                         |
| `action_mask`    | `uint8 [6]`                  | Binary mask — 1 = action is legal for this agent this step                                                 |

### Action space

Each agent has 6 discrete actions:

| Index | Name         | Description                                                          |
| ----- | ------------ | -------------------------------------------------------------------- |
| 0     | `FORWARD`    | Move one cell in the direction the agent is facing                   |
| 1     | `BACKWARD`   | Move one cell opposite to facing direction                           |
| 2     | `LEFT`       | Turn 90° counter-clockwise                                           |
| 3     | `RIGHT`      | Turn 90° clockwise                                                   |
| 4     | `STAY`       | Do not move                                                          |
| 5     | `PLACE_BOMB` | Place a bomb at the agent's current cell (requires `team_bombs > 0`) |

---

## Configuration

The default config may be obtained via: 

```python
from til_environment.config import default_config
cfg = default_config()
```

All tunable environment parameters live in a single OmegaConf config tree. During evaluation for qualifiers, we will use this default configuration: However, you are free to modify it for your training purposes. You can simply modify the YAML file, or do it code-wise as follows:

```python
from til_environment.config import default_config
from omegaconf import OmegaConf

cfg = default_config()
# override specific values
cfg = OmegaConf.merge(cfg, {"env": {"grid_size": 10, "num_teams": 2}})
# but like why. i love writing more config files
```

The environment is highly configurable, if you want to explore any ideas, create your own custom reward hooks/events or mutate existing ones, heck even change the environment fundementally for your model training / DP development. Who am I to stop you from trying lots of things out! Not me, probably the time you have. Anyways this ae is quite sandboxable, as long as you acknowledge the default config is what the evaluatons / qualifiers / finals will run. 

### Key config sections

**`env`**

| Key                  | Default | Description                                           |
| -------------------- | ------- | ----------------------------------------------------- |
| `grid_size`          | 16      | Grid side length                                      |
| `num_teams`          | 6       | Number of competing teams (one agent + one base each) |
| `num_iters`          | 200     | Maximum steps before truncation                       |
| `novice`             | `false` | If `true`, fixes the map layout (same every episode)  |
| `render_mode`        | `null`  | `"human"`, `"rgb_array"`, or `null`                   |
| `tile_respawn_steps` | 40      | * Max Steps before a collected tile reappears. Note this is randomly generated across the board based on perlin noise! Read the code / til-26-ae README for more info               |

**`entities.agent`**

| Key                     | Default | Description                                    |
| ----------------------- | ------- | ---------------------------------------------- |
| `health` / `max_health` | 60      | Agent HP                                       |
| `freeze_turns`          | 3       | Iterations the agent is frozen after reaching 0 HP |

**`entities.bomb`**

| Key            | Default | Description                   |
| -------------- | ------- | ----------------------------- |
| `attack`       | 20.0    | Damage per blast cell         |
| `blast_radius` | 2       | Radius of explosion |
| `timer`        | 4       | Steps until detonation       |

**`resources`**

| Key                  | Default | Description                                                          |
| -------------------- | ------- | -------------------------------------------------------------------- |
| `bomb_cost`          | 1.5     | Resource ratio required to convert into one bomb                     |
| `base_resource_rate` | 0.1     | Resources added to every team's pool per step (fixed, unconditional) |
| `starting_bombs`     | 3       | Bombs each team starts with                                          |

### Reward shaping

All reward values are in the config under `rewards`. As you may notice, several are left as 0: We do not penalize these actions, but leave them as open options for reward shaping (that is within your own training pipeline, modifying them in hopes of attaining different behaviour).


| Key                   | Default                    | Credited to                   | Description                                                                                   |
| --------------------- | -------------------------- | ----------------------------- | --------------------------------------------------------------------------------------------- |
| `agent_collide_wall`  | 0                          | Agent                         | Agent attempts to move into a wall                                                           |
| `agent_collide_agent` | 0                          | Agent                         | Agent attempts to move into another agent                                                     |
| `collect_mission`     | 5.0                        | Agent                         | Agent steps onto a mission tile                                                               |
| `collect_recon`       | 1.0                        | Agent                         | Agent steps onto a recon tile                                                                 |
| `attack_damage`       | 1.0× damage dealt (20 pts) | Attacker (-1.0× for Defender) | Bomb damages an agent or base; defender receives the equal-and-opposite penalty automatically |
| `attack_kill`         | 15.0                       | Attacker (flat)               | Bomb reduces an agent to 0 HP                                                                 |
| `destroy_wall`        | 0                          | Attacker                      | Bomb destroys a destructible wall                                                             |
| `destroy_enemy_base`  | 50.0                       | Attacker                      | Bomb destroys an enemy base                                                                   |
| `own_base_destroyed`  | -50.0                      | Defending agent               | This agent's base is destroyed                                                                |
| `step_penalty`        | 0                          | Agent                         | Applied every step                                                                            |
| `stationary_penalty`  | 0                          | Agent                         | Agent chose `STAY`                                                                            |
| `invalid_action`      | 0                          | Agent                         | Agent submitted a masked action                                                               |

> **Defenders are penalised automatically.** You do not need a separate config entry. For every `attack_damage` event, the damaged entity's controlling agent receives `-attack_damage × damage_dealt` in the same step.

---

## Use during training

### Writing an environment wrapper

Custom wrappers inherit from `BaseWrapper` in PettingZoo. You may want to write custom wrappers to modify the way observations are returned. A basic form of wrapping can be observed in the [basic_env function](til-26-ae/til_environment/bomberman_env.py)

```python
import functools
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType
from pettingzoo.utils.wrappers.base import BaseWrapper

class CustomWrapper(BaseWrapper[AgentID, ObsType, ActionType]):
    def __init__(self, env: AECEnv[AgentID, ObsType, ActionType]):
        super().__init__(env)

    def reset(self, seed=None, options=None):
        super().reset(seed, options)

    def step(self, action: ActionType):
        super().step(action)

    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)
        # modify obs here
        return obs

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return super().observation_space(agent)
```
Then simply wrap it around by passing the environment inside it. A basic example with some existing wrappers can be found at

For more wrappers see [PettingZoo Wrappers](https://pettingzoo.farama.org/api/wrappers/pz_wrappers/) and [SuperSuit](https://github.com/Farama-Foundation/SuperSuit).

---

## Package structure

```
til_environment/
├── bomberman_env.py        # Top-level PettingZoo AECEnv; basic_env() factory lives here
├── dynamics.py            # All game logic: phases, rewards, registry access
├── arena.py               # Maze generation (structural + destructible walls, tile placement)
├── config.py              # OmegaConf-based config tree; load_config() / default_config()
├── observation.py         # Build per-agent viewcone tensors; ViewChannel enum (25 channels)
├── actions.py             # ActionV2 enum, ActionMask (legal-move computation)
├── types.py               # Direction, Tile enums and shared type aliases
├── renderer.py            # Pygame renderer; HUD, replay recording
├── flatten_dict.py        # FlattenDictWrapper — flattens obs dict to 1-D array
├── helpers.py             # Miscellaneous utilities
├── play.py                # Interactive harness: launch with `uv run python -m til_environment.play`
│
├── entities/
│   ├── base.py            # Entity base class (entity_id, team, position, alive)
│   ├── dynamic.py         # Agent, Base, Bomb classes
│   ├── static.py          # Mission, Recon, Resource, DestructibleWall, AttackIntent
│   ├── protocols.py       # Structural interfaces: Attacker, Defender, Vision, Item, Timed, ExternalSideEffect
│   ├── geometry.py        # Blast/vision geometry helpers (RadiusAttack, SquareVision)
│   └── registry.py        # EntityRegistry — indexed container for all live entities
│
├── events/
    ├── base.py            # EmitRule dataclass + wrap_multi() — hooks entity methods to reward events
    └── rewards.py         # Rewards accumulator (step + episode); award() method

```

### Key concepts

**Agents and teams.** There is exactly one agent per team. Agent IDs are `"agent_0"`, `"agent_1"`, … `"agent_{N-1}"`. Each team also has one base.

**Death and respawn.** When an agent's HP reaches 0 it enters a *frozen* state for 3 steps — it stays on the grid but its entire action mask is zeroed. After the freeze it respawns at the same tile at full HP. Agents are never removed from the game; only step truncation ends an episode.

**Resource economy.** Each team has a shared resource pool (`team_resources`) and bomb stockpile (`team_bombs`). Resources accumulate passively every step at a fixed rate (`0.1`), and are also collected by stepping on resource tiles (`0.5`). Whenever `team_resources ≥ 1.5` the surplus auto-converts into bombs.

**Bombs.** A placed bomb counts down for `timer` steps then detonates in a cross-shaped blast of square radius `2` (so 2 + 1 + 2, where 1 is the center where the bomb is place). The blast damages agents, bases, and destroys destructible walls in range. Indestructible walls block the blast.

**Observation channels.** The viewcone tensor has 25 channels encoding visibility, wall edges (structural and destructible, per direction), tile types, ally/enemy entities, bomb presence and timers, and HP levels. Once more, see [`observation.py`](til-26-ae/til_environment/observation.py) (`ViewChannel` enum) for the full list.
