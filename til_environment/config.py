"""
config.py - Hierarchical OmegaConf-based configuration for the TIL Bomberman.

This module provides a **single source of truth** for every tuneable
parameter in the environment.

Design
------
The config tree mirrors the sub-system hierarchy::

    BombermanConfig
    ├── env          — top-level game rules (grid size, teams, iterations, …)
    ├── dynamics     — physics / game-logic tunables
    │   ├── arena    — maze generation
    │   └── vision   — viewcone shape
    ├── entities     — per-entity-type defaults
    │   ├── agent
    │   ├── base
    │   ├── bomb
    │   ├── mission
    │   ├── recon
    │   └── resource
    ├── resources    — per-team resource / bomb economy (ratios)
    ├── renderer     — display settings
    └── rewards      — reward shaping values
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf, DictConfig


# ═══════════════════════════════════════════════════════════════════════════
# env — top-level game rules
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class EnvConfig:
    """Top-level environment parameters."""

    grid_size: int = 16
    num_teams: int = 6
    num_iters: int = 200
    novice: bool = False
    render_mode: str | None = None
    tile_respawn_steps: int = 40


# ═══════════════════════════════════════════════════════════════════════════
# dynamics — physics / game-logic tunables
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class ArenaConfig:
    """Maze generation parameters."""

    wall_prob: float = 1.0
    wall_destructible_ratio: float = 0.4
    mission_prob: float = 0.30
    recon_prob: float = 0.40
    resource_prob: float = 0.30


@dataclass
class VisionConfig:
    """Viewcone shape definition."""

    left: int = 2
    right: int = 2
    behind: int = 2
    ahead: int = 4


@dataclass
class DynamicsConfig:
    """Game-logic tunables passed to the ``Dynamics`` engine."""

    arena: ArenaConfig = field(default_factory=ArenaConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)


# ═══════════════════════════════════════════════════════════════════════════
# entities — per-type default attribute values
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class AgentEntityConfig:
    """Default attributes for newly created Agents."""

    health: float = 60
    max_health: float = 60
    freeze_turns: int = 3


@dataclass
class BaseEntityConfig:
    """Default attributes for newly created Bases."""

    health: float = 100
    max_health: float = 100
    vision_radius: int = 3


@dataclass
class BombEntityConfig:
    """Default attributes for Bomb entities."""

    attack: float = 20.0
    blast_radius: int = 2
    timer: int = 4


@dataclass
class MissionEntityConfig:
    """Default attributes for Mission entities."""

    reward_value: float = 5.0
    difficulty: float = 1.0


@dataclass
class ReconEntityConfig:
    """Default attributes for Recon entities."""

    reward_value: float = 1.0


@dataclass
class ResourceEntityConfig:
    """Default attributes for Resource tile entities (amount is a ratio)."""

    amount: float = 0.5


@dataclass
class EntitiesConfig:
    """Collected entity defaults — one sub-config per entity type."""

    agent: AgentEntityConfig = field(default_factory=AgentEntityConfig)
    base: BaseEntityConfig = field(default_factory=BaseEntityConfig)
    bomb: BombEntityConfig = field(default_factory=BombEntityConfig)
    mission: MissionEntityConfig = field(default_factory=MissionEntityConfig)
    recon: ReconEntityConfig = field(default_factory=ReconEntityConfig)
    resource: ResourceEntityConfig = field(default_factory=ResourceEntityConfig)


# ═══════════════════════════════════════════════════════════════════════════
# resources — per-team resource / bomb economy
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class ResourcesConfig:
    """Per-team resource economy expressed as ratios.

    All values are floats; ``bomb_cost`` acts as the unit.  Resources
    accumulate in a shared team pool and auto-convert into bombs whenever
    the pool reaches ``bomb_cost``.

    ``max_team_resources`` and ``max_team_bombs`` are used as the upper
    bounds for the observation space declarations.
    """

    bomb_cost: float = 1.5
    base_resource_rate: float = 0.1
    starting_bombs: int = 3
    starting_resources: float = 0.0
    max_team_resources: float = 100.0
    max_team_bombs: int = 50


# ═══════════════════════════════════════════════════════════════════════════
# renderer — display settings
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class RendererConfig:
    """Settings for the pygame renderer."""

    window_size: int = 768
    debug: bool = False
    render_fps: int = 5
    replay_dir: str | None = None


# ═══════════════════════════════════════════════════════════════════════════
# rewards — reward shaping values
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class RewardsConfig:
    """Every reward-producing event and its default value."""

    agent_collide_wall: float = 0
    agent_collide_agent: float = 0
    collect_mission: float = 5.0
    collect_recon: float = 1.0
    collect_resource: float = 2.0
    attack_damage: float = 1.0
    attack_kill: float = 15.0
    destroy_wall: float = 0
    destroy_enemy_base: float = 50.0
    own_base_destroyed: float = -50.0
    step_penalty: float = 0
    stationary_penalty: float = 0
    invalid_action: float = 0
    truncation: float = 0


# ═══════════════════════════════════════════════════════════════════════════
# Root config
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class BombermanConfig:
    """Root configuration object."""

    env: EnvConfig = field(default_factory=EnvConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    entities: EntitiesConfig = field(default_factory=EntitiesConfig)
    resources: ResourcesConfig = field(default_factory=ResourcesConfig)
    renderer: RendererConfig = field(default_factory=RendererConfig)
    rewards: RewardsConfig = field(default_factory=RewardsConfig)


# ═══════════════════════════════════════════════════════════════════════════
# Factories
# ═══════════════════════════════════════════════════════════════════════════

_STRUCTURED = OmegaConf.structured(BombermanConfig)


def default_config() -> DictConfig:
    """Return a fresh ``BombermanConfig`` as an OmegaConf DictConfig."""
    return OmegaConf.structured(BombermanConfig)


def load_config(path: str | Path) -> DictConfig:
    """Load a YAML file and merge it onto the structured defaults."""
    schema = OmegaConf.structured(BombermanConfig)
    user = OmegaConf.load(path)
    merged = OmegaConf.merge(schema, user)
    return merged


def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Serialise a config to YAML on disk."""
    Path(path).write_text(OmegaConf.to_yaml(cfg))


def viewcone_tuple(cfg: VisionConfig | DictConfig) -> tuple[int, int, int, int]:
    """Return the ``(left, right, behind, ahead)`` tuple from a VisionConfig."""
    return (int(cfg.left), int(cfg.right), int(cfg.behind), int(cfg.ahead))
