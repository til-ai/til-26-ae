"""entities/static.py — Stationary collectible / environmental entities.

Entities defined here include:
- Mission / Recon — one-shot collectible reward tiles.
- Resource — collectible that credits the team resource pool.
- DestructibleWall — cell-filling Defender entity that blocks movement and
  can be destroyed by a bomb blast.
- AttackIntent — data class for queued attacks (not an Entity).

Imports: base.py, protocols.py.
"""

from dataclasses import dataclass

from til_environment.entities.base import Entity
from til_environment.entities.protocols import Item


# ---------------------------------------------------------------------------
# Attack intent – a queued attack produced by an Attacker
# ---------------------------------------------------------------------------
@dataclass
class AttackIntent:
    """A single attack that will be applied in the damage phase."""

    attribute_rewards: str
    attacker_id: str
    defender_id: str
    damage: float


# ---------------------------------------------------------------------------
# Mission
# ---------------------------------------------------------------------------
@dataclass
class Mission(Entity, Item):
    """Collectible objective on the grid.

    Awards ``reward_value`` to the agent that collects it (via reward hook on
    ``collect``), then destroys itself.
    """

    reward_value: float = 5.0
    difficulty: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.team is not None:
            self.team = None

    def collect(self, team, agent_id):
        super().collect(team)
        self.use(agent_id)
        self.destroy()


# ---------------------------------------------------------------------------
# Recon
# ---------------------------------------------------------------------------
@dataclass
class Recon(Entity, Item):
    """Lightweight collectible recon token on the grid."""

    reward_value: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.team is not None:
            self.team = None

    def collect(self, team, agent_id):
        super().collect(team)
        self.use(agent_id)
        self.destroy()


# ---------------------------------------------------------------------------
# Resource
# ---------------------------------------------------------------------------
@dataclass
class Resource(Entity, Item):
    """
    Collectible resource tile.

    When collected, credits ``amount`` to the collecting agent's team
    resource pool (handled by ``Dynamics``).  The value is expressed as a
    ratio against ``resources.bomb_cost``.
    """

    amount: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        if self.team is not None:
            self.team = None

    def collect(self, team, agent_id):
        super().collect(team)
        self.use(agent_id)
        self.destroy()
        return self.amount
