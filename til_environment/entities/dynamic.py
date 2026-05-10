"""entities/dynamic.py — Mobile and stationary entities with rich mutable state.

These entities persist across multiple ticks, can be damaged, moved, or produce
effects.  Each explicitly inherits the trait protocols it satisfies.

Imports: base.py, protocols.py, geometry.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from til_environment.entities.base import Entity
from til_environment.entities.protocols import (
    Attacker,
    Defender,
    ExternalSideEffect,
    Timed,
    Vision,
)
from til_environment.entities.geometry import (
    AttackType,
    VisionType,
    SquareVision,
    RadiusAttack,
)

if TYPE_CHECKING:
    from til_environment.entities.static import AttackIntent

import numpy as np


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
@dataclass
class Agent(Entity, Defender, Vision):
    """A mobile agent controlled by a team's policy.

    Agents move on the grid, collect tiles, and place bombs.  Health can
    drop to zero — at which point the agent enters a "frozen" state for
    ``freeze_turns`` upkeeps, then respawns in place at full HP.  The agent
    is never removed from the grid (``destroy()`` is never called during
    normal gameplay).
    """

    health: float = 60.0
    max_health: float = 60.0
    direction: int = 0
    frozen_ticks: int = 0
    freeze_duration: int = 10
    vision_type: VisionType = field(default_factory=lambda: SquareVision(0))

    def __post_init__(self):
        super().__post_init__()

    @property
    def is_frozen(self) -> bool:
        return self.frozen_ticks > 0

    def get_visible_cells(self) -> list[tuple[int, int]]:
        """Vision protocol — delegates to ``self.vision_type``.

        The directional viewcone is applied by ``VisionSystem``; the
        default ``SquareVision(0)`` is a no-op sentinel.
        """
        return self.vision_type(self.position, self.direction)

    def receive_damage(
        self, amount: float, intent: "AttackIntent | None" = None
    ) -> float:
        """Apply damage.  If HP drops to 0, enter the frozen state.

        Unlike v1, we do NOT call ``destroy()`` — the agent persists on the
        grid so the real-world robot counterpart isn't pulled off the mat.
        Reward hooks (attack_damage, attack_kill) fire via event wrappers
        installed by ``Dynamics``.
        """
        if self.is_frozen or not self.alive:
            return 0.0
        effective = max(0.0, amount)
        before = self.health
        self.health = max(0.0, self.health - effective)
        return before - self.health

    def heal(self, amount: float) -> float:
        """Heal the agent, clamped to ``max_health``."""
        before = self.health
        self.health = min(self.max_health, self.health + amount)
        return self.health - before


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
@dataclass
class Base(Entity, Defender, Vision):
    """Stationary team structure.  The team loses if its base is destroyed."""

    health: float = 100.0
    max_health: float = 100.0
    vision_type: VisionType = field(default_factory=lambda: SquareVision(3))

    def get_visible_cells(self) -> list[tuple[int, int]]:
        return self.vision_type(self.position, 0)

    def receive_damage(
        self, amount: float, intent: "AttackIntent | None" = None
    ) -> float:
        effective = max(0.0, amount)
        before = self.health
        self.health = max(0.0, self.health - effective)
        if self.health <= 0 and self.alive:
            self.destroy()
        return before - self.health


# ---------------------------------------------------------------------------
# Bomb
# ---------------------------------------------------------------------------
@dataclass
class Bomb(Entity, Attacker, Timed, ExternalSideEffect):
    """Timed area-of-effect explosive placed by an agent.

    Placed directly onto the board via the ``PLACE_BOMB`` action (no
    inventory, no manufacture cost — the team's ``team_bombs`` counter is
    decremented by Dynamics).  Each upkeep tick decrements its ``timer``.
    Once expired, the detonation phase resolves the blast: every Defender
    within ``RadiusAttack(blast_radius)`` (including ``DestructibleWall``s)
    takes ``attack`` damage.  Edge walls (non-destructible) are unaffected.
    """

    attack: float = 30.0
    blast_radius: int = 3
    timer: int = 3
    attribute_rewards: str = ""  # set at placement to the placer's entity_id
    attack_type: AttackType = field(default_factory=lambda: RadiusAttack(3))
    _external_fn: Callable | None = field(default=None, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if (
            not isinstance(self.attack_type, RadiusAttack)
            or self.attack_type.r != self.blast_radius
        ):
            self.attack_type = RadiusAttack(self.blast_radius)
        # Placed bombs tick down once immediately in upkeep, so compensate.
        self.timer += 1

    def get_attack_cells(self) -> list[np.ndarray]:
        return self.attack_type(self.position, 0)

    def tick_timer(self) -> int:
        self.timer -= 1
        return self.timer

    @property
    def expired(self) -> bool:
        return self.timer <= 0

    # -- ExternalSideEffect protocol ----------------------------------------

    def register_external_fn(self, fn: Callable) -> None:
        self._external_fn = fn

    def apply_external_sideeffects(self) -> None:
        if self._external_fn is not None:
            self._external_fn(self.get_attack_cells())

    # -- Attacker protocol --------------------------------------------------

    def attack_sideeffects(self):
        """On detonation: fire external side effects, then destroy self."""
        if not self.alive:
            return
        self.apply_external_sideeffects()
        self.destroy()
