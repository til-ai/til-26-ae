"""
entities/protocols.py — Structural trait protocols for phased dispatch.

Pure structural typing — no game logic.  No imports from any other
entities/ submodule; concrete entity files import from here.

^
thanks claude

if you're wondering why all these protocols exist, and why in general this repo is so heavily
typed and OOP'd its because the scope of this env used to be significantly larger and the env specs
more uncertain. Hence protocol maxxed. There are like 5 others that are now gone :skull:
"""

from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from til_environment.entities.geometry import AttackType, VisionType
    from til_environment.entities.static import AttackIntent


# ---------------------------------------------------------------------------
# Trait protocols – structural subtyping for phased dispatch
# ---------------------------------------------------------------------------
@runtime_checkable
class Attacker(Protocol):
    """Entity that can produce attack intents during the damage phase."""

    attack: float
    attack_type: "AttackType"
    attribute_rewards: str

    @property
    def attack_power(self) -> float:
        return self.attack

    def get_attack_cells(self) -> list[np.ndarray]: ...

    def attack_sideeffects(self):
        """Apply any side effects of the attack (e.g. self-destruction)."""
        ...

    def trigger_attack(self):
        """Fire side effects and return the attack's raw damage value."""
        self.attack_sideeffects()
        return self.attack_power


@runtime_checkable
class Defender(Protocol):
    """Entity that can receive damage during the damage phase."""

    health: float
    max_health: float

    def receive_damage(
        self, amount: float, intent: "AttackIntent | None" = None
    ) -> float:
        """Apply damage.  Return actual damage dealt after mitigation.

        The ``intent`` carries the attacker id so event wrappers on this
        method can attribute attacker-side rewards.
        """
        ...


@runtime_checkable
class Vision(Protocol):
    """Entity that contributes vision to its team's shared view."""

    vision_type: "VisionType"

    def get_visible_cells(self) -> list[tuple[int, int]]:
        """Return the world-coordinate tiles this entity can see this tick."""
        ...


@runtime_checkable
class Timed(Protocol):
    """Entity whose lifecycle is governed by a tick-down timer.

    The owning ``Dynamics`` decrements ``timer`` by one each upkeep phase.
    When the timer reaches zero, the entity is considered "expired".
    """

    timer: int

    def tick_timer(self) -> int:
        """Decrement the timer by one and return the new value."""
        ...

    @property
    def expired(self) -> bool:
        """True if the timer has hit zero (or below)."""
        ...


@runtime_checkable
class Item(Protocol):
    """Collectible on-the-ground entity.

    Concrete subclasses (Mission, Recon, Resource) each implement their own
    ``collect(team, agent_id)`` method with the type-specific logic (e.g.
    Resource credits the team pool, Mission/Recon award a flat reward).
    """

    team: str | None = None
    used_by_id: str | None = None

    def owned(self) -> bool:
        return self.team is not None

    def collect(self, team):
        if self.team is not None:
            raise ValueError(f"{type(self)} already owned by team {self.team}")
        self.team = team

    def used(self) -> bool:
        return self.used_by_id is None

    def use(self, agent_id: str):
        if self.used_by_id is not None:
            raise ValueError(f"Item already used by agent {self.used_by_id}")
        self.used_by_id = agent_id


@runtime_checkable
class ExternalSideEffect(Protocol):
    """Entity that holds an external callable and fires it as a side effect."""

    def register_external_fn(self, fn: Callable) -> None:
        """Store a callable to be invoked as the external side effect."""
        ...

    def apply_external_sideeffects(self) -> None:
        """Invoke the registered external callable with entity-specific args."""
        ...
