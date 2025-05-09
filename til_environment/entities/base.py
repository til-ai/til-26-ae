"""entities/base.py — Root entity class and lifecycle enum.

No imports from any other entities/ submodule.  All other submodules import
from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Entity status enum
# ---------------------------------------------------------------------------
class EntityStatus(IntEnum):
    """Lifecycle states shared by all entities."""

    ACTIVE = auto()
    DESTROYED = auto()


# ---------------------------------------------------------------------------
# Base entity
# ---------------------------------------------------------------------------
@dataclass
class Entity:
    """Abstract base for every object that occupies a grid cell.

    Attributes
    ----------
    entity_id : str
        Unique identifier, e.g. ``"agent_0"``, ``"base_team1"``.
    team : int | None
        Owning team index.  ``None`` for neutral entities.
    position : np.ndarray
        (x, y) grid coordinate.
    status : EntityStatus
        Whether the entity is still active on the board.
    """

    entity_id: str
    position: np.ndarray
    status: EntityStatus = EntityStatus.ACTIVE
    team: int | None = None

    def __post_init__(self):
        # Back-reference to the owning registry, set by EntityRegistry.add().
        # Stored as a plain attribute (not a dataclass field) so it is
        # excluded from __init__, __repr__, and __eq__.
        object.__setattr__(self, "_registry", None)

    def __setattr__(self, name: str, value) -> None:
        if name == "position":
            # Notify the registry so pos_index stays consistent.
            reg = (
                object.__getattribute__(self, "_registry")
                if "_registry" in self.__dict__
                else None
            )
            if reg is not None and hasattr(self, "position"):
                old = self.position
                old_pos = (int(old[0]), int(old[1]))
                object.__setattr__(self, name, value)
                reg._on_position_changed(self, old_pos)
                return
        object.__setattr__(self, name, value)

    # -- convenience --------------------------------------------------------

    @property
    def alive(self) -> bool:
        return self.status == EntityStatus.ACTIVE

    def destroy(self) -> None:
        """Mark the entity as destroyed and notify the registry."""
        self.status = EntityStatus.DESTROYED
        reg = self.__dict__.get("_registry")
        if reg is not None:
            reg._on_destroyed(self)

    def distance_to(self, other: "Entity") -> float:
        """Euclidean distance to another entity."""
        return float(np.linalg.norm(self.position - other.position))

    def manhattan_to(self, other: "Entity") -> int:
        """Manhattan distance to another entity."""
        return int(np.sum(np.abs(self.position - other.position)))
