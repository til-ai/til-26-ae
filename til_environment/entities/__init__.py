"""entities/ — Entity package for TIL Bomberman (Bomberman-scoped)."""

from til_environment.entities.base import Entity, EntityStatus
from til_environment.entities.protocols import (
    Attacker,
    Defender,
    ExternalSideEffect,
    Item,
    Timed,
    Vision,
)
from til_environment.entities.geometry import (
    AttackType,
    RadiusAttack,
    SkewVision,
    SquareVision,
    VisionType,
    _rotate,
)
from til_environment.entities.static import (
    AttackIntent,
    Mission,
    Recon,
    Resource,
)
from til_environment.entities.dynamic import Agent, Base, Bomb
from til_environment.entities.registry import EntityRegistry

__all__ = [
    # base
    "Entity",
    "EntityStatus",
    # protocols
    "Attacker",
    "Defender",
    "ExternalSideEffect",
    "Item",
    "Timed",
    "Vision",
    # geometry
    "AttackType",
    "RadiusAttack",
    "SkewVision",
    "SquareVision",
    "VisionType",
    "_rotate",
    # static
    "AttackIntent",
    "Mission",
    "Recon",
    "Resource",
    # dynamic
    "Agent",
    "Base",
    "Bomb",
    # registry
    "EntityRegistry",
]
