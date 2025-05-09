"""entities/geometry.py — Pluggable geometry callables for attack and vision.

No entity state, no registry.  Imports only numpy and stdlib typing.
No imports from any other entities/ submodule.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


# ---------------------------------------------------------------------------
# Shared rotation helper
# ---------------------------------------------------------------------------
def _rotate(offset: np.ndarray, direction: int) -> np.ndarray:
    """Rotate a RIGHT-relative offset by *direction* × 90° clockwise."""
    o = offset.copy()
    for _ in range(direction):
        o = np.array([-o[1], o[0]])
    return o


# ---------------------------------------------------------------------------
# Attack types – pluggable geometry for attack cell computation
# ---------------------------------------------------------------------------
class AttackType(Protocol):
    """Callable protocol: given attacker position and facing direction, return
    the list of world-coordinate cells that can be hit.
    """

    def __call__(self, position: np.ndarray, direction: int) -> list[np.ndarray]: ...


class RadiusAttack:
    """Square-radius blast hitting every cell within Chebyshev radius ``r``."""

    def __init__(self, r: int) -> None:
        self.r = r
        self._offsets = [
            np.array([dx, dy]) for dx in range(-r, r + 1) for dy in range(-r, r + 1)
        ]

    def __call__(self, position: np.ndarray, direction: int) -> list[np.ndarray]:
        return [position + off for off in self._offsets]


# ---------------------------------------------------------------------------
# Vision types – pluggable geometry for area-of-sight computation
# ---------------------------------------------------------------------------
class VisionType(Protocol):
    """Callable: given a world position and facing direction, return the list
    of ``(x, y)`` tile coordinates this entity can see.
    """

    def __call__(
        self, position: np.ndarray, direction: int
    ) -> list[tuple[int, int]]: ...

    @property
    def radius(self) -> int: ...


class SquareVision:
    """Omnidirectional square vision covering a ``(2n+1) x (2n+1)`` area."""

    def __init__(self, n: int) -> None:
        self.n = n
        self._offsets = [(dx, dy) for dx in range(-n, n + 1) for dy in range(-n, n + 1)]

    @property
    def radius(self) -> int:
        return self.n

    def __call__(self, position: np.ndarray, direction: int) -> list[tuple[int, int]]:
        px, py = int(position[0]), int(position[1])
        return [(px + dx, py + dy) for dx, dy in self._offsets]


class SkewVision:
    """Direction-aware vision: a ``(2n+1) x (2n+1)`` square extended by ``m``
    extra tiles deep in the facing direction.
    """

    def __init__(self, n: int, m: int) -> None:
        self.n = n
        self.m = m
        self._base_offsets = [
            (dx, dy) for dx in range(-n, n + 1) for dy in range(-n, n + 1)
        ]
        self._ext_offsets = [
            (dx, dy) for dx in range(n + 1, n + m + 1) for dy in range(-n, n + 1)
        ]

    @property
    def radius(self) -> int:
        return self.n

    def __call__(self, position: np.ndarray, direction: int) -> list[tuple[int, int]]:
        px, py = int(position[0]), int(position[1])
        cells: list[tuple[int, int]] = []
        for raw_dx, raw_dy in self._base_offsets + self._ext_offsets:
            rot = _rotate(np.array([raw_dx, raw_dy]), direction)
            cells.append((px + int(rot[0]), py + int(rot[1])))
        return cells
