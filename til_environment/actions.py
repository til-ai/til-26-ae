"""
actions.py - Action definitions and action-mask system for Bomberman.

Defines:
* ``Action`` — agent actions (movement, stay, place_bomb).
* ``ActionGroup`` — categorical grouping for mutual-exclusion rules.
* ``ActionMask`` — mask builder enforcing per-tick legality.
"""

from __future__ import annotations

from enum import IntEnum, auto
from typing import TYPE_CHECKING, Callable
from til_environment.types import Direction
import numpy as np

if TYPE_CHECKING:
    from til_environment.dynamics import Dynamics
    from til_environment.entities import Agent


# ---------------------------------------------------------------------------
# Action groups (mutual-exclusion categories)
# ---------------------------------------------------------------------------
class ActionGroup(IntEnum):
    MOVEMENT = auto()  # FORWARD, BACKWARD, LEFT, RIGHT
    IDLE = auto()      # STAY
    COMBAT = auto()    # PLACE_BOMB


# ---------------------------------------------------------------------------
# Action enum
# ---------------------------------------------------------------------------
class Action(IntEnum):
    """All actions an agent can take in a single step."""

    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4
    PLACE_BOMB = 5

    @property
    def group(self) -> ActionGroup:
        _GROUP_MAP = {
            Action.FORWARD: ActionGroup.MOVEMENT,
            Action.BACKWARD: ActionGroup.MOVEMENT,
            Action.LEFT: ActionGroup.MOVEMENT,
            Action.RIGHT: ActionGroup.MOVEMENT,
            Action.STAY: ActionGroup.IDLE,
            Action.PLACE_BOMB: ActionGroup.COMBAT,
        }
        return _GROUP_MAP[self]


NUM_ACTIONS_V2 = len(Action)


# ---------------------------------------------------------------------------
# ActionMask
# ---------------------------------------------------------------------------
class ActionMask:
    """
    Builds and validates per-agent action masks.
    
    If you properly read and discovered the code, this should render agent_collide_wall and agent_collide_agent
    """

    def __init__(self, dynamics: "Dynamics") -> None:
        self._dyn = dynamics

    def build(self, agent_id: str) -> np.ndarray:
        """Return action mask for *agent_id*.  Shape ``(NUM_ACTIONS_V2,)``."""
        from til_environment.entities import Agent as AgentCls

        agent = self._dyn.registry.get(agent_id)
        if not isinstance(agent, AgentCls) or not agent.alive:
            return np.zeros(NUM_ACTIONS_V2, dtype=np.int8)
        if agent.is_frozen:
            # Frozen agents may only STAY — all-zeros would confuse AEC wrappers
            # that interpret a fully-masked agent as "dead, send None".
            mask = np.zeros(NUM_ACTIONS_V2, dtype=np.int8)
            mask[Action.STAY] = np.int8(1)
            return mask

        mask = np.zeros(NUM_ACTIONS_V2, dtype=np.int8)
        for action in Action:
            validator = self._VALIDATORS.get(action)
            if validator is not None:
                mask[action.value] = np.int8(validator(self, agent))
            else:
                mask[action.value] = np.int8(1)
        return mask

    def validate(self, agent_id: str, action: int) -> bool:
        mask = self.build(agent_id)
        return bool(mask[action])

    # -- per-action validators ---------------------------------------------

    def _can_forward(self, agent: "Agent") -> bool:
        direction = Direction(agent.direction)
        return not self._dyn._blocks_movement(agent, direction)

    def _can_backward(self, agent: "Agent") -> bool:
        direction = Direction((agent.direction + 2) % 4)
        return not self._dyn._blocks_movement(agent, direction)

    def _can_left(self, agent: "Agent") -> bool:
        return True

    def _can_right(self, agent: "Agent") -> bool:
        return True

    def _can_stay(self, agent: "Agent") -> bool:
        return True

    def _can_place_bomb(self, agent: "Agent") -> bool:
        """PLACE_BOMB is legal only if the team has at least one bomb available."""
        return self._dyn.team_bombs.get(agent.team, 0) > 0

    _VALIDATORS: dict[Action, "_ValidatorFn"] = {
        Action.FORWARD: _can_forward,
        Action.BACKWARD: _can_backward,
        Action.LEFT: _can_left,
        Action.RIGHT: _can_right,
        Action.STAY: _can_stay,
        Action.PLACE_BOMB: _can_place_bomb,
    }


_ValidatorFn = Callable[["ActionMask", "Agent"], bool]
