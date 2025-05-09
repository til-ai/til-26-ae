"""events/rewards.py — Reward accumulation as an Event subclass.

``Rewards`` is a specialised ``Event`` that maintains two parallel
accumulators:

* ``_step``     — rewards received during the current round, cleared by
                  ``clear_step()`` at the end of each ``gridworld`` step.
* ``_episode``  — rewards received since the last ``reset()``, kept for
                  episode-level logging / scoring.

All writes go through ``award()``, which looks up the reward value from
the config (``cfg.rewards.<event_type>``), applies an optional multiplier,
and updates both accumulators.  ``award()`` is also what ``Event.wrap``
ends up calling via the ``emit`` override below, so wrapped entity
methods automatically feed the same pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from til_environment.events.base import Event


class Rewards(Event):
    """Reward bookkeeping for the Bomberman environment.

    Parameters
    ----------
    cfg : DictConfig
        Full environment config.  ``cfg.rewards`` is stored for lookups.
    log_path : str | Path | None
        Optional path to append reward events to (human-readable log).
    """

    def __init__(self, cfg: DictConfig, log_path: str | Path | None = None) -> None:
        super().__init__(name="rewards", log_path=log_path)
        self.config = cfg.rewards
        self._step: dict[str, float] = {}
        self._episode: dict[str, float] = {}

    # ── lifecycle ─────────────────────────────────────────────────────
    def reset(self, agent_ids: list[str] | None = None) -> None:
        """Clear both accumulators at the start of an episode.

        If ``agent_ids`` is given, the step/episode dicts are pre-seeded
        with zeros for those ids so callers can iterate without KeyErrors.
        """
        self._step.clear()
        self._episode.clear()
        if agent_ids:
            for a in agent_ids:
                self._step[a] = 0.0
                self._episode[a] = 0.0

    def clear_step(self, agent_ids: list[str] | None = None) -> None:
        """Reset the per-step accumulator (leave the episode accumulator)."""
        if agent_ids is None:
            for k in self._step:
                self._step[k] = 0.0
        else:
            for a in agent_ids:
                self._step[a] = 0.0

    # ── config lookup ─────────────────────────────────────────────────
    def get(self, event: str, default: float = 0.0) -> float:
        """Return the configured reward value for ``event``."""
        if hasattr(self.config, event):
            return float(getattr(self.config, event))
        return default

    # ── accumulators ──────────────────────────────────────────────────
    def award(
        self,
        recipient_id: str,
        event: str,
        multiplier: float = 1.0,
    ) -> float:
        """Credit ``recipient_id`` with the reward configured for ``event``.

        Returns the signed amount actually awarded (useful for debugging
        and for tests).  Unknown events resolve to ``0.0``.
        """
        value = self.get(event) * float(multiplier)
        if value == 0.0:
            return 0.0
        self._step[recipient_id] = self._step.get(recipient_id, 0.0) + value
        self._episode[recipient_id] = self._episode.get(recipient_id, 0.0) + value
        return value

    # ── Event.emit override ───────────────────────────────────────────
    def emit(self, event_type: str, **payload: Any) -> None:
        """Route wrapped-method events through ``award``.

        ``Event.wrap`` calls ``emit(event_type, recipient_id=..., multiplier=...)``
        after each wrapped call; we forward those into the accumulator.
        Unknown payload shapes fall back to the base log behaviour.
        """
        recipient = payload.get("recipient_id")
        if recipient is None:
            super().emit(event_type, **payload)
            return
        multiplier = float(payload.get("multiplier", 1.0))
        self.award(recipient, event_type, multiplier=multiplier)

    # ── read-only views ───────────────────────────────────────────────
    def step_rewards(self) -> dict[str, float]:
        """Return a copy of the per-step reward dict."""
        return dict(self._step)

    def episode_rewards(self) -> dict[str, float]:
        """Return a copy of the per-episode reward dict."""
        return dict(self._episode)
