"""
events/base.py — Base Event class.

Events encapsulate two cross-cutting concerns that RL simulations generate
far more of than ordinary games/simulations:

1. **Logging** — structured records of what happened, written to a
   configurable endpoint (text file by default; tensorboard/other sinks
   can be plugged in later).
2. **Side effects on external state** — most notably reward accumulation,
   but the same machinery can host other "observer" concerns (metrics,
   replays, curriculum signals, ...).

Subclasses override ``emit`` to add their own bookkeeping, and may expose
``wrap``-style helpers that decorate entity methods so the side effect
fires automatically whenever the underlying event-function is called.

^
thanks claude

If you're a human reading this, this class is meant to provide you an easier time with modifying
the dynamics of this environment. Say if you want specific logging for specific events, create
an emit rule and subclass this and tie it to an endpoint. That way you can figure out what's going
on under the hood and mutate your algos better. Or something

"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class EmitRule:
    """One emission rule attached to a wrapped method.

    A single call to the wrapped method can fire zero or more of these,
    so a method like ``receive_damage`` can produce defender-side *and*
    attacker-side rewards, plus conditional kill rewards, from one hook.

    Attributes
    ----------
    event_type : str
        Event identifier passed to ``Event.emit`` (and used by ``Rewards``
        as a config key for the reward value).
    recipient_id : str | None
        Static recipient.  Mutually exclusive with ``recipient_fn``.
    recipient_fn : Callable | None
        Dynamic recipient lookup.  Called as
        ``recipient_fn(result, *args, **kwargs)`` where ``result`` is the
        wrapped method's return value.  Return ``None`` to skip this rule.
    scale_by_return : bool
        If True, multiply the reward by the wrapped method's return value.
    condition : Callable | None
        Optional predicate ``(result, *args, **kwargs) -> bool``.  If
        provided and returns False, the rule is skipped for this call.
        Useful for kill-only emissions that depend on post-call state.
    """

    event_type: str
    recipient_id: str | None = None
    recipient_fn: Callable[..., str | None] | None = None
    scale_by_return: bool = False
    condition: Callable[..., bool] | None = None
    multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.recipient_id is None and self.recipient_fn is None:
            raise ValueError("EmitRule needs either recipient_id or recipient_fn")


class Event:
    """Base class for all event-producing subsystems.

    Parameters
    ----------
    name : str
        Short identifier used as a log prefix (e.g. ``"rewards"``).
    log_path : str | Path | None
        File path to append log lines to.  If ``None``, defaults to
        ``./logs/events/<name>.log`` relative to the current working
        directory.  Every Event always has a concrete log endpoint —
        text logs are cheap and clarity is worth more than a few KB.
    """

    DEFAULT_LOG_DIR = Path("logs/events")

    def __init__(self, name: str, log_path: str | Path | None = None) -> None:
        self.name = name
        self.log_path: Path = (
            Path(log_path)
            if log_path is not None
            else self.DEFAULT_LOG_DIR / f"{name}.log"
        )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ── logging endpoint ──────────────────────────────────────────────
    def log(self, message: str) -> None:
        """Append a single line to the log endpoint."""
        with self.log_path.open("a") as f:
            f.write(f"[{self.name}] {message}\n")

    def emit(self, event_type: str, **payload: Any) -> None:
        """Record a structured event.

        Subclasses override this to add bookkeeping (e.g. reward accumulation).
        The base implementation just logs a line.
        """
        self.log(f"{event_type} {payload}")

    # ── method wrapping ───────────────────────────────────────────────
    def wrap(
        self,
        obj: object,
        method_name: str,
        event_type: str,
        *,
        recipient_id: str | None = None,
        recipient_fn: Callable[..., str | None] | None = None,
        scale_by_return: bool = False,
        condition: Callable[..., bool] | None = None,
    ) -> None:
        """Convenience single-rule wrapper.

        Equivalent to ``wrap_multi(obj, method_name, [EmitRule(...)])``.
        See ``EmitRule`` for parameter semantics.
        """
        self.wrap_multi(
            obj,
            method_name,
            [
                EmitRule(
                    event_type=event_type,
                    recipient_id=recipient_id,
                    recipient_fn=recipient_fn,
                    scale_by_return=scale_by_return,
                    condition=condition,
                )
            ],
        )

    def wrap_multi(
        self,
        obj: object,
        method_name: str,
        rules: list[EmitRule],
    ) -> None:
        """Monkey-patch ``obj.method_name`` with a list of emission rules.

        The wrapped call runs normally and its return value is preserved.
        After the call, every rule in ``rules`` is evaluated in order:

        * If ``rule.condition`` is set and returns False → skip.
        * Recipient resolved from ``rule.recipient_id`` or
          ``rule.recipient_fn(result, *args, **kwargs)``.
        * Multiplier = float(result) if ``rule.scale_by_return`` else 1.0.
        * ``self.emit(rule.event_type, recipient_id=..., multiplier=...)``.

        A single call can therefore produce multiple emissions — e.g.
        ``receive_damage`` credits both the defender (self-penalty) and
        the attacker (via the passed ``intent``), plus optional
        kill-conditioned emissions.
        """
        if not rules:
            raise ValueError("wrap_multi requires at least one EmitRule")

        original = getattr(obj, method_name)

        @functools.wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            for rule in rules:
                if rule.condition is not None and not rule.condition(
                    result, *args, **kwargs
                ):
                    continue
                if rule.recipient_fn is not None:
                    recipient = rule.recipient_fn(result, *args, **kwargs)
                else:
                    recipient = rule.recipient_id
                if recipient is None:
                    continue
                multiplier = (
                    float(result) * rule.multiplier
                    if rule.scale_by_return and result is not None
                    else rule.multiplier
                )
                self.emit(
                    rule.event_type,
                    recipient_id=recipient,
                    multiplier=multiplier,
                )
            return result

        setattr(obj, method_name, wrapper)
