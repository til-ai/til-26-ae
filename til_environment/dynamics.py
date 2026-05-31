"""
dynamics.py - Game logic and physics for the TIL Bomberman (Bomberman-scoped).

Owns all mutable game-state transitions: movement, collision detection,
bomb placement/detonation, damage, tile collection, resource economy,
agent freeze/respawn, and visibility.
"""

import logging
from collections import defaultdict

import numpy as np
from gymnasium.utils.seeding import np_random
from omegaconf import DictConfig
from til_environment.actions import Action, ActionMask
from til_environment.arena import (  # noqa: F401
    ArenaGenerator,
    ArenaResult,
    ArenaState,
    WallEdge,
    WallResult,
)
from til_environment.config import default_config, viewcone_tuple
from til_environment.entities import (
    Agent,
    AttackIntent,
    Base,
    Bomb,
    EntityRegistry,
    Item,
    Mission,
    RadiusAttack,
    Recon,
    Resource,
    SquareVision,
)
from til_environment.entities.base import EntityStatus
from til_environment.entities.protocols import (
    Attacker,
    Defender,
    Timed,
    Vision,
)
from til_environment.events import EmitRule, Rewards
from til_environment.helpers import (
    _los_to_tile,
    idx_to_view,
    is_world_coord_valid,
    supercover_line,
    view_to_world,
)
from til_environment.observation import (
    NUM_CHANNELS,
    build_agent_viewcone,
    build_radius_view,
)
from til_environment.types import Direction

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Vision system
# ═══════════════════════════════════════════════════════════════════════════
class VisionSystem:
    """Line-of-sight and viewcone computation."""

    def __init__(
        self,
        grid_size: int,
        viewcone: tuple[int, int, int, int] = (2, 2, 2, 4),
    ) -> None:
        self.grid_size = grid_size
        self.viewcone = viewcone
        self.viewcone_width = viewcone[0] + viewcone[1] + 1
        self.viewcone_length = viewcone[2] + viewcone[3] + 1

    def is_visible(
        self,
        start: np.ndarray,
        end: np.ndarray,
        walls: set[tuple[tuple[int, int], tuple[int, int]]],
    ) -> bool:
        if (start == end).all():
            return True
        path = supercover_line(start, end)
        for i in range(len(path) - 1):
            tile, next_tile = path[i], path[i + 1]
            dx = tile[0] - next_tile[0]
            dy = tile[1] - next_tile[1]
            if dx != 0 and dy != 0:
                horiz0 = tuple(sorted((tile, (next_tile[0], tile[1]))))
                horiz1 = tuple(sorted(((next_tile[0], tile[1]), next_tile)))
                vert0 = tuple(sorted((tile, (tile[0], next_tile[1]))))
                vert1 = tuple(sorted(((tile[0], next_tile[1]), next_tile)))
                if (horiz0 in walls or horiz1 in walls) and (
                    vert0 in walls or vert1 in walls
                ):
                    return False
            else:
                edge = tuple(sorted((tile, next_tile)))
                if edge in walls:
                    return False
        return True

    def build_viewcone(
        self,
        agent_loc: np.ndarray,
        agent_dir: int,
        state: np.ndarray,
        walls: set,
        registry: EntityRegistry,
        observer_team: int | None = None,
    ) -> np.ndarray:
        return build_agent_viewcone(
            agent_pos=agent_loc,
            agent_dir=agent_dir,
            viewcone=self.viewcone,
            state=state,
            walls=walls,
            registry=registry,
            observer_team=observer_team,
            grid_size=self.grid_size,
        )

    def get_team_visible_area(
        self,
        team: int,
        registry: EntityRegistry,
        walls: set,
        state: np.ndarray,
    ) -> set[tuple[int, int]]:
        visible: set[tuple[int, int]] = set()

        for provider in registry.query().type(Vision).team(team):
            if isinstance(provider, Agent):
                agent = provider
                direction = Direction(agent.direction)
                for idx in np.ndindex((self.viewcone_length, self.viewcone_width)):
                    view_coord = idx_to_view(np.array(idx), self.viewcone)
                    world_coord = view_to_world(agent.position, direction, view_coord)
                    if not is_world_coord_valid(world_coord, self.grid_size):
                        continue
                    if self.is_visible(agent.position, world_coord, walls):
                        visible.add((int(world_coord[0]), int(world_coord[1])))
            else:
                for wx, wy in provider.get_visible_cells():
                    if 0 <= wx < self.grid_size and 0 <= wy < self.grid_size:
                        if self.is_visible(
                            provider.position, np.array([wx, wy]), walls
                        ):
                            visible.add((wx, wy))
        return visible


# ═══════════════════════════════════════════════════════════════════════════
# Dynamics
# ═══════════════════════════════════════════════════════════════════════════
class Dynamics:
    """Owns all mutable game-state transitions."""

    def __init__(self, cfg: DictConfig | None = None) -> None:
        if cfg is None:
            cfg = default_config()
        self.cfg = cfg

        self.grid_size: int = int(cfg.env.grid_size)
        self.num_teams: int = int(cfg.env.num_teams)

        self.viewcone = viewcone_tuple(cfg.dynamics.vision)

        self.arena_gen = ArenaGenerator(
            grid_size=self.grid_size,
            wall_prob=float(cfg.dynamics.arena.wall_prob),
            wall_destructible_ratio=float(cfg.dynamics.arena.wall_destructible_ratio),
            mission_prob=float(cfg.dynamics.arena.mission_prob),
            recon_prob=float(cfg.dynamics.arena.recon_prob),
            resource_prob=float(cfg.dynamics.arena.resource_prob),
            novice=bool(cfg.env.novice),
            base_respawn_steps=int(cfg.env.tile_respawn_steps),
        )
        self.vision = VisionSystem(grid_size=self.grid_size, viewcone=self.viewcone)
        self.action_mask = ActionMask(self)
        self.rewards = Rewards(cfg)

        self.registry = EntityRegistry(self.grid_size)
        self.arena_state: ArenaState | None = None
        self._bomb_counter: int = 0
        self._respawn_counter: int = 0
        self._respawn_queue: list[dict] = []

        # Populated by detonation_phase() each round; read by the renderer to
        # draw blast-cell overlays on the frame where a bomb explodes.
        self.last_explosions: list[dict] = []

        # Agent IDs that stepped onto a Mission tile during the current
        # movement phase. Cleared by Bomberman at the start of each round
        # (before resolve_movement_actions) and read by _build_info to set
        # info["add_mission"]. Drives the competition server's mission queue.
        self.mission_collectors_this_step: set[str] = set()

        # Per-team economy — floating-point ratios against bomb_cost.
        self.team_resources: dict[int, float] = {t: 0.0 for t in range(self.num_teams)}
        self.team_bombs: dict[int, int] = {t: 0 for t in range(self.num_teams)}

        self._rng: np.random.Generator | None = None
        self._rng_seed: int | None = None

    # ── reward hook installation ──────────────────────────────────────────

    def _install_reward_hooks(self, entity) -> None:
        """Install Rewards ``EmitRule`` wrappers on an entity's protocol methods.

        For ``Defender.receive_damage``:
        * ``attack_damage`` — to attacker, scaled by actual damage dealt.
        * ``attack_kill`` — to attacker when the defender's HP drops to zero
          (we key on ``health <= 0`` rather than ``not e.alive`` since Agents
          freeze rather than destroy themselves).
        * ``destroy_wall`` — awarded in the detonation phase per wall edge removed.

        For ``Item.collect`` (Mission, Recon, Resource) — awards
        ``collect_<classname>`` to the collecting agent.
        """
        if isinstance(entity, Agent):
            self.rewards.wrap_multi(
                entity,
                "receive_damage",
                [
                    EmitRule(
                        event_type="attack_damage",
                        recipient_fn=lambda r, amt, intent=None: (
                            intent.attribute_rewards if intent is not None else None
                        ),
                        scale_by_return=True,
                    ),
                    # attack_kill is NOT wired here — damage_phase awards it after
                    # collecting all contributors so the reward can be split evenly.
                    # Equal-and-opposite penalty to the agent who was hit.
                    EmitRule(
                        event_type="attack_damage",
                        recipient_id=entity.entity_id,
                        scale_by_return=True,
                        multiplier=-1.0,
                    ),
                ],
            )
        elif isinstance(entity, Base):
            self.rewards.wrap_multi(
                entity,
                "receive_damage",
                [
                    EmitRule(
                        event_type="attack_damage",
                        recipient_fn=lambda r, amt, intent=None: (
                            intent.attribute_rewards if intent is not None else None
                        ),
                        scale_by_return=True,
                    ),
                    # Equal-and-opposite penalty to the agent whose base was hit.
                    EmitRule(
                        event_type="attack_damage",
                        recipient_fn=lambda r, amt, intent=None, b=entity: (
                            self.registry.agents(b.team)[0].entity_id
                            if self.registry.agents(b.team)
                            else None
                        ),
                        scale_by_return=True,
                        multiplier=-1.0,
                    ),
                    # destroy_enemy_base is NOT wired here — same reason as attack_kill.
                    # own_base_destroyed fires to the owning agent when base dies.
                    EmitRule(
                        event_type="own_base_destroyed",
                        recipient_fn=lambda r, amt, intent=None, b=entity: (
                            self.registry.agents(b.team)[0].entity_id
                            if self.registry.agents(b.team)
                            else None
                        ),
                        condition=lambda r, amt, intent=None, e=entity: not e.alive,
                    ),
                ],
            )
        if isinstance(entity, Item):
            event_key = f"collect_{type(entity).__name__.lower()}"
            self.rewards.wrap(
                entity,
                "collect",
                event_key,
                recipient_fn=lambda result, team, agent_id: agent_id,
            )

    # ── reset ──────────────────────────────────────────────────────────────

    def init_rng(self, seed: int | None = None) -> None:
        self._rng, self._rng_seed = np_random(seed)

    def reset(self, seed: int | None = None) -> list[str]:
        """Generate a new episode and populate entities."""
        if self._rng is None or seed is not None:
            self.init_rng(seed)

        if self.arena_gen.novice:
            episode_rng, _ = np_random(88)
            episode_seed = 88
        else:
            episode_rng = self._rng
            episode_seed = int(self._rng.integers(0, 2**31))

        result = self.arena_gen.generate_episode(
            episode_rng,
            episode_seed,
            num_teams=self.num_teams,
        )
        self.arena_state = ArenaState(
            grid_size=self.grid_size,
            wall_grid=result.wall_grid,
            walls=result.walls,
        )
        self.respawn_map: np.ndarray = result.respawn_map

        self.registry.clear()
        self._bomb_counter = 0
        self._respawn_queue = []
        self.mission_collectors_this_step = set()

        # Reset per-team economy to config defaults.
        rcfg = self.cfg.resources
        self.team_resources = {
            t: float(rcfg.starting_resources) for t in range(self.num_teams)
        }
        self.team_bombs = {t: int(rcfg.starting_bombs) for t in range(self.num_teams)}

        self.rewards.reset()

        agent_ids: list[str] = []
        for team in range(self.num_teams):
            bcfg = self.cfg.entities.base
            base = Base(
                entity_id=f"base_team{team}",
                team=team,
                position=result.base_locations[team].copy(),
                health=float(bcfg.health),
                max_health=float(bcfg.max_health),
                vision_type=SquareVision(int(bcfg.vision_radius)),
            )
            self.registry.add(base)
            self._install_reward_hooks(base)

            acfg = self.cfg.entities.agent
            agent_id = f"agent_{team}"
            agent = Agent(
                entity_id=agent_id,
                team=team,
                position=result.starting_locations[team].copy(),
                direction=int(result.starting_directions[team]),
                health=float(acfg.health),
                max_health=float(acfg.max_health),
                freeze_duration=int(acfg.freeze_turns),
            )
            self.registry.add(agent)
            self._install_reward_hooks(agent)
            agent_ids.append(agent_id)

        # Scatter static entities.
        mcfg = self.cfg.entities.mission
        rcfg_e = self.cfg.entities.recon
        rescfg = self.cfg.entities.resource
        mission_idx = 0
        recon_idx = 0
        resource_idx = 0
        for spec in result.static_entities:
            if spec.kind == "mission":
                e = Mission(
                    entity_id=f"mission_{mission_idx}",
                    team=None,
                    position=spec.position.copy(),
                    reward_value=float(mcfg.reward_value),
                    difficulty=float(mcfg.difficulty),
                )
                self.registry.add(e)
                self._install_reward_hooks(e)
                mission_idx += 1
            elif spec.kind == "recon":
                e = Recon(
                    entity_id=f"recon_{recon_idx}",
                    team=None,
                    position=spec.position.copy(),
                    reward_value=float(rcfg_e.reward_value),
                )
                self.registry.add(e)
                self._install_reward_hooks(e)
                recon_idx += 1
            elif spec.kind == "resource":
                e = Resource(
                    entity_id=f"resource_{resource_idx}",
                    team=None,
                    position=spec.position.copy(),
                    amount=float(rescfg.amount),
                )
                self.registry.add(e)
                self._install_reward_hooks(e)
                resource_idx += 1

        return agent_ids

    # ── observation ────────────────────────────────────────────────────────

    def observe(self, agent_id: str) -> dict:
        """Build the observation dictionary for a single agent."""
        agent = self.registry.get(agent_id)
        if not isinstance(agent, Agent):
            raise ValueError(f"{agent_id} is not an Agent")

        viewcone = self.vision.build_viewcone(
            agent.position,
            agent.direction,
            self.arena_state._state,
            self.arena_state.walls,
            self.registry,
            observer_team=agent.team,
        )

        bases = self.registry.query().type(Base).team(agent.team).all()
        if bases:
            base = bases[0]
            base_view = build_radius_view(
                center=base.position,
                radius=base.vision_type.radius,
                state=self.arena_state._state,
                walls=self.arena_state.walls,
                registry=self.registry,
                observer_team=agent.team,
                grid_size=self.vision.grid_size,
            )
            base_health = base.health
        else:
            base_view = np.zeros((1, 1, NUM_CHANNELS), dtype=np.float32)
            base_health = 0.0

        return {
            "agent_viewcone": viewcone,
            "base_viewcone": base_view,
            "direction": agent.direction,
            "location": agent.position.copy(),
            "base_location": (
                base.position.copy() if bases else np.zeros(2, dtype=np.uint8)
            ),
            "health": np.array([agent.health], dtype=np.float32),
            "frozen_ticks": agent.frozen_ticks,
            "base_health": np.array([base_health], dtype=np.float32),
            "team_resources": np.array(
                [self.team_resources.get(agent.team, 0.0)], dtype=np.float32
            ),
            "team_bombs": self.team_bombs.get(agent.team, 0),
            "step": 0,  # overridden by Bomberman
            "action_mask": self.action_mask.build(agent_id),
        }

    # ── movement & collision ──────────────────────────────────────────────

    def _blocks_movement(self, agent: Agent, direction: Direction) -> bool:
        """Return True if *agent* cannot step in *direction*.

        Blocked by any edge wall (indestructible or destructible) on the
        current tile in *direction*, or an out-of-bounds destination.
        """
        return self.arena_state.enforce_wall_collision(agent.position, direction)

    # Legacy alias used by a few callers (e.g. older tests). Kept deliberately.
    def _enforce_wall_collision(self, agent: Agent, direction: Direction) -> bool:
        return self._blocks_movement(agent, direction)

    def _agent_at(
        self, position: np.ndarray, exclude: str | None = None
    ) -> Agent | None:
        for entity in self.registry.query().at(int(position[0]), int(position[1])):
            if not isinstance(entity, Agent):
                continue
            if entity.entity_id == exclude:
                continue
            return entity
        return None

    def move_agent(self, agent_id: str, action: int) -> dict:
        """execute a single agent's movement-class action (or no-op)."""
        agent = self.registry.get(agent_id)
        if not isinstance(agent, Agent) or not agent.alive or agent.is_frozen:
            return {"moved": False}

        _action = Action(action)
        info: dict = {
            "moved": False,
            "collided_wall": False,
            "collided_agent": None,
        }

        if _action in (Action.FORWARD, Action.BACKWARD):
            dir_idx = (
                agent.direction
                if _action is Action.FORWARD
                else (agent.direction + 2) % 4
            )
            direction = Direction(dir_idx)

            if self._blocks_movement(agent, direction):
                info["collided_wall"] = True
                self.rewards.award(agent_id, "agent_collide_wall")
                return info

            next_loc = np.clip(
                agent.position + direction.movement, 0, self.grid_size - 1
            )

            other = self._agent_at(next_loc, exclude=agent_id)
            if other is not None:
                info["collided_agent"] = other.entity_id
                self.rewards.award(agent_id, "agent_collide_agent")
                return info

            agent.position = next_loc
            info["moved"] = True

            self._handle_tile_pickup(agent, info)

        elif _action in (Action.LEFT, Action.RIGHT):
            agent.direction = (
                agent.direction + (3 if _action is Action.LEFT else 1)
            ) % 4
            info["moved"] = True

        # STAY / PLACE_BOMB -> movement is a no-op here.
        return info

    # ═══════════════════════════════════════════════════════════════════════
    # Phased tick
    # ═══════════════════════════════════════════════════════════════════════

    def place_bomb_phase(self, actions: dict[str, int]) -> dict[str, dict]:
        """
        do the bomb

        for each agent whose action is ``PLACE_BOMB``, if their team has at
        least one bomb available (``team_bombs[team] > 0``) and the agent is
        alive and not frozen, spawn a ``Bomb`` entity at the agent's tile and
        decrement the team's bomb count.
        """
        results: dict[str, dict] = {}
        for agent_id, raw_action in actions.items():
            if Action(raw_action) is not Action.PLACE_BOMB:
                continue
            agent = self.registry.get(agent_id)
            if not isinstance(agent, Agent) or not agent.alive or agent.is_frozen:
                results[agent_id] = {"placed": None, "reason": "agent_unavailable"}
                continue
            if self.team_bombs.get(agent.team, 0) <= 0:
                results[agent_id] = {"placed": None, "reason": "no_bombs"}
                continue

            self.team_bombs[agent.team] -= 1
            self._bomb_counter += 1
            bcfg = self.cfg.entities.bomb
            bomb_id = f"bomb_{agent.team}_{self._bomb_counter}"
            bomb = Bomb(
                entity_id=bomb_id,
                team=agent.team,
                position=agent.position.copy(),
                attack=float(bcfg.attack),
                blast_radius=int(bcfg.blast_radius),
                timer=int(bcfg.timer),
                attribute_rewards=agent_id,
                attack_type=RadiusAttack(int(bcfg.blast_radius)),
            )
            self.registry.add(bomb)
            results[agent_id] = {"placed": bomb_id, "reason": "ok"}
        return results

    def _compute_blast(
        self,
        bomb: "Bomb",
    ) -> tuple[list[np.ndarray], set[tuple[int, int, int]]]:
        """Compute blast cells and candidate wall destructions for an expired bomb.

        Reads the current wall state but does NOT mutate it — callers are
        responsible for applying destructions after all bombs have been evaluated.

        Pass 1 — tile reachability (blast damage cells):
            For each tile within the Chebyshev blast radius, compute LOS from
            the bomb using all walls (indestructible AND destructible) as
            blockers.  Reachable tiles are added to blast_cells.

        Pass 2 — wall destruction candidates (wall-first query):
            Iterate directly over every wall edge in the arena.  For each edge
            within blast radius, check if the blast can reach *either* adjacent
            tile.  If yes and the edge is destructible, add it to the returned set.

        Returns (blast_cells, walls_to_destroy) where walls_to_destroy is a set
        of (wx, wy, direction_value) tuples.  The bomb's own cell is always
        included in blast_cells.
        """
        origin = bomb.position
        ox, oy = int(origin[0]), int(origin[1])
        gs = self.arena_state.grid_size
        r = bomb.blast_radius
        state = self.arena_state._state

        # ── Pass 1: blast cells ───────────────────────────────────────────────
        blast: list[np.ndarray] = []
        reachable: set[tuple[int, int]] = set()
        for tx in range(max(0, ox - r), min(gs, ox + r + 1)):
            for ty in range(max(0, oy - r), min(gs, oy + r + 1)):
                if _los_to_tile(ox, oy, tx, ty, state):
                    blast.append(np.array([tx, ty]))
                    reachable.add((tx, ty))

        # ── Pass 2: wall destruction candidates ──────────────────────────────
        walls_to_destroy: set[tuple[int, int, int]] = set()
        for we in self.arena_state.wall_edges.values():
            if not we.destructible:
                continue
            # Within blast radius if either adjacent tile is in range.
            if (
                max(abs(we.ax - ox), abs(we.ay - oy)) > r
                and max(abs(we.bx - ox), abs(we.by - oy)) > r
            ):
                continue
            if (we.ax, we.ay) in reachable or (we.bx, we.by) in reachable:
                walls_to_destroy.add((we.ax, we.ay, we.direction.value))

        return blast, walls_to_destroy

    def resolve_movement_actions(self, actions: dict[str, int]) -> dict[str, dict]:
        """resolve all movement / turn / stay actions."""
        infos: dict[str, dict] = {}
        for agent_id, action in actions.items():
            infos[agent_id] = self.move_agent(agent_id, action)
        return infos

    def detonation_phase(self) -> list[AttackIntent]:
        """
        expired bombs detonate and produce AttackIntents.

        Two-pass design to guarantee determinism regardless of how many bombs
        expire on the same tick:

        Pass 1 — compute (no mutations):
            All bombs read the same unmodified wall snapshot.  Blast cells and
            AttackIntents are collected; candidate wall destructions are
            accumulated per bomb.  Bombs are processed in entity_id order so
            the resulting intent list is stable.

        Pass 2 — apply wall destructions:
            Wall edges are destroyed in entity_id order.  If two bombs target
            the same edge, the first bomb (by entity_id) gets the reward; the
            second silently skips it (already gone).

        Indestructible edges block blast without being destroyed.
        """
        intents: list[AttackIntent] = []
        self.last_explosions = []
        defender_ids: set[str] = self.registry.query().type(Defender).ids()

        timed_attackers = self.registry.query().type(Timed).type(Attacker).all()
        attackers = sorted(
            [e for e in timed_attackers if e.expired and e.alive],
            key=lambda b: b.entity_id,
        )

        # Pass 1 — compute all blasts against the unmodified wall snapshot.
        per_bomb: list[tuple[str, set[tuple[int, int, int]]]] = []
        for attacker in attackers:
            blast_cells, walls_to_destroy = self._compute_blast(attacker)
            self.last_explosions.append(
                {
                    "team": attacker.team,
                    "origin": attacker.position.copy(),
                    "blast_radius": attacker.blast_radius,
                    "cells": blast_cells,
                }
            )

            # kablow
            damage = attacker.attack_power
            attacker.destroy()  # this is a bomb so destroying after attack makes sense.

            # collect defender targets within the blast area.
            target_ids: set[str] = set()
            for c in blast_cells:
                cx, cy = int(c[0]), int(c[1])
                target_ids |= self.registry.pos_index[cx][cy]  # L bro

            for eid in sorted(target_ids & defender_ids):
                target = self.registry.get(eid)
                if target is None:
                    continue
                if target.team is not None and target.team == attacker.team:
                    continue
                intents.append(
                    AttackIntent(
                        attacker_id=attacker.entity_id,
                        defender_id=eid,
                        damage=damage,
                        attribute_rewards=attacker.attribute_rewards,
                    )
                )

            per_bomb.append((attacker.attribute_rewards, walls_to_destroy))

        # Pass 2 — apply wall destructions.
        # Process in entity_id order; first bomb to claim a shared edge gets the reward.
        destroyed_walls: set[tuple[int, int, int]] = set()
        for placing_agent, walls in per_bomb:
            for wx, wy, wd in walls:
                if (wx, wy, wd) in destroyed_walls:
                    continue
                if self.arena_state.destroy_wall(wx, wy, Direction(wd)):
                    destroyed_walls.add((wx, wy, wd))
                    self.rewards.award(placing_agent, "destroy_wall")

        return intents

    def damage_phase(self, intents: list[AttackIntent]) -> dict[str, dict]:
        """
        defenders receive queued damage; kill/destroy rewards are split evenly
        among all unique bomb-placers that dealt >0 damage to a given defender.

        ^ thanks claude. you will do damage anyways but yeah thanks for the help
        """
        # Group intents by defender so we can split kill/destroy credit.
        by_defender: dict[str, list[AttackIntent]] = defaultdict(list)
        for intent in intents:
            by_defender[intent.defender_id].append(intent)

        results: dict[str, dict] = {}
        for defender_id, group in by_defender.items():
            defender = self.registry.get(defender_id)
            if defender is None or not defender.alive:
                continue

            # Track which placers dealt actual damage (for split credit).
            contributors: list[str] = []

            for intent in group:
                # Skip if the defender died mid-group (Agent entered frozen state).
                if not defender.alive:
                    break
                if isinstance(defender, Agent) and defender.is_frozen:
                    break

                actual_dmg = defender.receive_damage(intent.damage, intent)
                if actual_dmg > 0 and intent.attribute_rewards not in contributors:
                    contributors.append(intent.attribute_rewards)

                results[intent.attacker_id] = {
                    "attacked": defender_id,
                    "attack_damage": actual_dmg,
                    "attribute_rewards": intent.attribute_rewards,
                }

            if not contributors:
                continue

            split = 1.0 / len(contributors)

            # Agent killed → freeze + split attack_kill among contributors.
            if (
                isinstance(defender, Agent)
                and defender.health <= 0
                and not defender.is_frozen
            ):
                defender.frozen_ticks = defender.freeze_duration
                for placer_id in contributors:
                    self.rewards.award(placer_id, "attack_kill", multiplier=split)

            # Base destroyed → split destroy_enemy_base among contributors.
            elif isinstance(defender, Base) and not defender.alive:
                for placer_id in contributors:
                    self.rewards.award(
                        placer_id, "destroy_enemy_base", multiplier=split
                    )

        return results

    def upkeep(self) -> dict:
        """
        end-of-round bookkeeping.

        * Tick Timed entity timers (bombs).
        * Base-generated resources credited to every team (fixed rate).
        * Auto-convert resources → bombs whenever the pool reaches ``bomb_cost``.
        * Tick agent freeze cooldowns; on hitting zero, respawn at full HP in
          place (no teleport).
        """
        summary: dict = {}

        # Tick all Timed entities (placed bombs).
        for timed in list(self.registry.query().type(Timed)):
            if timed.alive:
                timed.tick_timer()

        # Base-generated resources.
        rcfg = self.cfg.resources
        base_rate = float(rcfg.base_resource_rate)
        bomb_cost = float(rcfg.bomb_cost)
        for team in range(self.num_teams):
            self.team_resources[team] += base_rate
            # Auto-convert surplus into bombs.
            while self.team_resources[team] >= bomb_cost:
                self.team_resources[team] -= bomb_cost
                self.team_bombs[team] = self.team_bombs.get(team, 0) + 1

        # Tick agent freeze / respawn.
        for agent in self.registry.query().type(Agent):
            if agent.is_frozen:
                agent.frozen_ticks -= 1
                if agent.frozen_ticks <= 0:
                    agent.frozen_ticks = 0
                    agent.health = agent.max_health

        # Tick collectible respawn queue.
        next_queue = []
        for entry in self._respawn_queue:
            entry["steps_remaining"] -= 1
            if entry["steps_remaining"] <= 0:
                self._spawn_collectible(entry)
            else:
                next_queue.append(entry)
        self._respawn_queue = next_queue

        summary["team_resources"] = dict(self.team_resources)
        summary["team_bombs"] = dict(self.team_bombs)
        return summary

    # ── collectible respawn ────────────────────────────────────────────────

    def _queue_respawn(
        self, kind: str, position: np.ndarray, kwargs: dict, steps: int
    ) -> None:
        """Schedule a collectible to reappear at *position* after *steps* ticks."""
        self._respawn_queue.append(
            {
                "kind": kind,
                "position": position,
                "kwargs": kwargs,
                "steps_remaining": steps,
            }
        )

    def _spawn_collectible(self, entry: dict) -> None:
        """Materialise a queued collectible back into the registry."""
        self._respawn_counter += 1
        uid = self._respawn_counter
        kind = entry["kind"]
        pos = entry["position"]
        kwargs = entry["kwargs"]

        if kind == "mission":
            e = Mission(
                entity_id=f"mission_r{uid}", team=None, position=pos.copy(), **kwargs
            )
        elif kind == "recon":
            e = Recon(
                entity_id=f"recon_r{uid}", team=None, position=pos.copy(), **kwargs
            )
        elif kind == "resource":
            e = Resource(
                entity_id=f"resource_r{uid}", team=None, position=pos.copy(), **kwargs
            )
        else:
            return

        self.registry.add(e)
        self._install_reward_hooks(e)

    # ── tile pickup ────────────────────────────────────────────────────────

    def _handle_tile_pickup(self, agent: Agent, info: dict) -> None:
        """Check if the agent just stepped onto a collectible Item tile."""
        x, y = int(agent.position[0]), int(agent.position[1])
        tile_entities = (
            self.registry.query().type(Item).status(EntityStatus.ACTIVE).at(x, y).all()
        )

        for entity in tile_entities:
            pos = entity.position.copy()
            px, py = int(pos[0]), int(pos[1])
            steps = int(self.respawn_map[px, py])
            if isinstance(entity, Resource):
                amount = entity.collect(agent.team, agent.entity_id)
                self.team_resources[agent.team] = self.team_resources.get(
                    agent.team, 0.0
                ) + float(amount or 0.0)
                self._queue_respawn("resource", pos, {"amount": entity.amount}, steps)
            elif isinstance(entity, Mission):
                entity.collect(agent.team, agent.entity_id)
                self.mission_collectors_this_step.add(agent.entity_id)
                self._queue_respawn(
                    "mission",
                    pos,
                    {
                        "reward_value": entity.reward_value,
                        "difficulty": entity.difficulty,
                    },
                    steps,
                )
            elif isinstance(entity, Recon):
                entity.collect(agent.team, agent.entity_id)
                self._queue_respawn(
                    "recon", pos, {"reward_value": entity.reward_value}, steps
                )

    # ── hook called by Agent.receive_damage via event wrap ─────────────────

    def start_freeze(self, agent: Agent) -> None:
        """Enter the frozen state on an agent whose HP just hit 0.

        Used as a callback after the reward hook fires on ``receive_damage``.
        The agent persists on the grid, with frozen_ticks set from config.
        """
        if agent.frozen_ticks <= 0 and agent.health <= 0:
            agent.frozen_ticks = int(self.cfg.entities.agent.freeze_turns)

    # ── win / loss conditions ──────────────────────────────────────────────

    def check_termination(self) -> dict[int, str]:
        """Check if any team has lost.

        In the Bomberman-scoped game, agents cannot be permanently killed
        (they freeze and respawn), so only base destruction triggers loss.
        """
        losses: dict[int, str] = {}
        for team in range(self.num_teams):
            bases_alive = self.registry.bases(team)
            if not bases_alive:
                losses[team] = "base_destroyed"
        return losses

    # ── state access ───────────────────────────────────────────────────────

    @property
    def state(self) -> np.ndarray:
        return self.arena_state._state

    def _in_bounds_cells(self, cells):
        return [
            cell
            for cell in cells
            if (0 <= cell[0] < self.grid_size and 0 <= cell[1] < self.grid_size)
        ]
