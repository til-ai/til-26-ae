"""
bomberman_env.py - Thin AECEnv wrapper for the TIL Bomberman.

Top-level API class.  Inherits from PettingZoo's ``AECEnv`` and delegates all
game logic to ``Dynamics`` and rendering to ``Renderer``.
"""

import functools
import logging
from functools import partial
from pathlib import Path

import gymnasium
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete
from omegaconf import DictConfig
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.utils.env import ActionType, AgentID, ObsType
from pettingzoo.utils.wrappers.base import BaseWrapper
from supersuit import frame_stack_v2
from til_environment.actions import NUM_ACTIONS_V2, Action
from til_environment.config import (
    default_config,
    load_config,
    viewcone_tuple,
)
from til_environment.dynamics import Dynamics
from til_environment.entities import Agent
from til_environment.flatten_dict import FlattenDictWrapper
from til_environment.observation import NUM_CHANNELS
from til_environment.renderer import Renderer
from til_environment.types import Direction

logger = logging.getLogger(__name__)


def basic_env(
    cfg: DictConfig | str | Path | None = None,
    env_wrappers: list[BaseWrapper] | None = None,
    **kwargs,
):
    if cfg is None:
        resolved = default_config()
    elif isinstance(cfg, (str, Path)):
        resolved = load_config(cfg)
    else:
        resolved = cfg

    environment = Bomberman(resolved, **kwargs)
    if env_wrappers is None:
        env_wrappers = [
            FlattenDictWrapper,
            partial(frame_stack_v2, stack_size=4, stack_dim=-1),
        ]
    for wrapper in env_wrappers:
        environment = wrapper(environment)
    environment = wrappers.AssertOutOfBoundsWrapper(environment)
    environment = wrappers.OrderEnforcingWrapper(environment)
    return environment


parallel_basic_env = parallel_wrapper_fn(basic_env)


class Bomberman(AECEnv[AgentID, ObsType, ActionType]):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "til_bomberman",
        "is_parallelizable": True,
    }

    def __init__(self, cfg: DictConfig | None = None) -> None:
        super().__init__()

        if cfg is None:
            cfg = default_config()
        self.cfg = cfg

        self.grid_size: int = int(cfg.env.grid_size)
        self.num_teams: int = int(cfg.env.num_teams)
        self.num_iters: int = int(cfg.env.num_iters)

        self.viewcone = viewcone_tuple(cfg.dynamics.vision)
        self.viewcone_width = self.viewcone[0] + self.viewcone[1] + 1
        self.viewcone_length = self.viewcone[2] + self.viewcone[3] + 1

        self.possible_agents = [f"agent_{i}" for i in range(self.num_teams)]

        render_mode = cfg.env.render_mode
        replay_dir = cfg.renderer.replay_dir or None
        if replay_dir is not None and render_mode is None:
            render_mode = "rgb_array"
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.metadata["render_fps"] = int(cfg.renderer.render_fps)

        self.dynamics = Dynamics(cfg)

        self.renderer = Renderer(
            grid_size=self.grid_size,
            window_size=int(cfg.renderer.window_size),
            render_mode=render_mode,
            debug=bool(cfg.renderer.debug),
            viewcone_shape=(self.viewcone_length, self.viewcone_width),
            viewcone=self.viewcone,
            replay_dir=replay_dir,
            render_fps=int(cfg.renderer.render_fps),
            num_teams=self.num_teams,
        )

        self.num_moves: int = 0
        self._agent_actions: dict[AgentID, int] = {}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID | None = None):
        # build observation space bounds from config, uhh dont change it please
        # left agent in there for compat
        base_r = int(self.cfg.entities.base.vision_radius)
        base_side = 2 * base_r + 1
        max_freeze = int(self.cfg.entities.agent.freeze_turns)
        return Dict(
            {
                "agent_viewcone": Box(
                    0.0,
                    1.0,
                    shape=(self.viewcone_length, self.viewcone_width, NUM_CHANNELS),
                    dtype=np.float32,
                ),
                "base_viewcone": Box(
                    0.0,
                    1.0,
                    shape=(base_side, base_side, NUM_CHANNELS),
                    dtype=np.float32,
                ),
                "direction": Discrete(len(Direction)),
                "location": Box(0, self.grid_size, shape=(2,), dtype=np.uint8),
                "base_location": Box(0, self.grid_size, shape=(2,), dtype=np.uint8),
                "health": Box(
                    0.0,
                    float(self.cfg.entities.agent.max_health),
                    shape=(1,),
                    dtype=np.float32,
                ),
                "frozen_ticks": Discrete(max_freeze + 1),
                "base_health": Box(
                    0.0,
                    float(self.cfg.entities.base.max_health),
                    shape=(1,),
                    dtype=np.float32,
                ),
                "team_resources": Box(
                    0.0,
                    float(self.cfg.resources.max_team_resources),
                    shape=(1,),
                    dtype=np.float32,
                ),
                "team_bombs": Discrete(int(self.cfg.resources.max_team_bombs) + 1),
                "step": Discrete(self.num_iters + 1),
                "action_mask": Box(0, 1, shape=(NUM_ACTIONS_V2,), dtype=np.uint8),
            }
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID | None = None):
        return Discrete(NUM_ACTIONS_V2)

    def reset(self, seed: int | None = None, options: dict | None = None):
        prev_stats = self._collect_episode_stats()

        agent_ids = self.dynamics.reset(seed)
        self.agents = agent_ids[:]
        self.agent_selector = AgentSelector(self.agents)
        self.agent_selection = self.agent_selector.next()

        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self.observations = {a: self.observe(a) for a in self.agents}
        self._agent_actions = {}
        self.num_moves = 0

        prev_suffix = (options or {}).get("replay_suffix")
        self.renderer.start_episode(
            prev_suffix=prev_suffix,
            prev_episode_stats=prev_stats,
        )
        if self.render_mode in self.metadata["render_modes"]:
            self.render()

    def _collect_episode_stats(self) -> dict | None:
        if not getattr(self, "agents", None):
            return None

        cumulative = dict(getattr(self, "_cumulative_rewards", {}))
        agent_team = {}
        team_rewards: dict[int, float] = {}
        for agent_id in self.agents:
            try:
                agent = self.dynamics.registry.get(agent_id)
            except Exception:
                continue
            team = int(getattr(agent, "team", -1))
            agent_team[agent_id] = team
            team_rewards[team] = team_rewards.get(team, 0.0) + float(
                cumulative.get(agent_id, 0.0)
            )

        return {
            "num_moves": int(getattr(self, "num_moves", 0)),
            "num_teams": int(self.num_teams),
            "agent_team": agent_team,
            "agent_cumulative_rewards": {k: float(v) for k, v in cumulative.items()},
            "team_cumulative_rewards": {
                str(k): float(v) for k, v in team_rewards.items()
            },
            "terminations": dict(getattr(self, "terminations", {})),
            "truncations": dict(getattr(self, "truncations", {})),
        }

    def observe(self, agent: AgentID):
        obs = self.dynamics.observe(agent)
        obs["step"] = self.num_moves
        # lazily build
        obs["action_mask"] = self.dynamics.action_mask.build(agent)
        return obs

    def step(self, action: ActionType):
        if self.agent_selector.is_first():
            self._agent_actions = {}

        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0

        self._agent_actions[agent] = int(action)

        if self.agent_selector.is_last():
            self._execute_round()
        else:
            self._clear_rewards()

        self.agent_selection = self.agent_selector.next()
        self._accumulate_rewards()

    def _execute_round(self) -> None:
        """
        run once all agents have submitted actions for this step.

        Phase order:
            1. Validate actions (mask enforcement)
            2. Place-bomb phase   — PLACE_BOMB actions spawn bombs
            3. Movement phase     — FORWARD/BACKWARD/LEFT/RIGHT/STAY
            4. Detonation phase   — expired bombs produce AttackIntents
            5. Damage phase       — defenders receive damage; frozen_ticks set
            6. Upkeep             — bomb timers tick, resources/bombs update,
                                    freeze cooldowns tick / respawn
            7. Termination / truncation checks
            8. Pull rewards, rebuild observations, render
        """
        self.dynamics.rewards.clear_step()
        self.dynamics.mission_collectors_this_step.clear()

        validated_actions: dict[str, int] = {}
        for agent_id, action in self._agent_actions.items():
            agent_ent = self.dynamics.registry.get(agent_id)
            if getattr(agent_ent, "is_frozen", False):
                # stay put chill guy
                validated_actions[agent_id] = int(Action.STAY)
                continue
            mask = self.dynamics.action_mask.build(agent_id)
            if not mask[action]:
                self.dynamics.rewards.award(agent_id, "invalid_action")
                validated_actions[agent_id] = int(Action.STAY)
            else:
                validated_actions[agent_id] = action

        self.dynamics.place_bomb_phase(validated_actions)

        self.dynamics.resolve_movement_actions(validated_actions)

        intents = self.dynamics.detonation_phase()
        self.dynamics.damage_phase(intents)

        self.dynamics.upkeep()

        # this shouldn't be here
        for agent_id, action in validated_actions.items():
            agent_ent = self.dynamics.registry.get(agent_id)
            if getattr(agent_ent, "is_frozen", False):
                continue
            if Action(action) == Action.STAY:
                self.dynamics.rewards.award(agent_id, "stationary_penalty")

        # trunc
        self.num_moves += 1
        if self.num_moves >= self.num_iters:
            self.truncations = {a: True for a in self.agents}
            for a in self.agents:
                self.dynamics.rewards.award(a, "truncation")

        # money time
        step_rewards = self.dynamics.rewards.step_rewards()
        for a in self.agents:
            self.rewards[a] = step_rewards.get(a, 0.0)

        for a in self.agents:
            self.observations[a] = self.observe(a)
            self.infos[a] = self._build_info(a)

        if self.render_mode in self.metadata["render_modes"]:
            self.render()

    # ── termination ────────────────────────────────────────────────────────
    # "IMMORTALITY" - Argus MLBB
    #  ──────────────────────────────────────────────────────────────────────

    def _build_info(self, agent_id: str) -> dict:
        agent = self.dynamics.registry.get(agent_id)
        if not isinstance(agent, Agent):
            return {}
        bases = self.dynamics.registry.bases(agent.team)
        base_dist = (
            float(np.linalg.norm(agent.position - bases[0].position, ord=1))
            if bases
            else -1
        )
        return {
            "health": agent.health,
            "frozen_ticks": agent.frozen_ticks,
            "base_distance": base_dist,
            "base_health": bases[0].health if bases else 0.0,
            "team_resources": self.dynamics.team_resources.get(agent.team, 0.0),
            "team_bombs": self.dynamics.team_bombs.get(agent.team, 0),
            "add_mission": agent_id in self.dynamics.mission_collectors_this_step,
        }

    # ── render / close ─────────────────────────────────────────────────────

    def render(
        self,
        selected_agent_id: str | None = None,
        respawn_overlay: "np.ndarray | None" = None,
    ):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        return self.renderer.render(
            state=self.dynamics.state,
            registry=self.dynamics.registry,
            observations=self.observations,
            rewards=self.rewards,
            actions=self._agent_actions,
            num_moves=self.num_moves,
            agent_ids=self.agents,
            selected_agent_id=selected_agent_id,
            explosions=self.dynamics.last_explosions,
            respawn_overlay=respawn_overlay,
        )

    def close(self, replay_suffix: str | None = None):
        self.renderer.close(
            final_suffix=replay_suffix,
            final_episode_stats=self._collect_episode_stats(),
        )

    def finalise_current_replay(self, suffix: str | None = None) -> None:
        self.renderer.finalise_current_replay(suffix)

    def state(self):
        return self.dynamics.state
