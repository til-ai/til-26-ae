"""
play.py - Play the game yourself!

Controls
--------
    W / ↑       — FORWARD
    S / ↓       — BACKWARD
    A / ←       — turn LEFT
    D / →       — turn RIGHT
    SPACE       — STAY
    B / F       — PLACE_BOMB
    R           — reset the environment
    T           — toggle respawn-timer overlay
    Q / ESC     — quit

    LEFT CLICK on an agent    — take control of that agent at next round
    LEFT CLICK on anything    — print entity info to terminal

Non selected entities do random actions.
Pass --verbose / -v to print action masks and per-step reward breakdowns.
"""

import argparse
import random

import pygame

from til_environment.actions import Action
from til_environment.config import default_config, load_config
from til_environment.entities import Agent, Bomb, Resource
from til_environment.bomberman_env import Bomberman


KEY_TO_AGENT_ACTION = {
    pygame.K_w: Action.FORWARD,
    pygame.K_UP: Action.FORWARD,
    pygame.K_s: Action.BACKWARD,
    pygame.K_DOWN: Action.BACKWARD,
    pygame.K_a: Action.LEFT,
    pygame.K_LEFT: Action.LEFT,
    pygame.K_d: Action.RIGHT,
    pygame.K_RIGHT: Action.RIGHT,
    pygame.K_SPACE: Action.STAY,
    pygame.K_b: Action.PLACE_BOMB,
    pygame.K_f: Action.PLACE_BOMB,
}

FILLER_ACTION = int(Action.STAY)


def _entity_info(entity) -> str:
    lines = [
        f"[{type(entity).__name__}] id={entity.entity_id}",
        f"  team={entity.team}  pos={entity.position.tolist()}",
    ]
    if isinstance(entity, Agent):
        lines += [
            f"  health={entity.health:.0f}/{entity.max_health:.0f}",
            f"  direction={entity.direction}",
            f"  frozen_ticks={entity.frozen_ticks}",
        ]
    elif isinstance(entity, Bomb):
        lines += [
            f"  timer={entity.timer}  attack={entity.attack}  "
            f"blast_radius={entity.blast_radius}",
            f"  placed_by={entity.attribute_rewards}",
        ]
    elif isinstance(entity, Resource):
        lines.append(f"  amount={entity.amount}")
    elif hasattr(entity, "health"):
        lines.append(f"  health={entity.health:.0f}/{entity.max_health:.0f}")
    if hasattr(entity, "reward_value"):
        lines.append(f"  reward_value={entity.reward_value}")
    return "\n".join(lines)


def _print_action_mask(env: Bomberman, agent_id: str) -> None:
    obs = env.observe(agent_id)
    mask = obs.get("action_mask")
    if mask is None:
        return
    available = [a.name for a in Action if mask[a]]
    blocked = [a.name for a in Action if not mask[a]]
    print(f"  ✓ {', '.join(available)}")
    if blocked:
        print(f"  ✗ {', '.join(blocked)}")
    print(
        f"  team_bombs={env.dynamics.team_bombs[env.dynamics.registry.get(agent_id).team]}  "
        f"team_resources={env.dynamics.team_resources[env.dynamics.registry.get(agent_id).team]:.2f}"
    )


def _handle_click(env: Bomberman, mouse_pos, player_agent: str, pending: str) -> str:
    entity = env.renderer.hit_test(mouse_pos[0], mouse_pos[1], env.dynamics.registry)
    if entity is None:
        print("[click] empty tile")
        return pending
    if isinstance(entity, Agent):
        if entity.entity_id != player_agent:
            print(f"[click] will control {entity.entity_id} from next round")
        else:
            print(f"[click] already controlling {entity.entity_id}")
        print(_entity_info(entity))
        return entity.entity_id
    print("[click]")
    print(_entity_info(entity))
    return pending


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file (defaults to til_environment/bomberman_config.yaml)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print action masks and per-step reward breakdowns.",
    )
    args = parser.parse_args()
    verbose: bool = args.verbose

    if args.config is None:
        cfg = default_config()
        cfg.env.render_mode = "human"
    else:
        cfg = load_config(args.config)

    cfg.env.novice = True
    env = Bomberman(cfg)
    env.reset(seed=random.randint(0, 99999))

    player_agent = env.possible_agents[0]
    pending_player_agent = player_agent
    print(f"Controlling: {player_agent}  (click another agent to switch)")

    running = True
    clock = pygame.time.Clock()
    show_respawn_overlay = False

    while running:
        agent = env.agent_selection

        is_new_round = env.agent_selector.is_first()
        if is_new_round:
            if pending_player_agent != player_agent:
                player_agent = pending_player_agent
                print(f"Now controlling: {player_agent}")

            _overlay = env.dynamics.respawn_map if show_respawn_overlay else None
            env.render(selected_agent_id=player_agent, respawn_overlay=_overlay)

            clock.tick(env.cfg.renderer.render_fps)

        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
            if all(env.terminations.values()) or all(env.truncations.values()):
                print("\n── episode over ──")
                for a in env.possible_agents:
                    print(f"  {a}  reward={env.rewards.get(a, 0):.2f}")
                print("Press R to reset or Q to quit.\n")
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            waiting = False
                        if event.type == pygame.KEYDOWN:
                            if event.key in (pygame.K_q, pygame.K_ESCAPE):
                                running = False
                                waiting = False
                            elif event.key == pygame.K_r:
                                env.reset(seed=random.randint(0, 99999))
                                player_agent = env.possible_agents[0]
                                pending_player_agent = player_agent
                                waiting = False
                        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                            pending_player_agent = _handle_click(
                                env, event.pos, player_agent, pending_player_agent
                            )
            continue

        if agent == player_agent:
            if verbose:
                _print_action_mask(env, agent)
            agent_action = None
            while agent_action is None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        agent_action = Action.STAY
                    if event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_q, pygame.K_ESCAPE):
                            running = False
                            agent_action = Action.STAY
                        elif event.key == pygame.K_r:
                            env.reset(seed=random.randint(0, 99999))
                            player_agent = env.possible_agents[0]
                            pending_player_agent = player_agent
                            agent_action = None
                            break
                        elif event.key == pygame.K_t:
                            show_respawn_overlay = not show_respawn_overlay
                            print(f"[respawn overlay] {'ON' if show_respawn_overlay else 'OFF'}")
                        elif event.key in KEY_TO_AGENT_ACTION:
                            agent_action = KEY_TO_AGENT_ACTION[event.key]
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        pending_player_agent = _handle_click(
                            env, event.pos, player_agent, pending_player_agent
                        )
                        _overlay = env.dynamics.respawn_map if show_respawn_overlay else None
                        env.render(selected_agent_id=player_agent, respawn_overlay=_overlay)
            if not running:
                break
            env.step(int(agent_action))
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_r:
                        env.reset(seed=random.randint(0, 99999))
                        player_agent = env.possible_agents[0]
                        pending_player_agent = player_agent
                    elif event.key == pygame.K_t:
                        show_respawn_overlay = not show_respawn_overlay
                        print(f"[respawn overlay] {'ON' if show_respawn_overlay else 'OFF'}")
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    pending_player_agent = _handle_click(
                        env, event.pos, player_agent, pending_player_agent
                    )
            env.step(FILLER_ACTION)

        if verbose:
            print(
                "step rewards:",
                {a: f"{r:.2f}" for a, r in env.dynamics.rewards._step.items()},
            )
            print(
                "ep rewards:",
                {a: f"{r:.2f}" for a, r in env.dynamics.rewards._episode.items()},
            )
    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
