"""
renderer.py - Rendering subsystem for the TIL Bomberman environment.

Encapsulates all pygame drawing logic so that the main Bomberman class only
needs to call ``Renderer.render()`` without knowing about surfaces, fonts,
or pixel math.

Design notes
------------
* The Renderer is **stateless with respect to game logic** – it reads the
  current world state, entity registry, and observations but never mutates
  them.
* Pygame initialisation is lazy: the display / surface is only created the
  first time ``render()`` is called.
* Both ``"human"`` and ``"rgb_array"`` render modes are supported.
"""

from __future__ import annotations

import colorsys
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pygame

from til_environment.entities import (
    Agent,
    Base,
    Bomb,
    EntityRegistry,
    Mission,
    Recon,
    Resource,
)
from itertools import zip_longest
from til_environment.helpers import get_bit, idx_to_view, view_to_world
from til_environment.observation import ViewChannel
from til_environment.types import Direction, Tile, Wall


def _json_default(o):
    """Fallback JSON serialiser for numpy scalars/arrays and pathlib.Path."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
# Centralised colour definitions so tweaking the look-and-feel is trivial.

COLORS = {
    "background": (255, 255, 255),
    "gridline": (211, 211, 211),
    "no_vision": (80, 80, 80),
    "wall": (0, 0, 0),
    "direction_indicator": (0, 255, 0),
    "text_default": "black",
    # entity colours
    "agent_team0": (0, 120, 255),
    "agent_team1": (255, 60, 60),
    "agent_team2": (60, 200, 60),
    "agent_team3": (255, 200, 0),
    "base": (160, 82, 45),
    "mission": (147, 112, 219),
    "recon": (255, 165, 0),
    "resource": (128, 128, 128),
    "destructible_wall": (160, 130, 85),
    "bomb": (30, 30, 30),
    "bomb_blast": (255, 80, 0),
    "frozen_overlay": (180, 180, 255),
}

# Fixed palette for the first four teams; teams beyond this are generated
# dynamically using golden-ratio hue spacing so they stay visually distinct.
_TEAM_COLOR_PALETTE: list[tuple[int, int, int]] = [
    COLORS["agent_team0"],  # 0 — blue
    COLORS["agent_team1"],  # 1 — red
    COLORS["agent_team2"],  # 2 — green
    COLORS["agent_team3"],  # 3 — yellow
]
_GOLDEN_RATIO = 0.618033988749895


def _team_color(team: int | None) -> tuple[int, int, int]:
    if team is None:
        return (200, 200, 200)
    if team < len(_TEAM_COLOR_PALETTE):
        return _TEAM_COLOR_PALETTE[team]
    # Extend palette infinitely: golden-ratio hue walk starting after the fixed entries
    hue = ((team - len(_TEAM_COLOR_PALETTE)) * _GOLDEN_RATIO + 0.45) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.80, 0.85)
    return (int(r * 255), int(g * 255), int(b * 255))


def chunk_list(lst, size):
    """
    Splits a list into sublists of given size using zip_longest.
    Fills missing values in the last chunk with None.
    """
    if not isinstance(lst, list):
        raise TypeError("Input must be a list.")
    if not isinstance(size, int) or size <= 0:
        raise ValueError("Chunk size must be a positive integer.")

    args = [iter(lst)] * size
    return list(zip_longest(*args))  # Returns tuples


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------
class Renderer:
    """Handles all pygame-based rendering for the Bomberman.

    Parameters
    ----------
    grid_size : int
        Number of cells along one side of the square grid.
    window_size : int
        Height (and width, when debug is off) of the pygame window in pixels.
    render_mode : str | None
        ``"human"``, ``"rgb_array"``, or ``None``.
    debug : bool
        If ``True`` an additional panel is drawn showing per-agent observations.
    viewcone_shape : tuple[int, int]
        (viewcone_length, viewcone_width) used for the debug observation view.
    viewcone : tuple[int, int, int, int]
        The raw viewcone offsets ``(left, right, behind, ahead)``.
    """

    def __init__(
        self,
        grid_size: int,
        window_size: int = 768,
        render_mode: str | None = None,
        debug: bool = False,
        viewcone_shape: tuple[int, int] = (7, 5),
        viewcone: tuple[int, int, int, int] = (2, 2, 2, 4),
        replay_dir: str | None = None,
        render_fps: int = 10,
        num_teams: int = 2,
    ) -> None:
        self.grid_size = grid_size
        self.window_size = window_size
        self.render_mode = render_mode
        self.debug = debug
        self.viewcone_shape = viewcone_shape
        self.viewcone = viewcone
        self.replay_dir = Path(replay_dir) if replay_dir else None
        # backup create folder if not
        if self.replay_dir is not None:
            self.replay_dir.mkdir(parents=True, exist_ok=True)
        self.render_fps = render_fps
        self.num_teams = num_teams

        self.window_width = int(window_size * 2) if debug else window_size

        # lazily initialised
        self._window: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._font: pygame.font.Font | None = None
        self._video_writer = None   # imageio writer, opened on first frame of each episode
        self._episode_video_path: Path | None = None  # path for the current episode
        self._episode_count = 0  # monotonic per-Renderer episode counter
        # Path of the most recently finalised mp4 (set by _finalise_writer).
        # Consumers (e.g. the eval callback) can read this after start_episode()
        # to correlate the just-closed episode with external metadata.
        self.last_finalised_path: Path | None = None

        # Cached background surface: gridlines + walls drawn once, reused until
        # the arena state changes (i.e. a destructible wall is blown up).
        self._bg_surface: pygame.Surface | None = None
        self._bg_state_key: bytes | None = None

        # Pre-allocated SRCALPHA surface for explosion cell overlays, sized to
        # one grid cell. Re-used every detonation frame to avoid per-cell alloc.
        self._explosion_cell_surf: pygame.Surface | None = None

    # -- public API ---------------------------------------------------------

    def render(
        self,
        state: np.ndarray,
        registry: EntityRegistry,
        observations: dict[str, dict],
        rewards: dict[str, float],
        actions: dict[str, int],
        num_moves: int,
        agent_ids: list[str],
        selected_agent_id: str | None = None,
        explosions: list[dict] | None = None,
        respawn_overlay: "np.ndarray | None" = None,
    ) -> np.ndarray | None:
        """Draw one frame.

        Returns ``None`` for human mode, or an RGB numpy array for
        ``"rgb_array"`` mode.
        """
        if self.render_mode is None:
            return None

        self._ensure_display()

        pix = self.window_size / self.grid_size

        # 1+2. Blit cached background (gridlines + walls).
        # Only rebuilt when the arena state changes (destructible wall blown up).
        self._blit_background(state, pix)

        # 3. explosion overlays — drawn before entities so agents/bases appear
        #    on top of the blast flash
        if explosions:
            self._draw_explosion_overlays(explosions, pix)

        # 4. entities from the registry
        self._draw_entities(registry, pix, selected_agent_id)

        # 5. debug panel (per-agent observation views)
        if self.debug:
            self._draw_debug_panel(
                observations, rewards, actions, num_moves, agent_ids, registry
            )

        # 6. optional respawn-timer overlay (toggled from play.py via T key)
        if respawn_overlay is not None:
            self._draw_respawn_overlay(respawn_overlay, pix)

        # 7. present
        return self._present()

    def start_episode(
        self,
        prev_suffix: str | None = None,
        prev_episode_stats: dict | None = None,
    ) -> None:
        """Finalise the current episode's video and prepare a fresh file path.

        Called by Bomberman.reset() so each episode gets its own timestamped
        MP4.  The writer itself is opened lazily on the first rendered frame so
        no empty files are created for episodes that are never rendered.

        Parameters
        ----------
        prev_suffix : str | None
            If provided, the just-closed episode's file is renamed to append
            this suffix (e.g., a score string) before the ``.mp4`` extension.
        prev_episode_stats : dict | None
            If provided, a JSON sidecar with the same stem as the just-closed
            episode's mp4 is written alongside it, containing these stats.
        """
        self._finalise_writer(prev_suffix, prev_episode_stats)
        if self.replay_dir is not None:
            self._episode_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self._episode_video_path = (
                self.replay_dir / f"{timestamp}_episode_{self._episode_count}.mp4"
            )
        else:
            self._episode_video_path = None

    def close(
        self,
        final_suffix: str | None = None,
        final_episode_stats: dict | None = None,
    ) -> None:
        """Tear down the pygame display and finalise any in-progress video.

        Parameters
        ----------
        final_suffix : str | None
            Optional suffix to append to the final episode's filename.
        final_episode_stats : dict | None
            If provided, a JSON sidecar is written alongside the final mp4.
        """
        self._finalise_writer(final_suffix, final_episode_stats)
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()
            self._window = None

    def finalise_current_replay(self, suffix: str | None) -> None:
        """Close and rename the current episode's video without teardown.

        Unlike ``close()`` this leaves the pygame window/surface alive so the
        env can keep rendering.  Useful when an external caller (e.g. the eval
        callback) wants to tag the just-closed episode's replay with a score.
        """
        self._finalise_writer(suffix)

    def _finalise_writer(
        self,
        suffix: str | None,
        episode_stats: dict | None = None,
    ) -> None:
        """Close the current video writer and optionally rename + sidecar it."""
        if self._video_writer is not None:
            self._video_writer.close()
            self._video_writer = None
        if (
            suffix
            and self._episode_video_path is not None
            and self._episode_video_path.exists()
        ):
            stem = self._episode_video_path.stem
            ext = self._episode_video_path.suffix
            renamed = self._episode_video_path.with_name(f"{stem}_{suffix}{ext}")
            self._episode_video_path.rename(renamed)
            self._episode_video_path = renamed

        # Sidecar JSON: written only when we actually produced an mp4 and the
        # caller handed us stats.  Same stem as the mp4, ``.json`` extension.
        if (
            episode_stats is not None
            and self._episode_video_path is not None
            and self._episode_video_path.exists()
        ):
            json_path = self._episode_video_path.with_suffix(".json")
            payload = {
                "mp4_filename": self._episode_video_path.name,
                "episode_index": self._episode_count,
                **episode_stats,
            }
            try:
                with open(json_path, "w") as f:
                    json.dump(payload, f, indent=2, default=_json_default)
            except OSError as e:
                warnings.warn(f"failed to write replay sidecar JSON {json_path}: {e}")

        # Record the mp4 path as just-finalised so external correlators can pick
        # it up, then clear the "current" pointer (next episode gets a new one).
        self.last_finalised_path = self._episode_video_path
        self._episode_video_path = None

    # -- lazy initialisation ------------------------------------------------

    def _ensure_display(self) -> None:
        if self._window is None:
            pygame.init()
            if self.render_mode == "human":
                self._window = pygame.display.set_mode(
                    (self.window_width, self.window_size)
                )
                pygame.display.set_caption("TIL-AI Bomberman")
            else:
                self._window = pygame.Surface((self.window_width, self.window_size))

        if self._clock is None and self.render_mode == "human":
            self._clock = pygame.time.Clock()

        if self._font is None:
            try:
                self._font = pygame.font.Font("freesansbold.ttf", 12)
            except Exception:
                warnings.warn("unable to import font")

    def _present(self) -> np.ndarray | None:
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            # Do NOT call clock.tick here — that would sleep inside env.step(),
            # blocking the caller. Pacing belongs in the outer game loop.

        needs_frame = self.render_mode == "rgb_array" or self.replay_dir is not None
        frame = (
            np.transpose(np.array(pygame.surfarray.pixels3d(self._window)), axes=(1, 0, 2))
            if needs_frame
            else None
        )

        if self._episode_video_path is not None and frame is not None:
            if self._video_writer is None:
                import imageio
                self._episode_video_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    self._episode_video_path.parent.touch(exist_ok=True)
                except (OSError, PermissionError) as e:
                    raise RuntimeError(
                        f"Cannot write to replay directory {self._episode_video_path.parent}: {e}"
                    ) from e
                self._video_writer = imageio.get_writer(
                    str(self._episode_video_path), fps=self.render_fps, codec="libx264", quality=7
                )
            self._video_writer.append_data(frame)

        return frame if self.render_mode == "rgb_array" else None

    # -- primitive drawing helpers ------------------------------------------

    def _draw_text(self, text: str, text_col: str = "black", **kwargs) -> None:
        if self._font is not None and self._window is not None:
            img = self._font.render(text, True, text_col)
            rect = img.get_rect(**kwargs)
            self._window.blit(img, rect)

    def _draw_respawn_overlay(self, respawn_map: "np.ndarray", pix: float) -> None:
        """Draw per-cell respawn timers as coloured tiles + white text.

        Colour ranges from green (fast centre) through yellow to red (slow edge),
        with a semi-transparent fill so the underlying entities remain visible.
        """
        if self._window is None or self._font is None:
            return

        vmin = int(respawn_map.min())
        vmax = int(respawn_map.max())
        span = max(vmax - vmin, 1)

        # Vectorized color computation — one numpy pass over the whole grid.
        t = (respawn_map.astype(np.float32) - vmin) / span  # (gs, gs), 0=fast 1=slow
        R = (220.0 * t).astype(np.uint8)
        G = (220.0 * (1.0 - t)).astype(np.uint8)
        B = np.full_like(R, 40)

        # Scale each cell to pix×pix pixels and build a (W, H, 3) surface array.
        pix_int = max(1, round(pix))
        R_big = np.repeat(np.repeat(R, pix_int, axis=0), pix_int, axis=1)
        G_big = np.repeat(np.repeat(G, pix_int, axis=0), pix_int, axis=1)
        B_big = np.repeat(np.repeat(B, pix_int, axis=0), pix_int, axis=1)
        rgb = np.stack([R_big, G_big, B_big], axis=-1)  # (W, H, 3)

        overlay = pygame.surfarray.make_surface(rgb)
        overlay.set_alpha(140)
        self._window.blit(overlay, (0, 0))

        # Text labels still need a per-cell loop (font.render is serial).
        half = pix / 2
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                label = self._font.render(str(int(respawn_map[x, y])), True, (255, 255, 255))
                lw, lh = label.get_size()
                self._window.blit(
                    label,
                    (int(x * pix + half - lw / 2), int(y * pix + half - lh / 2)),
                )

    def _draw_gridlines(
        self,
        max_x: int,
        max_y: int,
        square_size: float,
        x_corner: float = 0,
        y_corner: float = 0,
        width: int = 3,
        *,
        surface: pygame.Surface | None = None,
    ) -> None:
        target = surface if surface is not None else self._window
        assert target is not None
        for x in range(max_x + 1):
            pygame.draw.line(
                target,
                COLORS["gridline"],
                (x_corner + square_size * x, y_corner),
                (x_corner + square_size * x, y_corner + square_size * max_y),
                width=width,
            )
        for y in range(max_y + 1):
            pygame.draw.line(
                target,
                COLORS["gridline"],
                (x_corner, y_corner + square_size * y),
                (x_corner + square_size * max_x, y_corner + square_size * y),
                width=width,
            )

    # -- higher-level draw routines -----------------------------------------

    def _blit_background(self, state: np.ndarray, pix: float) -> None:
        """Blit the cached background (gridlines + walls) onto the window.

        The surface is rebuilt only when the arena state changes, which only
        happens when a destructible wall is destroyed by a bomb.  Every other
        frame is a single blit — O(1) regardless of grid size.
        """
        state_key = state.tobytes()
        if self._bg_surface is None or state_key != self._bg_state_key:
            surf = pygame.Surface((self.window_size, self.window_size))
            surf.fill(COLORS["background"])
            self._draw_gridlines(self.grid_size, self.grid_size, pix, surface=surf)
            self._draw_tiles(state, pix, surface=surf)
            self._bg_surface = surf
            self._bg_state_key = state_key
        self._window.blit(self._bg_surface, (0, 0))

    def _get_explosion_cell_surf(self, side: int) -> pygame.Surface:
        """Return the shared SRCALPHA cell surface, (re)creating it if the tile size changed."""
        if self._explosion_cell_surf is None or self._explosion_cell_surf.get_width() != side:
            self._explosion_cell_surf = pygame.Surface((side, side), pygame.SRCALPHA)
        return self._explosion_cell_surf

    def _draw_explosion_overlays(self, explosions: list[dict], pix: float) -> None:
        """Draw blast-cell fills and radius outlines for bombs that detonated this frame.

        Each blast cell gets a semi-transparent fill in a lightened variant of
        the owning team's colour, making it easy to see which team's bomb
        exploded and which cells were affected.  The blast-radius bounding box
        is also drawn so the full potential area is visible.
        """
        assert self._window is not None
        for exp in explosions:
            team_color = _team_color(exp["team"])
            # Lighten: blend 60% toward white for the fill tint.
            light_color = tuple(min(255, int(c * 0.4 + 255 * 0.6)) for c in team_color)

            # Cell fills — semi-transparent overlay.
            # Reuse one pre-allocated SRCALPHA surface to avoid per-cell alloc.
            side = int(pix)
            cell_surf = self._get_explosion_cell_surf(side)
            cell_surf.fill((*light_color, 170))
            for cell in exp["cells"]:
                x, y = int(cell[0]), int(cell[1])
                self._window.blit(cell_surf, (x * pix, y * pix))

            # Blast-radius bounding box outline (same as the live-bomb outline,
            # but drawn 2 px wide and in the team colour for the detonation frame).
            ox, oy = int(exp["origin"][0]), int(exp["origin"][1])
            r = exp["blast_radius"]
            r_pix = (r + 0.5) * pix
            cx_px = (ox + 0.5) * pix
            cy_px = (oy + 0.5) * pix
            blast_rect = pygame.Rect(cx_px - r_pix, cy_px - r_pix, 2 * r_pix, 2 * r_pix)
            pygame.draw.rect(self._window, team_color, blast_rect, width=2)

    def _draw_tiles(
        self,
        state: np.ndarray,
        pix: float,
        surface: pygame.Surface | None = None,
    ) -> None:
        """Draw wall lines from the raw state array.

        Static entities (missions, recon tokens) are drawn separately by
        ``_draw_entities`` via the EntityRegistry.  Wall presence is in bits
        0–3; bits 4–7 flag which of those edges are destructible (drawn in
        a different colour).
        """
        target = surface if surface is not None else self._window
        assert target is not None
        for x, y in np.ndindex((self.grid_size, self.grid_size)):
            tile_val = state[x, y]
            for wall in Wall:
                # Wall enum values are 4-7 (v1); subtract 4 to get the
                # presence bit (0-3) in the v2 _state encoding.
                # The same index without the -4 shift gives the destructible
                # flag bit (4-7).
                presence_bit = wall.value - 4
                if get_bit(tile_val, presence_bit):
                    color = (
                        COLORS["destructible_wall"]
                        if get_bit(tile_val, wall.value)
                        else 0
                    )
                    wall.draw(target, x, y, pix, color=color)

    def _draw_entities(
        self, registry: EntityRegistry, pix: float, selected_agent_id: str | None = None
    ) -> None:
        """Draw all entities from the registry onto the main grid."""
        assert self._window is not None

        for base in registry.bases():
            self._draw_base(base, pix)

        for mission in registry.missions():
            self._draw_mission(mission, pix)

        for recon in registry.recons():
            self._draw_recon(recon, pix)

        for resource in registry.resources():
            self._draw_resource(resource, pix)

        for bomb in registry.bombs():
            self._draw_bomb(bomb, pix)

        for agent in registry.agents():
            self._draw_agent(agent, pix, selected=agent.entity_id == selected_agent_id)

    def _draw_agent(self, agent: Agent, pix: float, selected: bool = False) -> None:
        assert self._window is not None
        color = _team_color(agent.team)
        cx = (agent.position[0] + 0.5) * pix
        cy = (agent.position[1] + 0.5) * pix

        if selected:
            pygame.draw.circle(
                self._window, (255, 255, 0), (cx, cy), pix / 3 + 4, width=3
            )

        # body — frozen agents get a blue-tinted desaturated body
        if agent.is_frozen:
            body_color = COLORS["frozen_overlay"]
            pygame.draw.circle(self._window, body_color, (cx, cy), pix / 3)
            pygame.draw.circle(self._window, color, (cx, cy), pix / 3, width=2)
        else:
            pygame.draw.circle(self._window, color, (cx, cy), pix / 3)

        # direction indicator
        direction = Direction(agent.direction)
        end = (
            (agent.position[0] + 0.5 + direction.movement[0] * 0.33) * pix,
            (agent.position[1] + 0.5 + direction.movement[1] * 0.33) * pix,
        )
        pygame.draw.line(self._window, COLORS["direction_indicator"], (cx, cy), end, 3)

        # label
        self._draw_text(agent.entity_id[-1], center=(cx, cy))

        # health bar (small bar above agent)
        bar_w = pix * 0.6
        bar_h = 4
        bar_x = cx - bar_w / 2
        bar_y = cy - pix / 3 - 6
        health_frac = agent.health / agent.max_health if agent.max_health > 0 else 0
        pygame.draw.rect(
            self._window, (80, 80, 80), pygame.Rect(bar_x, bar_y, bar_w, bar_h)
        )
        pygame.draw.rect(
            self._window,
            (0, 200, 0),
            pygame.Rect(bar_x, bar_y, bar_w * health_frac, bar_h),
        )

    def _draw_base(self, base: Base, pix: float) -> None:
        assert self._window is not None
        color = _team_color(base.team)
        rect = pygame.Rect(
            base.position[0] * pix + pix * 0.1,
            base.position[1] * pix + pix * 0.1,
            pix * 0.8,
            pix * 0.8,
        )

        # outline
        pygame.draw.rect(self._window, color, rect, width=3)

        # health bar — runs along the bottom edge of the outline rect

        health_frac = base.health / base.max_health if base.max_health > 0 else 0.0
        bar_h = max(4, int(pix * 0.08))
        bar_y = rect.bottom - bar_h
        # background track (dark)
        pygame.draw.rect(
            self._window,
            (60, 60, 60),
            pygame.Rect(rect.left, bar_y, rect.width, bar_h),
        )
        # filled portion: green → red as health drops
        bar_color = (
            int(255 * (1.0 - health_frac)),  # R: rises as health falls
            int(200 * health_frac),  # G: falls as health falls
            0,
        )
        pygame.draw.rect(
            self._window,
            bar_color,
            pygame.Rect(rect.left, bar_y, int(rect.width * health_frac), bar_h),
        )

        # small "B" label
        self._draw_text("B", center=(rect.centerx, rect.centery))

    def _draw_mission(self, mission: Mission, pix: float) -> None:
        assert self._window is not None
        cx = (mission.position[0] + 0.5) * pix
        cy = (mission.position[1] + 0.5) * pix
        pygame.draw.circle(self._window, COLORS["mission"], (cx, cy), pix / 6)

    def _draw_recon(self, recon: Recon, pix: float) -> None:
        assert self._window is not None
        cx = (recon.position[0] + 0.5) * pix
        cy = (recon.position[1] + 0.5) * pix
        pygame.draw.circle(self._window, COLORS["recon"], (cx, cy), pix / 10)

    def _draw_bomb(self, bomb: Bomb, pix: float) -> None:
        """Draw a placed bomb with team colour, shrinking orange fuse, and timer overlay."""
        assert self._window is not None
        cx = (bomb.position[0] + 0.5) * pix
        cy = (bomb.position[1] + 0.5) * pix
        team_color = _team_color(bomb.team)
        body_r = pix / 4

        # Blast-radius outline (square matches Chebyshev blast shape).
        r_pix = (bomb.blast_radius + 0.5) * pix
        blast_rect = pygame.Rect(cx - r_pix, cy - r_pix, 2 * r_pix, 2 * r_pix)
        pygame.draw.rect(self._window, COLORS["bomb_blast"], blast_rect, width=1)

        # Fuse — orange line from top of body, length proportional to timer.
        # Reference max: 8 ticks covers the typical post-__post_init__ range.
        _FUSE_MAX = 8
        fuse_frac = min(1.0, max(0.0, bomb.timer / _FUSE_MAX))
        fuse_max_len = pix * 0.38
        fuse_len = fuse_max_len * fuse_frac
        fuse_base = (int(cx), int(cy - body_r))
        if fuse_len >= 1.5:
            fuse_tip = (int(cx), int(cy - body_r - fuse_len))
            pygame.draw.line(self._window, (220, 110, 0), fuse_base, fuse_tip, 2)
            # Spark at tip: bright yellow dot
            pygame.draw.circle(self._window, (255, 240, 60), fuse_tip, 3)

        # Body: team-coloured fill so ownership is instantly obvious.
        pygame.draw.circle(self._window, team_color, (cx, cy), body_r)
        # Dark border for contrast.
        pygame.draw.circle(self._window, COLORS["bomb"], (cx, cy), body_r, width=2)

        # Timer number overlay.
        self._draw_text(str(bomb.timer), center=(cx, cy))

    def _draw_resource(self, resource: Resource, pix: float) -> None:
        assert self._window is not None
        cx = (resource.position[0] + 0.5) * pix
        cy = (resource.position[1] + 0.5) * pix
        pygame.draw.circle(self._window, COLORS["resource"], (cx, cy), pix / 8)

    # -- click hit-testing -------------------------------------------------

    def hit_test(
        self,
        mouse_x: float,
        mouse_y: float,
        registry: EntityRegistry,
    ):
        """Return the topmost entity at pixel coordinates ``(mouse_x, mouse_y)``.

        Checks entities in priority order: agents → bases → beacons → scouts
        → missions → recon → powerups.  Returns ``None`` if no entity is close
        enough to the click.

        Parameters
        ----------
        mouse_x, mouse_y:
            Pixel coordinates from a ``pygame.MOUSEBUTTONDOWN`` event.
        registry:
            The live ``EntityRegistry`` for the current episode.

        Returns
        -------
        Entity | None
        """
        pix = self.window_size / self.grid_size

        def _dist(entity) -> float:
            cx = (entity.position[0] + 0.5) * pix
            cy = (entity.position[1] + 0.5) * pix
            return ((mouse_x - cx) ** 2 + (mouse_y - cy) ** 2) ** 0.5

        for agent in registry.agents():
            if _dist(agent) <= pix / 3 + 6:
                return agent

        for base in registry.bases():
            bx = base.position[0] * pix + pix * 0.1
            by = base.position[1] * pix + pix * 0.1
            if bx <= mouse_x <= bx + pix * 0.8 and by <= mouse_y <= by + pix * 0.8:
                return base

        for bomb in registry.bombs():
            if _dist(bomb) <= pix / 4 + 4:
                return bomb

        for mission in registry.missions():
            if _dist(mission) <= pix / 6 + 4:
                return mission

        for recon in registry.recons():
            if _dist(recon) <= pix / 10 + 4:
                return recon

        for resource in registry.resources():
            if _dist(resource) <= pix / 8 + 4:
                return resource

        return None

    # -- debug panel --------------------------------------------------------

    def _draw_debug_panel(
        self,
        observations: dict[str, dict],
        rewards: dict[str, float],
        actions: dict[str, int],
        num_moves: int,
        agent_ids: list[str],
        registry: EntityRegistry,
    ) -> None:
        """Draw the right-hand debug panel with per-agent observations."""
        assert self._window is not None
        vc_len, vc_wid = self.viewcone_shape
        subpix = int(0.2 * self.window_size / vc_wid)
        """
        NOTE: We can the number panels per column to be 4, for the sake of visibility. Hence the coefficient 0.24 used to determine
        y_corner.
        Then, we do this for 2 columns only. If you need more just edit this function and get a massive screen i guess
        """
        cols = chunk_list(agent_ids, 4)

        for i, col in enumerate(cols):
            if i >= 2:
                break
            x_corner = int(self.window_size * (1.04 + 0.50 * i))
            x_lim = int(self.window_size * (1.47 + 0.50 * i))
            for j, agent_id in enumerate(col):
                if agent_id is None:
                    break
                obs = observations.get(agent_id)
                if obs is None:
                    continue

                y_corner = int(self.window_size * (0.24 * j + 0.04))
                self._draw_gridlines(vc_len, vc_wid, subpix, x_corner, y_corner)

                # text info
                agent_ent = registry.get(agent_id)
                hp_str = (
                    f"{agent_ent.health:.0f}/{agent_ent.max_health:.0f}"
                    if isinstance(agent_ent, Agent)
                    else "?"
                )
                # derive team-accurate colors for this agent's viewcone
                agent_team = agent_ent.team if isinstance(agent_ent, Agent) else 0
                ally_color = _team_color(agent_team)
                enemy_color = _team_color((agent_team + 1) % max(2, self.num_teams))
                for j, text in enumerate(
                    [
                        f"id: {agent_id}",
                        f"dir: {obs.get('direction', '?')}",
                        f"hp: {hp_str}",
                        f"reward: {rewards.get(agent_id, 0):.1f}",
                        f"loc: {obs.get('location', '?')}",
                        f"step {num_moves}: act={actions.get(agent_id, '?')}",
                    ]
                ):
                    self._draw_text(text, topright=(x_lim, y_corner + j * 15))

                # plot viewcone observation (multi-channel H×W×C tensor)
                viewcone = obs.get("agent_viewcone")
                if viewcone is not None:
                    vc_h, vc_w = viewcone.shape[0], viewcone.shape[1]

                    # Agent direction needed for both world-pos lookup and wall rotation.
                    agent_dir = (
                        agent_ent.direction
                        if isinstance(agent_ent, Agent)
                        else int(Direction.RIGHT)
                    )

                    # Precompute viewcone-index → world-coord mapping so entity
                    # colors can be resolved from the registry (correct for any
                    # number of teams, not just 2).
                    agent_pos_arr = np.array(obs.get("location", [0, 0]))
                    vc_to_world: dict[tuple[int, int], tuple[int, int]] = {}
                    for xi in range(vc_h):
                        for yi in range(vc_w):
                            wc = view_to_world(
                                agent_pos_arr,
                                Direction(agent_dir),
                                idx_to_view(np.array([xi, yi]), self.viewcone),
                            )
                            wx, wy = int(wc[0]), int(wc[1])
                            if 0 <= wx < self.grid_size and 0 <= wy < self.grid_size:
                                vc_to_world[(xi, yi)] = (wx, wy)

                    for x in range(vc_h):
                        for y in range(vc_w):
                            # visibility / tile type
                            if viewcone[x, y, ViewChannel.VISIBLE] < 0.5:
                                Tile.NO_VISION.draw(
                                    self._window, x, y, subpix, x_corner, y_corner, True
                                )
                            else:
                                if viewcone[x, y, ViewChannel.TILE_RECON] > 0.5:
                                    Tile.RECON.draw(
                                        self._window, x, y, subpix, x_corner, y_corner
                                    )
                                elif viewcone[x, y, ViewChannel.TILE_MISSION] > 0.5:
                                    Tile.MISSION.draw(
                                        self._window, x, y, subpix, x_corner, y_corner
                                    )
                                elif viewcone[x, y, ViewChannel.TILE_RESOURCE] > 0.5:
                                    Tile.RESOURCE.draw(
                                        self._window, x, y, subpix, x_corner, y_corner
                                    )

                            cx = x_corner + (x + 0.5) * subpix
                            cy = y_corner + (y + 0.5) * subpix
                            wpos = vc_to_world.get((x, y))

                            if any(
                                viewcone[x, y, ch] > 0.5
                                for ch in (
                                    ViewChannel.DESTR_WALL_RIGHT,
                                    ViewChannel.DESTR_WALL_DOWN,
                                    ViewChannel.DESTR_WALL_LEFT,
                                    ViewChannel.DESTR_WALL_UP,
                                )
                            ):
                                pygame.draw.rect(
                                    self._window,
                                    COLORS["destructible_wall"],
                                    pygame.Rect(
                                        x_corner + x * subpix + 1,
                                        y_corner + y * subpix + 1,
                                        subpix - 2,
                                        subpix - 2,
                                    ),
                                )

                            # Base — drawn as a small square in the base's actual team color.
                            if (
                                viewcone[x, y, ViewChannel.ALLY_BASE] > 0.5
                                or viewcone[x, y, ViewChannel.ENEMY_BASE] > 0.5
                            ):
                                base_color = ally_color  # fallback
                                if wpos is not None:
                                    for ent in registry.at(*wpos):
                                        if isinstance(ent, Base):
                                            base_color = _team_color(ent.team)
                                            break
                                elif viewcone[x, y, ViewChannel.ENEMY_BASE] > 0.5:
                                    base_color = enemy_color
                                side = subpix * 0.55
                                pygame.draw.rect(
                                    self._window,
                                    base_color,
                                    pygame.Rect(cx - side / 2, cy - side / 2, side, side),
                                )

                            # Agent — circle in the entity's actual team color.
                            if (
                                viewcone[x, y, ViewChannel.ALLY_AGENT] > 0.5
                                or viewcone[x, y, ViewChannel.ENEMY_AGENT] > 0.5
                            ):
                                agent_color = ally_color  # fallback
                                if wpos is not None:
                                    for ent in registry.at(*wpos):
                                        if isinstance(ent, Agent):
                                            agent_color = _team_color(ent.team)
                                            break
                                elif viewcone[x, y, ViewChannel.ENEMY_AGENT] > 0.5:
                                    agent_color = enemy_color
                                pygame.draw.circle(
                                    self._window, agent_color, (cx, cy), subpix / 3
                                )
                            ally_bomb = viewcone[x, y, ViewChannel.ALLY_BOMB] > 0.5
                            enemy_bomb = viewcone[x, y, ViewChannel.ENEMY_BOMB] > 0.5
                            if ally_bomb or enemy_bomb:
                                timer_val = int(viewcone[x, y, ViewChannel.ALLY_BOMB_TIMER if ally_bomb else ViewChannel.ENEMY_BOMB_TIMER])
                                bomb_r = subpix / 4
                                pygame.draw.circle(
                                    self._window, COLORS["bomb"], (cx, cy), bomb_r
                                )
                                pygame.draw.circle(
                                    self._window, COLORS["bomb_blast"], (cx, cy), bomb_r, width=1
                                )
                                # Mini fuse
                                _FUSE_MAX = 8
                                fuse_frac = min(1.0, max(0.0, timer_val / _FUSE_MAX))
                                fuse_len = subpix * 0.3 * fuse_frac
                                fuse_base = (int(cx), int(cy - bomb_r))
                                if fuse_len >= 1:
                                    fuse_tip = (int(cx), int(cy - bomb_r - fuse_len))
                                    pygame.draw.line(
                                        self._window, (220, 110, 0), fuse_base, fuse_tip, 1
                                    )
                                    pygame.draw.circle(
                                        self._window, (255, 240, 60), fuse_tip, 2
                                    )

                    # Walls pass (drawn on top).
                    # Wall channels are world-absolute (1=east,2=south,3=west,4=north).
                    # Rotate by agent direction so walls line up in the display frame.
                    for x in range(vc_h):
                        for y in range(vc_w):
                            for ch in (
                                ViewChannel.WALL_RIGHT,
                                ViewChannel.WALL_DOWN,
                                ViewChannel.WALL_LEFT,
                                ViewChannel.WALL_UP,
                            ):
                                if viewcone[x, y, ch] > 0.5:
                                    Wall(4 + (ch - 1 - agent_dir) % 4).draw(
                                        self._window, x, y, subpix, x_corner, y_corner
                                    )
