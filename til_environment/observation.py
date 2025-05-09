"""
observation.py - Multi-channel observation builders for the Bomberman Bomberman.

Each agent's viewcone is an **H × W × C** ``float32`` tensor where every
channel is a semantically meaningful, human-readable layer.  The exact
channel layout is the canonical ``ViewChannel`` enum below; consumers
should index by enum (``view[..., ViewChannel.BOMB]``) rather than by
literal index, so additions are non-breaking.

Channel layout (aligned with ``ViewChannel`` — see the enum for the
authoritative source):

    Ch  Name                Range    Semantics
    ──  ──────────────────  ───────  ──────────────────────────────────────────
    0   VISIBLE             {0, 1}   1 if the cell is within line-of-sight
    1   WALL_RIGHT          {0, 1}   wall on right edge (destructible or structural)
    2   WALL_DOWN           {0, 1}   wall on bottom edge (destructible or structural)
    3   WALL_LEFT           {0, 1}   wall on left edge (destructible or structural)
    4   WALL_UP             {0, 1}   wall on top edge (destructible or structural)
    5   TILE_EMPTY          {0, 1}   no collectible on this visible cell
    6   TILE_RECON          {0, 1}   recon token present
    7   TILE_MISSION        {0, 1}   mission token present
    8   TILE_RESOURCE       {0, 1}   resource tile present (credits team pool on pickup)
    9   ALLY_AGENT          {0, 1}   agent belonging to the observer's team
    10  ENEMY_AGENT         {0, 1}   agent belonging to a different team
    11  ALLY_BASE           {0, 1}   base belonging to the observer's team
    12  ENEMY_BASE          {0, 1}   base belonging to a different team
    13  DESTR_WALL_RIGHT    {0, 1}   right edge exists AND is destructible
    14  DESTR_WALL_DOWN     {0, 1}   bottom edge exists AND is destructible
    15  DESTR_WALL_LEFT     {0, 1}   left edge exists AND is destructible
    16  DESTR_WALL_UP       {0, 1}   top edge exists AND is destructible
    17  ALLY_BOMB           {0, 1}   active allied bomb present on this cell
    18  ENEMY_BOMB          {0, 1}   active enemy bomb present on this cell
    19  ALLY_BOMB_TIMER     [0, ∞)   countdown ticks remaining on the allied bomb
    20  ENEMY_BOMB_TIMER    [0, ∞)   countdown ticks remaining on the enemy bomb
    21  ALLY_AGENT_HEALTH   [0, 1]   health ratio of ally agent on this cell (0 = frozen)
    22  ENEMY_AGENT_HEALTH  [0, 1]   health ratio of enemy agent on this cell (0 = frozen)
    23  ALLY_BASE_HEALTH    [0, 1]   health ratio of ally base on this cell
    24  ENEMY_BASE_HEALTH   [0, 1]   health ratio of enemy base on this cell

Wall channels (1–4) reflect all wall edges (destructible and structural).
Channels 13–16 (DESTR_WALL_*) mirror the wall channels but are only 1 when
that specific edge is also flagged destructible — agents can use these to
identify which edges a bomb will break.
TILE_EMPTY (5) is 1 for any visible cell that does not hold a collectible;
cells outside LOS have all channels at 0.

Public API
----------
* ``ViewChannel`` – IntEnum of channel indices (source of truth for ordering).
* ``NUM_CHANNELS`` – total channel count (= len(ViewChannel)); used for Box shapes.
* ``build_agent_viewcone`` – directional H×W×C tensor for an agent.
* ``build_radius_view``    – square (2R+1)²×C tensor for a radius-based entity.
* ``populate_channels``    – shared core; call directly in tests or custom viewers.
"""

from __future__ import annotations

import functools
from enum import IntEnum

import numpy as np

from til_environment.entities import EntityRegistry
from til_environment.helpers import (
    get_bit,
    idx_to_view,
    is_world_coord_valid,
    supercover_line,
    view_to_world,
)
from til_environment.types import Direction  # Wall kept for _WALL_TO_CHANNEL compat


# ═══════════════════════════════════════════════════════════════════════════
# Channel definitions
# ═══════════════════════════════════════════════════════════════════════════
class ViewChannel(IntEnum):
    """Index of each channel in the H × W × C observation tensor."""

    VISIBLE = 0

    # walls (one channel per cardinal direction)
    WALL_RIGHT = 1
    WALL_DOWN = 2
    WALL_LEFT = 3
    WALL_UP = 4

    # tile type (one-hot)
    TILE_EMPTY = 5
    TILE_RECON = 6
    TILE_MISSION = 7
    TILE_RESOURCE = 8

    # entities — ally / enemy for each type
    ALLY_AGENT = 9
    ENEMY_AGENT = 10
    ALLY_BASE = 11
    ENEMY_BASE = 12

    # breakable terrain — one channel per direction (mirrors WALL_* but only set when destructible)
    DESTR_WALL_RIGHT = 13
    DESTR_WALL_DOWN = 14
    DESTR_WALL_LEFT = 15
    DESTR_WALL_UP = 16

    # active bombs on the cell — split by team so belief can track each independently
    ALLY_BOMB = 17
    ENEMY_BOMB = 18
    ALLY_BOMB_TIMER = 19
    ENEMY_BOMB_TIMER = 20

    # normalized health ratios for agents and bases ([0, 1])
    ALLY_AGENT_HEALTH = 21
    ENEMY_AGENT_HEALTH = 22
    ALLY_BASE_HEALTH = 23
    ENEMY_BASE_HEALTH = 24


NUM_CHANNELS = len(ViewChannel)

# Wall presence bits 0–3 (Direction.value) → WALL_* channel.
_WALL_BIT_TO_CHANNEL: dict[int, ViewChannel] = {
    Direction.RIGHT.value: ViewChannel.WALL_RIGHT,
    Direction.DOWN.value:  ViewChannel.WALL_DOWN,
    Direction.LEFT.value:  ViewChannel.WALL_LEFT,
    Direction.UP.value:    ViewChannel.WALL_UP,
}

# Destructible flags bits 4–7 (Direction.value + 4) → DESTR_WALL_* channel.
_DESTR_BIT_TO_CHANNEL: dict[int, ViewChannel] = {
    Direction.RIGHT.value: ViewChannel.DESTR_WALL_RIGHT,
    Direction.DOWN.value:  ViewChannel.DESTR_WALL_DOWN,
    Direction.LEFT.value:  ViewChannel.DESTR_WALL_LEFT,
    Direction.UP.value:    ViewChannel.DESTR_WALL_UP,
}

# NOTE: _TILE_TO_CHANNEL is removed.  Tile presence (mission / recon) is now
# determined exclusively from the EntityRegistry, not from the state grid bits.


# ═══════════════════════════════════════════════════════════════════════════
# Vectorized LOS kernel
# ═══════════════════════════════════════════════════════════════════════════

def _encode_path_steps(
    path: list[tuple[int, int]],
) -> list[tuple[int, list[tuple[int, int, int]]]]:
    """Convert a supercover path into encoded wall checks.

    Each step is (step_type, checks):
      step_type=1  non-diagonal — checks = [(rx, ry, bit)]
      step_type=2  diagonal     — checks = [(h0), (h1), (v0), (v1)]

    Coordinates are relative to the path origin (agent position).
    ``bit`` ∈ {0=RIGHT, 1=DOWN} indexes the canonical direction stored in the
    state array (we always use the cell with the smaller coordinate).
    """
    steps = []
    for k in range(len(path) - 1):
        px, py = path[k]
        nx, ny = path[k + 1]
        sdx = nx - px
        sdy = ny - py

        if sdx != 0 and sdy != 0:
            # Diagonal step: four edge checks, blocked when (H0∨H1)∧(V0∨V1).
            # H0: horizontal edge (px,py)–(nx,py)
            # H1: vertical   edge (nx,py)–(nx,ny)
            # V0: vertical   edge (px,py)–(px,ny)
            # V1: horizontal edge (px,ny)–(nx,ny)
            h0 = (min(px, nx), py,          0)  # RIGHT bit of left cell
            h1 = (nx,          min(py, ny), 1)  # DOWN  bit of upper cell
            v0 = (px,          min(py, ny), 1)  # DOWN  bit of upper cell
            v1 = (min(px, nx), ny,          0)  # RIGHT bit of left cell
            steps.append((2, [h0, h1, v0, v1]))
        else:
            if sdx > 0:
                checks = [(px, py, 0)]       # RIGHT bit of left cell
            elif sdx < 0:
                checks = [(nx, py, 0)]       # RIGHT bit of left cell
            elif sdy > 0:
                checks = [(px, py, 1)]       # DOWN  bit of upper cell
            else:
                checks = [(px, ny, 1)]       # DOWN  bit of upper cell
            steps.append((1, checks))

    return steps


@functools.lru_cache(maxsize=8)
def _precompute_los_table(viewcone: tuple[int, int, int, int]) -> list[dict]:
    """Build vectorized LOS lookup tables for every Direction × viewcone cell.

    Cached by viewcone shape; called once per unique config at startup.

    Returns a list of 4 dicts (one per Direction) containing numpy arrays:
      ``world_rel``  (N, 2) int32  — target cell offset from agent, per cell
      ``step_type``  (N, P) int8   — 0=unused, 1=non-diagonal, 2=diagonal
      ``step_rx``    (N, P, 4) int32 — relative x for each of 4 sub-checks
      ``step_ry``    (N, P, 4) int32 — relative y
      ``step_bit``   (N, P, 4) uint8 — bit index (0=RIGHT, 1=DOWN)
      ``self_idx``   int            — flat index of the agent's own cell
    where N = vc_l × vc_w and P = max path length across all cells.
    """
    vc_l = viewcone[2] + viewcone[3] + 1   # rows (behind + ahead + 1)
    vc_w = viewcone[0] + viewcone[1] + 1   # cols (left  + right + 1)
    N = vc_l * vc_w

    origin = np.zeros(2, dtype=np.int64)
    tables = []

    for d in range(4):
        direction = Direction(d)

        world_rel = np.zeros((N, 2), dtype=np.int32)
        all_steps: list[list] = []
        self_idx = 0

        for n, (i, j) in enumerate(np.ndindex((vc_l, vc_w))):
            view_coord = np.array(
                [i - viewcone[2], j - viewcone[0]], dtype=np.int64
            )
            world_coord = view_to_world(origin, direction, view_coord)
            world_rel[n] = world_coord

            if (world_coord == 0).all():
                self_idx = n
                all_steps.append([])
                continue

            path = supercover_line(origin, world_coord)
            all_steps.append(_encode_path_steps(path))

        max_path = max((len(s) for s in all_steps), default=0)

        step_type = np.zeros((N, max_path), dtype=np.int8)
        step_rx   = np.zeros((N, max_path, 4), dtype=np.int32)
        step_ry   = np.zeros((N, max_path, 4), dtype=np.int32)
        step_bit  = np.zeros((N, max_path, 4), dtype=np.uint8)

        for n, steps in enumerate(all_steps):
            for k, (stype, checks) in enumerate(steps):
                step_type[n, k] = stype
                for q, (rx, ry, bit) in enumerate(checks):
                    step_rx[n, k, q] = rx
                    step_ry[n, k, q] = ry
                    step_bit[n, k, q] = bit

        tables.append({
            "world_rel": world_rel,
            "step_type": step_type,
            "step_rx":   step_rx,
            "step_ry":   step_ry,
            "step_bit":  step_bit,
            "self_idx":  self_idx,
            "vc_l":      vc_l,
            "vc_w":      vc_w,
        })

    return tables


def _vectorized_los(
    ax: int,
    ay: int,
    state: np.ndarray,
    table: dict,
    grid_size: int,
) -> np.ndarray:
    """Compute LOS visibility for all viewcone cells simultaneously.

    Returns a (N,) bool array where ``True`` means the cell is visible from
    ``(ax, ay)``.  Uses only numpy array ops — no Python loops over cells.
    """
    world_rel = table["world_rel"]   # (N, 2)
    step_type = table["step_type"]   # (N, P)
    step_rx   = table["step_rx"]     # (N, P, 4)
    step_ry   = table["step_ry"]     # (N, P, 4)
    step_bit  = table["step_bit"]    # (N, P, 4)

    # Target world coords and OOB mask.
    wx = ax + world_rel[:, 0]   # (N,)
    wy = ay + world_rel[:, 1]   # (N,)
    oob = (wx < 0) | (wx >= grid_size) | (wy < 0) | (wy >= grid_size)

    if step_type.shape[1] == 0:
        visible = ~oob
        visible[table["self_idx"]] = True
        return visible

    # Absolute check coordinates — clip for safe indexing; OOB cells are
    # already excluded by ``oob`` so clipped reads don't affect results.
    cx = np.clip(ax + step_rx, 0, grid_size - 1)   # (N, P, 4)
    cy = np.clip(ay + step_ry, 0, grid_size - 1)   # (N, P, 4)

    # Batch state lookup + bit extraction.
    state_vals = state[cx, cy]                              # (N, P, 4) uint8
    hits = ((state_vals >> step_bit) & np.uint8(1)).astype(bool)  # (N, P, 4)

    # Per-step blocking.
    is_valid   = step_type > 0                               # (N, P)
    is_nondiag = step_type == 1                              # (N, P)
    is_diag    = step_type == 2                              # (N, P)

    # Non-diagonal: blocked when hits[..., 0].
    nd_blocked = hits[..., 0]                                # (N, P)
    # Diagonal: blocked when (H0∨H1) ∧ (V0∨V1).
    dg_blocked = (hits[..., 0] | hits[..., 1]) & (hits[..., 2] | hits[..., 3])

    step_blocked = (is_nondiag & nd_blocked) | (is_diag & dg_blocked)  # (N, P)

    path_blocked = np.any(step_blocked & is_valid, axis=1)  # (N,)
    visible = ~oob & ~path_blocked
    visible[table["self_idx"]] = True                       # own cell always visible
    return visible


# ═══════════════════════════════════════════════════════════════════════════
# Line-of-sight (kept for build_radius_view and VisionSystem.is_visible)
# ═══════════════════════════════════════════════════════════════════════════
def _is_visible_los(
    start: np.ndarray,
    end: np.ndarray,
    walls: set[tuple[tuple[int, int], tuple[int, int]]],
) -> bool:
    """Return True if there is an unobstructed line of sight.

    All wall edges (destructible and indestructible alike) block LOS.
    Both types are present in ``walls`` until a destructible edge is removed
    by a bomb blast, at which point LOS opens naturally.
    """
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


# ═══════════════════════════════════════════════════════════════════════════
# Core channel-population routine
# ═══════════════════════════════════════════════════════════════════════════
def populate_channels(
    view: np.ndarray,
    visible_world_coords: list[tuple[int, int]],
    coord_to_idx: dict[tuple[int, int], tuple[int, int]],
    state: np.ndarray,
    registry: EntityRegistry,
    observer_team: int | None,
    grid_size: int,
) -> None:
    """Fill an ``H × W × C`` tensor *in place* for every visible world cell.

    Parameters
    ----------
    view : np.ndarray
        Pre-allocated zeros, shape ``(H, W, NUM_CHANNELS)``, dtype float32.
    visible_world_coords : list[tuple[int, int]]
        World ``(x, y)`` coordinates that passed the LOS check.
    coord_to_idx : dict
        Maps world ``(x, y)`` → view ``(row, col)`` index for stamping.
    state : np.ndarray
        Raw ``(grid_size, grid_size)`` uint8 arena.  Wall presence bits occupy
        positions 0–3 (aligned with ``Direction.value``); bits 4–7 encode
        which of those edges are destructible (same direction order).
    registry : EntityRegistry
        Active entity registry.
    observer_team : int | None
        Team index of the observing entity (used for ally / enemy tagging).
    grid_size : int
        Side-length of the world grid.
    """
    # ── 1. terrain (tile type + walls) for every visible cell ─────────────
    for wx, wy in visible_world_coords:
        idx = coord_to_idx.get((wx, wy))
        if idx is None:
            continue
        r, c = idx

        view[r, c, ViewChannel.VISIBLE] = 1.0

        tile_val = int(state[wx, wy])

        # bits 0–3: wall presence; bits 4–7: which of those edges are destructible
        for direction in Direction:
            if get_bit(tile_val, direction.value):
                view[r, c, _WALL_BIT_TO_CHANNEL[direction.value]] = 1.0
            if get_bit(tile_val, direction.value + 4):
                view[r, c, _DESTR_BIT_TO_CHANNEL[direction.value]] = 1.0

    # ── 2. entities ───────────────────────────────────────────────────────
    # Build a quick set for O(1) lookups
    visible_set = set(visible_world_coords)

    def _stamp(
        pos: np.ndarray,
        ally_ch: ViewChannel,
        enemy_ch: ViewChannel,
        entity_team: int | None,
    ) -> None:
        key = (int(pos[0]), int(pos[1]))
        if key not in visible_set:
            return
        idx = coord_to_idx.get(key)
        if idx is None:
            return
        r, c = idx
        if observer_team is not None and entity_team == observer_team:
            view[r, c, ally_ch] = 1.0
        else:
            view[r, c, enemy_ch] = 1.0

    for agent in registry.agents():
        _stamp(
            agent.position, ViewChannel.ALLY_AGENT, ViewChannel.ENEMY_AGENT, agent.team
        )
        key = (int(agent.position[0]), int(agent.position[1]))
        if key in visible_set:
            idx = coord_to_idx.get(key)
            if idx is not None:
                health_ratio = float(max(0.0, agent.health / agent.max_health))
                if observer_team is not None and agent.team == observer_team:
                    view[idx[0], idx[1], ViewChannel.ALLY_AGENT_HEALTH] = health_ratio
                else:
                    view[idx[0], idx[1], ViewChannel.ENEMY_AGENT_HEALTH] = health_ratio

    for base in registry.bases():
        _stamp(base.position, ViewChannel.ALLY_BASE, ViewChannel.ENEMY_BASE, base.team)
        key = (int(base.position[0]), int(base.position[1]))
        if key in visible_set:
            idx = coord_to_idx.get(key)
            if idx is not None:
                health_ratio = float(max(0.0, base.health / base.max_health))
                if observer_team is not None and base.team == observer_team:
                    view[idx[0], idx[1], ViewChannel.ALLY_BASE_HEALTH] = health_ratio
                else:
                    view[idx[0], idx[1], ViewChannel.ENEMY_BASE_HEALTH] = health_ratio

    for bomb in registry.bombs():
        key = (int(bomb.position[0]), int(bomb.position[1]))
        if key in visible_set:
            idx = coord_to_idx.get(key)
            if idx is not None:
                if observer_team is not None and bomb.team == observer_team:
                    view[idx[0], idx[1], ViewChannel.ALLY_BOMB] = 1.0
                    view[idx[0], idx[1], ViewChannel.ALLY_BOMB_TIMER] = float(bomb.timer)
                else:
                    view[idx[0], idx[1], ViewChannel.ENEMY_BOMB] = 1.0
                    view[idx[0], idx[1], ViewChannel.ENEMY_BOMB_TIMER] = float(bomb.timer)

    # ── 3. static collectibles ─────────────────────────────────────────────
    occupied_by_collectible: set[tuple[int, int]] = set()

    for mission in registry.missions():
        key = (int(mission.position[0]), int(mission.position[1]))
        if key in visible_set:
            idx = coord_to_idx.get(key)
            if idx is not None:
                view[idx[0], idx[1], ViewChannel.TILE_MISSION] = 1.0
                occupied_by_collectible.add(key)

    for recon in registry.recons():
        key = (int(recon.position[0]), int(recon.position[1]))
        if key in visible_set:
            idx = coord_to_idx.get(key)
            if idx is not None:
                view[idx[0], idx[1], ViewChannel.TILE_RECON] = 1.0
                occupied_by_collectible.add(key)

    for res in registry.resources():
        key = (int(res.position[0]), int(res.position[1]))
        if key in visible_set:
            idx = coord_to_idx.get(key)
            if idx is not None:
                view[idx[0], idx[1], ViewChannel.TILE_RESOURCE] = 1.0
                occupied_by_collectible.add(key)

    # TILE_EMPTY: visible cells with no static collectible
    for wx, wy in visible_world_coords:
        key = (wx, wy)
        if key not in occupied_by_collectible:
            idx = coord_to_idx.get(key)
            if idx is not None:
                view[idx[0], idx[1], ViewChannel.TILE_EMPTY] = 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Directional viewcone builder (for Agents)
# ═══════════════════════════════════════════════════════════════════════════
def build_agent_viewcone(
    agent_pos: np.ndarray,
    agent_dir: int,
    viewcone: tuple[int, int, int, int],
    state: np.ndarray,
    walls: set[tuple[tuple[int, int], tuple[int, int]]],
    registry: EntityRegistry,
    observer_team: int | None,
    grid_size: int,
) -> np.ndarray:
    """Build an ``H × W × C`` observation tensor for an agent's viewcone.

    Parameters
    ----------
    agent_pos : np.ndarray
        ``(x, y)`` world coordinate of the observing agent.
    agent_dir : int
        Facing direction index (see ``Direction``).
    viewcone : tuple[int, int, int, int]
        ``(left, right, behind, ahead)`` offsets defining the cone shape.
    state, walls, registry, observer_team, grid_size
        Forwarded to ``populate_channels``.

    Returns
    -------
    np.ndarray
        Shape ``(viewcone_length, viewcone_width, NUM_CHANNELS)``, float32.
    """
    vc_width = viewcone[0] + viewcone[1] + 1
    vc_length = viewcone[2] + viewcone[3] + 1

    view = np.zeros((vc_length, vc_width, NUM_CHANNELS), dtype=np.float32)

    tables = _precompute_los_table(viewcone)
    table = tables[agent_dir]
    ax, ay = int(agent_pos[0]), int(agent_pos[1])

    visible_flat = _vectorized_los(ax, ay, state, table, grid_size)  # (N,) bool

    world_rel = table["world_rel"]  # (N, 2)
    visible_coords: list[tuple[int, int]] = []
    coord_to_idx: dict[tuple[int, int], tuple[int, int]] = {}

    for n, (i, j) in enumerate(np.ndindex((vc_length, vc_width))):
        if not visible_flat[n]:
            continue
        wx = ax + int(world_rel[n, 0])
        wy = ay + int(world_rel[n, 1])
        key = (wx, wy)
        visible_coords.append(key)
        coord_to_idx[key] = (i, j)

    populate_channels(
        view,
        visible_coords,
        coord_to_idx,
        state,
        registry,
        observer_team,
        grid_size,
    )
    return view


# ═══════════════════════════════════════════════════════════════════════════
# Circular radius view builder (for Bases, Beacons, Scouts)
# ═══════════════════════════════════════════════════════════════════════════
def build_radius_view(
    center: np.ndarray,
    radius: int,
    state: np.ndarray,
    walls: set[tuple[tuple[int, int], tuple[int, int]]],
    registry: EntityRegistry,
    observer_team: int | None,
    grid_size: int,
) -> np.ndarray:
    """Build a ``(2R+1) × (2R+1) × C`` observation tensor centred on a position.

    Used for any entity with a circular ``vision_radius`` — bases, beacons,
    and scouts all use this.

    Parameters
    ----------
    center : np.ndarray
        ``(x, y)`` world coordinate of the observing entity.
    radius : int
        Vision radius in tiles (Chebyshev / square region, filtered by LOS).
    state, walls, registry, observer_team, grid_size
        Forwarded to ``populate_channels``.

    Returns
    -------
    np.ndarray
        Shape ``(2*radius+1, 2*radius+1, NUM_CHANNELS)``, float32.
    """
    side = 2 * radius + 1
    view = np.zeros((side, side, NUM_CHANNELS), dtype=np.float32)

    cx, cy = int(center[0]), int(center[1])

    # Reuse the vectorized kernel with a symmetric square viewcone facing RIGHT
    # (direction 0).  (R,R,R,R) produces offsets (-R..R, -R..R) at direction=0,
    # which matches the dx/dy iteration above exactly.
    sym_vc = (radius, radius, radius, radius)
    table = _precompute_los_table(sym_vc)[0]   # direction 0 = RIGHT

    visible_flat = _vectorized_los(cx, cy, state, table, grid_size)  # (N,) bool

    world_rel = table["world_rel"]  # (N, 2)
    visible_coords: list[tuple[int, int]] = []
    coord_to_idx: dict[tuple[int, int], tuple[int, int]] = {}

    for n, (i, j) in enumerate(np.ndindex((side, side))):
        if not visible_flat[n]:
            continue
        wx = cx + int(world_rel[n, 0])
        wy = cy + int(world_rel[n, 1])
        key = (wx, wy)
        visible_coords.append(key)
        coord_to_idx[key] = (i, j)

    populate_channels(
        view,
        visible_coords,
        coord_to_idx,
        state,
        registry,
        observer_team,
        grid_size,
    )
    return view
