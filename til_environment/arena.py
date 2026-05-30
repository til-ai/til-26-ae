"""
arena.py - Maze-based arena generation for the TIL Bomberman environment.

This module owns the procedural generation of the game map:

* ``ArenaGenerator`` – builds a randomised maze, places tile types, computes
  wall edge-sets, and assigns starting positions for agents and bases.
* ``WallResult``      – cached wall layout (wall grid + edge set).
* ``StaticEntitySpec`` – lightweight spec for a static entity to be spawned.
* ``ArenaResult``     – lightweight dataclass that bundles everything the
  ``Dynamics`` class needs to populate the world after a reset.

Generation is split into two stages:

1. **Wall generation** (``generate_walls``) – runs the Recursive Backtracking
   maze algorithm and produces a ``uint8`` grid whose bits 0–3 encode wall
   presence per edge (RIGHT=0, DOWN=1, LEFT=2, UP=3, matching
   ``Direction.value``).  Bits 4–7 flag which of those edges are destructible.
   This result is **cached** on the ``ArenaGenerator`` instance; it is only
   recomputed when the seed changes.

2. **Episode generation** (``generate_episode``) – takes the cached wall
   layout and independently scatters static entities (missions, recon tokens,
   powerups) and assigns starting positions for agents/bases.  Run on every
   ``Dynamics.reset()`` call so entities are fresh each episode even if walls
   are reused.

Extracted from ``dynamics.py`` for readability; Dynamics imports and
composes an ``ArenaGenerator`` instance.
"""

import logging
import math
from dataclasses import dataclass

import numpy as np
from gymnasium.utils.seeding import np_random
from mazelib import Maze
from mazelib.generate.DungeonRooms import DungeonRooms
from perlin_noise import PerlinNoise
from til_environment.helpers import convert_tile_to_edge, get_bit, is_world_coord_valid
from til_environment.types import Direction

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# WallEdge — first-class wall object with a canonical half-integer position
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class WallEdge:
    """A single wall edge between two adjacent grid tiles.

    *ax, ay* is always the tile with lower coordinates (the first element of
    the sorted ``((ax,ay),(bx,by))`` edge-key).  *direction* is therefore
    always RIGHT (dx=+1) or DOWN (dy=+1) — never LEFT or UP.  This gives every
    interior wall a single canonical object rather than a pair of bits spread
    across two tiles.

    *position* is the half-integer world-space midpoint, e.g. (4.5, 4) for
    a RIGHT wall starting at tile (4, 4).  This is used to query whether a wall
    falls within a viewcone or blast radius without referencing either tile
    separately.
    """

    ax: int
    ay: int
    direction: "Direction"  # RIGHT or DOWN only
    destructible: bool = False

    @property
    def bx(self) -> int:
        return self.ax + (1 if self.direction == Direction.RIGHT else 0)

    @property
    def by(self) -> int:
        return self.ay + (1 if self.direction == Direction.DOWN else 0)

    @property
    def position(self) -> tuple[float, float]:
        """Half-integer world-space midpoint of this wall edge."""
        if self.direction == Direction.RIGHT:
            return (self.ax + 0.5, float(self.ay))
        return (float(self.ax), self.ay + 0.5)


# ═══════════════════════════════════════════════════════════════════════════
# WallResult — cached output of the wall-generation stage
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class WallResult:
    """Output of the (cached) wall-generation stage.

    Attributes
    ----------
    wall_grid : np.ndarray
        ``(grid_size, grid_size)`` uint8 array.  Bits 0–3 encode wall
        presence per edge (RIGHT=0, DOWN=1, LEFT=2, UP=3).  Bits 4–7 flag
        which of those edges are destructible (same direction order).
    walls : set
        Set of ordered-pair edges ``((x0,y0),(x1,y1))`` for O(1) LOS
        look-ups.
    seed : int | None
        The RNG seed used to produce this layout (used to detect staleness).
    num_teams : int
        Team count baked into the layout (used to detect cache staleness
        when the symmetry fold changes between episodes).
    """

    wall_grid: np.ndarray
    walls: set[tuple[tuple[int, int], tuple[int, int]]]
    seed: int | None
    num_teams: int = 1


# ═══════════════════════════════════════════════════════════════════════════
# StaticEntitySpec — lightweight spawn descriptor for static entities
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class StaticEntitySpec:
    """Describes a single static entity to be placed on the map.

    ``kind`` is one of ``"mission"``, ``"recon"``, ``"resource"``, or
    ``"destructible_wall"``.  Multiple specs can share the same position
    (e.g. a mission and a resource on the same tile — though see
    ``generate_static_layer`` for the actual non-overlap rules).
    """

    kind: str
    position: np.ndarray


# ═══════════════════════════════════════════════════════════════════════════
# ArenaResult
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class ArenaResult:
    """Bundle returned by ``ArenaGenerator.generate_episode``.

    Attributes
    ----------
    wall_grid : np.ndarray
        ``(grid_size, grid_size)`` uint8 array.  Bits 0–3 encode wall
        presence; bits 4–7 flag destructible edges.  The canonical
        ``_state`` array used by Dynamics.
    walls : set
        Set of ordered-pair edges for O(1) collision look-ups.
    static_entities : list[StaticEntitySpec]
        Static entities to instantiate in the EntityRegistry (missions,
        recon tokens, powerups).  Dynamic entities (agents, bases, scouts,
        beacons) are NOT included here.
    starting_directions : np.ndarray
        ``(num_agents,)`` int array of initial facing directions.
    starting_locations : np.ndarray
        ``(num_agents, 2)`` int array of initial ``(x, y)`` positions.
    base_locations : np.ndarray
        ``(num_teams, 2)`` int array of base positions.
    """

    wall_grid: np.ndarray
    walls: set[tuple[tuple[int, int], tuple[int, int]]]
    static_entities: list[StaticEntitySpec]
    starting_directions: np.ndarray
    starting_locations: np.ndarray
    base_locations: np.ndarray
    respawn_map: np.ndarray


# ═══════════════════════════════════════════════════════════════════════════
# ArenaState — mutable wall state owned by Dynamics
# ═══════════════════════════════════════════════════════════════════════════
class ArenaState:
    """Encapsulates the mutable wall state of the arena.

    ``Dynamics`` composes one ``ArenaState`` and delegates all wall queries
    and mutations here, keeping wall logic out of the main ``Dynamics`` class.

    The ``_state`` array is a mutable ``uint8`` grid where bits 0–3 encode
    wall presence in each ``Direction`` (bit index == ``Direction.value``).
    The ``walls`` edge-set mirrors ``_state`` for O(1) LOS look-ups;
    the two are always kept in sync.

    Parameters
    ----------
    grid_size : int
        Side-length of the square grid.
    wall_grid : np.ndarray
        Canonical (immutable) wall grid produced by ``ArenaGenerator``.
        Stored as ``_arena``; ``_state`` is a mutable copy.
    walls : set
        Initial edge-set produced by ``ArenaGenerator``.
    """

    def __init__(
        self,
        grid_size: int,
        wall_grid: np.ndarray,
        walls: set[tuple[tuple[int, int], tuple[int, int]]],
    ) -> None:
        self.grid_size = grid_size
        self._arena: np.ndarray = wall_grid  # original — never mutated
        # _state bit layout:
        #   bits 0-3: wall presence  (RIGHT=0, BOTTOM=1, LEFT=2, TOP=3)
        #   bits 4-7: wall is destructible (same direction order)
        # A wall is destructible iff BOTH its presence bit AND its +4 bit are set.
        self._state: np.ndarray = wall_grid.copy()  # mutable working copy
        self.walls: set[tuple[tuple[int, int], tuple[int, int]]] = walls

        # wall_edges: one WallEdge per edge whose canonical tile (ax, ay) is
        # within the grid.  Boundary walls whose neighbour is out of bounds
        # (RIGHT/DOWN edges at the grid border) are included; LEFT/UP edges at
        # the top/left border have a negative canonical tile and are excluded.
        self.wall_edges: dict[tuple, WallEdge] = {}
        for edge in walls:
            (ax, ay), (bx, by) = edge
            if not (0 <= ax < grid_size and 0 <= ay < grid_size):
                continue
            dx, dy = bx - ax, by - ay
            canonical_dir = Direction.RIGHT if dx == 1 else Direction.DOWN
            d = canonical_dir.value
            destr = bool(int(wall_grid[ax, ay]) & (1 << (d + 4)))
            self.wall_edges[edge] = WallEdge(
                ax=ax, ay=ay, direction=canonical_dir, destructible=destr
            )

    # -- wall helpers ----------------------------------------------------------

    def _wall_edge(self, x: int, y: int, direction: Direction) -> tuple:
        """Canonical edge key for the wall between (x, y) and its neighbour."""
        mv = direction.movement
        return tuple(sorted(((x, y), (x + int(mv[0]), y + int(mv[1])))))

    def enforce_wall_collision(
        self, position: np.ndarray, direction: Direction
    ) -> bool:
        """Return True if moving *direction* from *position* is wall-blocked."""
        tile_val = self._state[tuple(position)]
        if get_bit(tile_val, direction.value):
            return True
        next_loc = position + direction.movement
        if not is_world_coord_valid(next_loc, self.grid_size):
            return True
        next_tile_val = self._state[tuple(next_loc)]
        if get_bit(next_tile_val, (direction.value + 2) % 4):
            return True
        return False

    def is_destructible(self, x: int, y: int, direction: Direction) -> bool:
        """Return True if the wall edge in *direction* from (x, y) is destructible."""
        key = self._wall_edge(x, y, direction)
        edge = self.wall_edges.get(key)
        return edge is not None and edge.destructible

    def set_destructible(
        self, x: int, y: int, direction: Direction, value: bool = True
    ) -> None:
        """Set or clear the destructible flag on the wall edge from (x, y).

        Updates both the ``WallEdge`` object and the ``_state`` bits so every
        consumer (blast, observation, renderer) stays in sync.  No-op if no
        wall exists at this edge.
        """
        key = self._wall_edge(x, y, direction)
        we = self.wall_edges.get(key)
        if we is None:
            return
        we.destructible = value
        b = direction.value
        mv = direction.movement
        nx, ny = x + int(mv[0]), y + int(mv[1])
        if value:
            self._state[x, y] = np.uint8(int(self._state[x, y]) | (1 << (b + 4)))
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                opp_b = (b + 2) % 4
                self._state[nx, ny] = np.uint8(
                    int(self._state[nx, ny]) | (1 << (opp_b + 4))
                )
        else:
            self._state[x, y] = np.uint8(
                int(self._state[x, y]) & ~(1 << (b + 4)) & 0xFF
            )
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                opp_b = (b + 2) % 4
                self._state[nx, ny] = np.uint8(
                    int(self._state[nx, ny]) & ~(1 << (opp_b + 4)) & 0xFF
                )

    @property
    def indestructible_walls(self) -> frozenset:
        """Wall edges that cannot be destroyed by bombs."""
        return frozenset(
            key for key, we in self.wall_edges.items() if not we.destructible
        )

    def destroy_wall(self, x: int, y: int, direction: Direction) -> bool:
        """Remove the wall on the *direction* edge of tile (x, y).

        Clears both the presence bit and the destructible bit.
        ``_state``, ``walls``, and ``wall_edges`` are all updated atomically.
        Returns True if a wall was present and removed; False if no-op.
        """
        bit = direction.value
        if not get_bit(int(self._state[x, y]), bit):
            return False
        clear = ~(1 << bit) & ~(1 << (bit + 4)) & 0xFF
        self._state[x, y] = np.uint8(int(self._state[x, y]) & clear)
        mv = direction.movement
        nx, ny = x + int(mv[0]), y + int(mv[1])
        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
            opp_bit = (direction.value + 2) % 4
            opp_clear = ~(1 << opp_bit) & ~(1 << (opp_bit + 4)) & 0xFF
            self._state[nx, ny] = np.uint8(int(self._state[nx, ny]) & opp_clear)
        edge_key = self._wall_edge(x, y, direction)
        self.walls.discard(edge_key)
        self.wall_edges.pop(edge_key, None)
        return True

    def destroy_destructible_walls_in_radius(self, cells: list[np.ndarray]) -> int:
        """Destroy every destructible wall edge touching any cell in *cells*.

        Returns the number of edges removed (each shared edge counted once).
        Called by the detonation phase; only destructible edges are affected —
        indestructible maze walls survive bomb blasts.
        """
        count = 0
        for cell in cells:
            x, y = int(cell[0]), int(cell[1])
            if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
                continue
            for direction in Direction:
                if self.is_destructible(x, y, direction):
                    self.destroy_wall(x, y, direction)
                    count += 1
        return count

    def add_wall(self, x: int, y: int, direction: Direction) -> bool:
        """Add a wall on the *direction* edge of tile (x, y).

        ``_state``, ``walls``, and ``wall_edges`` are all updated atomically.
        Returns True if the wall was added; False if already present (no-op).
        """
        bit = direction.value
        if get_bit(int(self._state[x, y]), bit):
            return False
        self._state[x, y] = np.uint8(int(self._state[x, y]) | (1 << bit))
        mv = direction.movement
        nx, ny = x + int(mv[0]), y + int(mv[1])
        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
            opp_bit = (direction.value + 2) % 4
            self._state[nx, ny] = np.uint8(int(self._state[nx, ny]) | (1 << opp_bit))
        edge_key = self._wall_edge(x, y, direction)
        self.walls.add(edge_key)
        if edge_key not in self.wall_edges:
            (ax, ay), (bx, by) = edge_key
            if 0 <= ax < self.grid_size and 0 <= ay < self.grid_size:
                dx, dy = bx - ax, by - ay
                canonical_dir = Direction.RIGHT if dx == 1 else Direction.DOWN
                self.wall_edges[edge_key] = WallEdge(
                    ax=ax, ay=ay, direction=canonical_dir, destructible=False
                )
        return True


# ═══════════════════════════════════════════════════════════════════════════
# ArenaGenerator
# ═══════════════════════════════════════════════════════════════════════════
class ArenaGenerator:
    """Encapsulates maze-based arena generation.

    Generation is split into two stages for efficiency:

    1. ``generate_walls(rng, rng_seed)`` — expensive maze algorithm.
       Result is cached and reused until the seed changes.
    2. ``generate_episode(rng, ...)`` — cheap static-entity scatter +
       start-position assignment.  Always runs on every episode reset.

    Parameters
    ----------
    grid_size : int
        Side-length of the square grid.
    wall_prob : float
        Fraction of interior maze walls to randomly knock down.
    mission_prob : float
        Per-tile probability of spawning a Mission entity.
    recon_prob : float
        Per-tile probability of spawning a Recon token (independent of
        mission_prob; a cell may become either mission or recon but not both
        — the outcome is drawn from a multinomial).
    powerup_prob : float
        Per-tile probability of spawning a PowerUp entity.  Applied as a
        second independent roll, so a tile can have a powerup alongside a
        mission or recon token.
    novice : bool
        If True, use a fixed seed for reproducible maps.
    """

    def __init__(
        self,
        grid_size: int = 16,
        wall_prob: float = 0.8,
        wall_destructible_ratio: float = 0.5,
        mission_prob: float = 0.2,
        recon_prob: float = 0.15,
        resource_prob: float = 0.08,
        novice: bool = False,
        base_respawn_steps: int = 40,
    ) -> None:
        self.grid_size = grid_size
        self.wall_drop_prob = 1 - wall_prob
        self.wall_destructible_ratio = wall_destructible_ratio
        self.mission_prob = mission_prob
        self.recon_prob = recon_prob
        self.resource_prob = resource_prob
        self.novice = novice
        self.base_respawn_steps = base_respawn_steps
        self._maze = Maze()

        # cached wall layout — None until first generate_walls() call
        self._wall_cache: WallResult | None = None

    # -- public API ---------------------------------------------------------

    def generate_walls(
        self,
        rng: np.random.Generator,
        rng_seed: int | None,
        num_teams: int = 1,
        phase_offset: float = 0.0,
    ) -> WallResult:
        """Generate (or retrieve from cache) the wall layout for a given seed.

        The maze algorithm is only re-run when the seed or ``num_teams``
        changes, or when no cached result exists.  Rooms are pre-carved at
        each team's approximate spawn location so corridors lead naturally
        into the base areas.

        Returns
        -------
        WallResult
            Wall grid and edge set.
        """
        if (
            self._wall_cache is not None
            and self._wall_cache.seed == rng_seed
            and self._wall_cache.num_teams == num_teams
            and not self.novice
        ):
            return self._wall_cache

        if self.novice:
            rng_for_maze, _ = np_random(19)
            self._maze.set_seed(19 % 2**32)
        else:
            rng_for_maze = rng
            self._maze.set_seed((rng_seed or 0) % 2**32)

        self._generate_maze_with_timeout(
            rng_for_maze, num_teams=num_teams, phase_offset=phase_offset
        )
        # knock down some walls
        grid = self._maze.grid.copy()
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 0
        grid *= np.indices(grid.shape).sum(axis=0) % 2
        wall_idx = np.where(grid == 1)
        if wall_idx[0].shape[0] > 0:
            drop = rng.choice(
                wall_idx[0].shape[0],
                size=int(wall_idx[0].shape[0] * self.wall_drop_prob),
                replace=False,
            )
            self._maze.grid[wall_idx[0][drop], wall_idx[1][drop]] = 0

        # Bits 0–3: wall presence (RIGHT=0, DOWN=1, LEFT=2, UP=3 matching Direction.value).
        # Bits 4–7: destructible flag for the same four edges.
        wall_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        convert_tile_to_edge(wall_grid, self._maze.grid)

        walls = self._build_wall_set(wall_grid)

        # Mark a fraction of walls as destructible
        self._mark_destructible_walls(wall_grid, walls, rng, num_teams)

        result = WallResult(
            wall_grid=wall_grid, walls=walls, seed=rng_seed, num_teams=num_teams
        )
        self._wall_cache = result
        return result

    def generate_static_layer(
        self,
        rng: np.random.Generator,
        wall_grid: np.ndarray,
        num_teams: int = 1,
        phase_offset: float = 0.0,
    ) -> list[StaticEntitySpec]:
        """Scatter collectibles with N-fold rotational symmetry.

        Pass A — sector-0 cells, centre-biased Bernoulli, rotate to all N sectors.
            Sector-0 is the angular wedge ``[phase_offset, phase_offset + 2π/N)``.
            Each cell's placement probability scales from 2× at the grid centre
            to 0.5× at corners.  Winning cells are batched; types are assigned
            in **fixed proportion** (mission : recon : resource as configured)
            using residual rounding, so the type mix is consistent across
            episodes regardless of how many cells fired.

        Pass B — centre fill (radius ≤ 2.5), no rotation.
            Any centre cell not yet occupied gets the same Bernoulli roll; types
            assigned with the same budget method.
        """
        specs: list[StaticEntitySpec] = []

        mp = float(np.clip(self.mission_prob, 0.0, 1.0))
        rp = float(np.clip(self.recon_prob, 0.0, 1.0))
        sp = float(np.clip(self.resource_prob, 0.0, 1.0))
        base_coll = mp + rp + sp
        if base_coll > 1.0:
            mp /= base_coll
            rp /= base_coll
            sp /= base_coll
            base_coll = 1.0

        # Raw (already-clipped) probability per type name — normalization is
        # deferred to _assign_types so subsets renormalize automatically.
        type_probs: dict[str, float] = {
            "mission": mp,
            "recon": rp,
            "resource": sp,
        }

        ALL_WALLS_V2 = 0x0F
        gs = self.grid_size
        cx = (gs - 1) / 2.0
        cy = (gs - 1) / 2.0
        max_dist = math.sqrt(2.0) * (gs - 1) / 2.0
        theta_sector = 2.0 * math.pi / max(num_teams, 1)

        placed: set[tuple[int, int]] = set()

        def _add(kind: str, tx: int, ty: int, overwrite: bool = False) -> bool:
            if (wall_grid[tx, ty] & ALL_WALLS_V2) == ALL_WALLS_V2:
                return False
            if (tx, ty) in placed:
                if not overwrite:
                    return False
                # Remove the existing spec so the centre placement takes priority.
                for i in range(len(specs) - 1, -1, -1):
                    p = specs[i].position
                    if int(p[0]) == tx and int(p[1]) == ty:
                        specs.pop(i)
                        break
                placed.discard((tx, ty))
            specs.append(StaticEntitySpec(kind=kind, position=np.array([tx, ty])))
            placed.add((tx, ty))
            return True

        # def _placement_prob(x: int, y: int) -> float:
        #     d_norm = min(1.0, math.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max_dist)
        #     scale = 0.5 + 1.5 * (1.0 - d_norm)  # 2.0 at centre, 0.5 at corners
        #     return min(0.9, base_coll * scale)

        def _assign_types(
            cells: list[tuple[int, int]],
            rotate: bool,
            type_names: list[str] = ["mission", "recon", "resource"],
            overwrite: bool = False,
        ) -> None:
            """Assign types to *cells* via residual rounding.

            Fractions are derived from ``type_probs`` restricted to *type_names*
            and renormalized, so any subset of types is handled correctly.
            When *overwrite* is True, cells already occupied by a pass-A tile
            are replaced rather than skipped.
            """
            n = len(cells)
            if n == 0:
                return

            raw_probs = [type_probs.get(t, 0.0) for t in type_names]
            total = sum(raw_probs)
            if total > 0.0:
                fracs = [p / total for p in raw_probs]
            else:
                fracs = [1.0 / len(type_names)] * len(type_names)

            raw = [f * n for f in fracs]
            counts = [int(r) for r in raw]
            deficit = n - sum(counts)
            residuals = [r - c for r, c in zip(raw, counts)]
            for i in sorted(range(len(residuals)), key=lambda i: -residuals[i])[
                :deficit
            ]:
                counts[i] += 1

            shuffled = list(cells)
            rng.shuffle(shuffled)
            idx = 0
            for kind, cnt in zip(type_names, counts):
                for _ in range(cnt):
                    if idx >= len(shuffled):
                        break
                    x, y = shuffled[idx]
                    idx += 1
                    if rotate:
                        for k in range(num_teams):
                            if k == 0:
                                _add(kind, x, y, overwrite=overwrite)
                            else:
                                rc = self._rotate_cell(x, y, k * theta_sector)
                                if rc is not None:
                                    _add(kind, rc[0], rc[1], overwrite=overwrite)
                    else:
                        _add(kind, x, y, overwrite=overwrite)

        # --- Pass A: sector-0 cells, rotate to all sectors ---
        sector0_hits: list[tuple[int, int]] = []
        for x, y in np.ndindex((gs, gs)):
            if (wall_grid[x, y] & ALL_WALLS_V2) == ALL_WALLS_V2:
                continue
            angle = (math.atan2(y - cy, x - cx) - phase_offset) % (2.0 * math.pi)
            if angle >= theta_sector:
                continue
            # if rng.random() < _placement_prob(x, y):
            sector0_hits.append((x, y))
        _assign_types(sector0_hits, rotate=True)

        # --- Pass B: centre fill (radius ≤ 2.5) — overrides pass A placements ---
        centre_hits: list[tuple[int, int]] = []
        for x, y in np.ndindex((gs, gs)):
            if math.sqrt((x - cx) ** 2 + (y - cy) ** 2) > 2.5:
                continue
            if (wall_grid[x, y] & ALL_WALLS_V2) == ALL_WALLS_V2:
                continue
            centre_hits.append((x, y))
        _assign_types(
            centre_hits,
            rotate=False,
            type_names=["mission", "resource"],
            overwrite=True,
        )

        return specs

    def generate_respawn_map(self, rng_seed: int | None) -> np.ndarray:
        """Build a per-cell tile-respawn delay map using Perlin noise + a centre gradient.

        The result is a ``(grid_size, grid_size)`` int32 array where each value
        is the number of steps before a collected tile at that position reappears.
        Centre cells receive the shortest delays; peripheral cells receive delays
        up to ``base_respawn_steps``.

        The approach (adapted from the noise.ipynb demo):
          1. Sample a Perlin noise field; boost the inner 4×4 region to ensure
             centre values are always positive.
          2. Build a circular gradient: 1.0 at the centre, 0.0 at the corners.
          3. Multiply noise × gradient and amplify positive values so the centre
             "plateau" dominates.
          4. Normalise to [0, 1] and map linearly to
             [``base_respawn_steps``//4, ``base_respawn_steps``].
        """

        gs = self.grid_size
        base = self.base_respawn_steps
        min_steps = max(base // 4, 5)
        seed = (int(rng_seed) & 0x7FFFFFFF) if rng_seed is not None else 0

        # ── 1. Perlin noise ─────────────────────────────────────────────────
        noise_fn = PerlinNoise(octaves=4, seed=seed)
        raw = np.array(
            [[noise_fn([i / gs, j / gs]) for j in range(gs)] for i in range(gs)],
            dtype=np.float64,
        )
        # Centre bias: force inner 4×4 positive so they always land fast.
        c = gs // 2
        for i in range(c - 2, c + 2):
            for j in range(c - 2, c + 2):
                if 0 <= i < gs and 0 <= j < gs:
                    raw[i, j] = abs(raw[i, j]) + 0.2
        raw -= raw.min()
        raw /= raw.max() + 1e-9

        # ── 2. Circular centre gradient ─────────────────────────────────────
        cx, cy = (gs - 1) / 2.0, (gs - 1) / 2.0
        ys, xs = np.mgrid[0:gs, 0:gs]
        dist = np.sqrt((xs.astype(float) - cx) ** 2 + (ys.astype(float) - cy) ** 2)
        grad = 1.0 - dist / (dist.max() + 1e-9)  # 1 at centre, 0 at corners
        grad = np.where(grad > 0, grad * 20.0, grad)
        grad /= grad.max() + 1e-9
        grad = np.clip(grad, 0.0, 0.7)
        grad /= grad.max() + 1e-9  # cap becomes 1.0

        # ── 3. Combine ──────────────────────────────────────────────────────
        combined = raw * grad
        combined = np.where(combined > 0, combined * 5.0, combined)
        combined -= combined.min()
        combined /= combined.max() + 1e-9  # [0, 1], high = centre-like

        # ── 4. Map to steps ─────────────────────────────────────────────────
        # combined = 1.0 → min_steps  (fast centre)
        # combined = 0.0 → base       (slow periphery)
        respawn_map = np.round(
            min_steps + (base - min_steps) * (1.0 - combined)
        ).astype(np.int32)

        return respawn_map

    def generate_episode(
        self,
        rng: np.random.Generator,
        rng_seed: int | None,
        num_teams: int = 2,
    ) -> ArenaResult:
        """Generate a complete episode setup.

        Calls (cached) ``generate_walls`` for the structural layout, then
        freshly generates static entity placements and start positions.

        Parameters
        ----------
        rng : np.random.Generator
        rng_seed : int | None
            Seed used to index into the wall cache.
        num_teams : int
            Number of competing teams (one agent + one base each).

        Returns
        -------
        ArenaResult
        """
        # Draw phase_offset first so the same value is used for both the maze
        # room positions (generate_walls) and the spawn/collectible placement.
        phase_offset = rng.uniform(0.0, 2.0 * math.pi / max(num_teams, 1))

        wall_result = self.generate_walls(
            rng, rng_seed, num_teams=num_teams, phase_offset=phase_offset
        )
        # Copy so we can mutate without corrupting the cache.
        wall_grid = wall_result.wall_grid.copy()
        walls: set = set(wall_result.walls)

        starting_directions = rng.integers(0, 4, size=num_teams)
        base_locations, starting_locations = self._spread_spawns(
            rng,
            num_teams,
            wall_grid,
            phase_offset=phase_offset,
        )
        self._ensure_base_destructible_walls(wall_grid, base_locations)

        static_entities = self.generate_static_layer(
            rng, wall_grid, num_teams=num_teams, phase_offset=phase_offset
        )
        static_entities = self._deconflict_spawn_tiles(
            static_entities, starting_locations, base_locations
        )

        respawn_map = self.generate_respawn_map(rng_seed=rng_seed)

        return ArenaResult(
            wall_grid=wall_grid,
            walls=walls,
            static_entities=static_entities,
            starting_directions=starting_directions,
            starting_locations=starting_locations,
            base_locations=base_locations,
            respawn_map=respawn_map,
        )

    # -- kept for backward compatibility -----------------------------------

    def generate(
        self,
        rng: np.random.Generator,
        rng_seed: int | None,
        num_teams: int = 2,
    ) -> ArenaResult:
        """Backward-compatible alias for ``generate_episode``."""
        return self.generate_episode(rng, rng_seed, num_teams)

    # -- internal -----------------------------------------------------------

    def _base_rooms(self, num_teams: int, phase_offset: float) -> list:
        """Return DungeonRooms room specs centred on each team's spawn area."""
        gs = self.grid_size
        cx, cy = (gs - 1) / 2.0, (gs - 1) / 2.0
        radius = gs * 0.38  # matches spawn_r in _spread_spawns
        rooms = []
        for t in range(max(num_teams, 1)):
            theta = phase_offset + 2.0 * math.pi * t / max(num_teams, 1)
            gx = int(round(cx + radius * math.cos(theta)))
            gy = int(round(cy + radius * math.sin(theta)))
            # 2×2 game-cell clearing → odd maze-space corners (clamped to grid)
            tl = (2 * max(0, gx - 1) + 1, 2 * max(0, gy - 1) + 1)
            br = (2 * min(gs - 2, gx + 1) + 1, 2 * min(gs - 2, gy + 1) + 1)
            rooms.append([tl, br])
        return rooms

    def _new_maze_generator(
        self, rng: np.random.Generator, num_teams: int = 1, phase_offset: float = 0.0
    ):
        return DungeonRooms(
            self.grid_size,
            self.grid_size,
            rooms=self._base_rooms(num_teams, phase_offset),
        )

    def _generate_maze_with_timeout(
        self,
        rng: np.random.Generator,
        timeout: int = 2,
        retries: int = 5,
        num_teams: int = 1,
        phase_offset: float = 0.0,
    ) -> None:
        for attempt in range(retries):
            try:
                self._maze.generator = self._new_maze_generator(
                    rng, num_teams=num_teams, phase_offset=phase_offset
                )
                self._maze.generate()
                return
            except Exception as exc:
                logger.warning("Maze gen attempt %d failed: %s", attempt, exc)
        raise RuntimeError(f"Failed to generate maze after {retries} attempts")

    def _build_wall_set(
        self, wall_grid: np.ndarray
    ) -> set[tuple[tuple[int, int], tuple[int, int]]]:
        """Build the edge-pair set from the wall grid (wall presence in bits 0–3)."""
        gs = self.grid_size
        walls: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        for x, y in np.ndindex((gs, gs)):
            tile = wall_grid[x, y]
            for direction in Direction:
                if not get_bit(tile, direction.value):
                    continue
                mv = direction.movement
                nx, ny = x + int(mv[0]), y + int(mv[1])
                if not (0 <= nx < gs and 0 <= ny < gs):
                    continue  # skip border edges that point outside the grid
                edge = tuple(sorted(((x, y), (nx, ny))))
                walls.add(edge)
        return walls

    # ── rotational symmetry ────────────────────────────────────────────────

    def _rotate_edge(
        self, edge: tuple, theta: float
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """Rotate a wall edge by *theta* radians about the grid centre.

        Returns the canonically sorted edge tuple, or ``None`` if either
        rotated endpoint falls outside the grid or if rounding collapses the
        two endpoints to the same cell / into a diagonal (non-adjacent) pair.
        """
        gs = self.grid_size
        cx = (gs - 1) / 2.0
        cy = (gs - 1) / 2.0
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        new_cells: list[tuple[int, int]] = []
        for ex, ey in edge:
            rx, ry = ex - cx, ey - cy
            nxi = int(round(cx + rx * cos_t - ry * sin_t))
            nyi = int(round(cy + rx * sin_t + ry * cos_t))
            if not (0 <= nxi < gs and 0 <= nyi < gs):
                return None
            new_cells.append((nxi, nyi))

        # Discard non-adjacent results (rounding artifact for non-4-fold N).
        (ax, ay), (bx, by) = new_cells
        if abs(ax - bx) + abs(ay - by) != 1:
            return None

        return tuple(sorted(new_cells))

    def _rotate_cell(self, x: int, y: int, theta: float) -> tuple[int, int] | None:
        """Rotate cell (x, y) by *theta* radians about the grid centre.

        Returns the rounded integer coordinate, or ``None`` if the result
        falls outside the grid.
        """
        gs = self.grid_size
        cx = (gs - 1) / 2.0
        cy = (gs - 1) / 2.0
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        rx, ry = x - cx, y - cy
        nxi = int(round(cx + rx * cos_t - ry * sin_t))
        nyi = int(round(cy + rx * sin_t + ry * cos_t))
        if not (0 <= nxi < gs and 0 <= nyi < gs):
            return None
        return (nxi, nyi)

    def _mark_destructible_walls(
        self,
        wall_grid: np.ndarray,
        walls: set,
        rng: np.random.Generator,
        num_teams: int,
    ) -> None:
        """Mark ``wall_destructible_ratio`` fraction of edges as destructible.

        Each edge in *walls* receives an independent Bernoulli roll.
        Destructibility is encoded in bits 4–7 of ``wall_grid`` (same
        direction order as the presence bits in 0–3).  Mutates in-place.
        """
        ratio = float(np.clip(self.wall_destructible_ratio, 0.0, 1.0))
        if ratio <= 0:
            return

        for edge in walls:
            if rng.random() >= ratio:
                continue
            (ax, ay), (bx, by) = edge
            dx, dy = bx - ax, by - ay
            d = 0 if dx == 1 else 1 if dy == 1 else 2 if dx == -1 else 3
            wall_grid[ax, ay] = np.uint8(int(wall_grid[ax, ay]) | (1 << (d + 4)))
            wall_grid[bx, by] = np.uint8(
                int(wall_grid[bx, by]) | (1 << ((d + 2) % 4 + 4))
            )

    def _ensure_connected(
        self, wall_grid: np.ndarray, walls: set
    ) -> tuple[np.ndarray, set]:
        """Open walls until all cells form a single connected region.

        Each iteration BFS-floods from cell (0, 0).  If any cell is
        unreachable, the first frontier wall between the visited and
        unvisited regions is removed and the loop restarts.  Worst-case
        O(G⁴) but in practice only a handful of walls need opening.
        """
        gs = self.grid_size

        while True:
            visited = np.zeros((gs, gs), dtype=bool)
            stack = [(0, 0)]
            visited[0, 0] = True
            while stack:
                x, y = stack.pop()
                for direction in Direction:
                    mv = direction.movement
                    nx, ny = x + int(mv[0]), y + int(mv[1])
                    if 0 <= nx < gs and 0 <= ny < gs and not visited[nx, ny]:
                        if tuple(sorted(((x, y), (nx, ny)))) not in walls:
                            visited[nx, ny] = True
                            stack.append((nx, ny))

            if visited.all():
                return wall_grid, walls

            # Find the first frontier wall.
            frontier_edge = None
            outer: bool = False
            for x in range(gs):
                for y in range(gs):
                    if not visited[x, y]:
                        continue
                    for direction in Direction:
                        mv = direction.movement
                        nx, ny = x + int(mv[0]), y + int(mv[1])
                        if 0 <= nx < gs and 0 <= ny < gs and not visited[nx, ny]:
                            e = tuple(sorted(((x, y), (nx, ny))))
                            if e in walls:
                                frontier_edge = e
                                outer = True
                                break
                    if outer:
                        break
                if outer:
                    break

            if frontier_edge is None:
                return wall_grid, walls  # cannot repair (shouldn't happen)

            walls.discard(frontier_edge)
            (ax, ay), (bx, by) = frontier_edge
            dx, dy = bx - ax, by - ay
            d = 0 if dx == 1 else 1 if dy == 1 else 2 if dx == -1 else 3
            wall_grid[ax, ay] &= np.uint8(~(1 << d) & ~(1 << (d + 4)) & 0xFF)
            wall_grid[bx, by] &= np.uint8(
                ~(1 << (d + 2) % 4) & ~(1 << ((d + 2) % 4 + 4)) & 0xFF
            )

    # ── unified spawn placement ────────────────────────────────────────────

    def _spread_spawns(
        self,
        rng: np.random.Generator,
        num_teams: int,
        wall_grid: np.ndarray,
        phase_offset: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Place one base and one agent per team around the grid.

        Teams are distributed at equal angular intervals around the grid
        centre, all at the **same radius**.

        Returns
        -------
        base_locations : np.ndarray  ``(num_teams, 2)``
        agent_locations : np.ndarray ``(num_teams, 2)``
        """
        gs = self.grid_size

        # continuous centre of the grid
        cx = (gs - 1) / 2.0
        cy = (gs - 1) / 2.0

        # spawn radius — 38% of the grid keeps bases well inside
        # the border (~1-tile margin) while being far from centre.
        spawn_r = 0.38 * gs

        # global set of occupied tiles (shared across all teams)
        occupied: set[tuple[int, int]] = set()

        base_locs: list[np.ndarray] = []
        agent_locs: list[np.ndarray] = []

        for t in range(num_teams):
            # equal angular spacing with random phase so the orientation
            # varies each episode
            theta = phase_offset + 2.0 * math.pi * t / num_teams
            ideal_x = cx + spawn_r * math.cos(theta)
            ideal_y = cy + spawn_r * math.sin(theta)

            # snap to nearest integer tile, clamped inside the grid
            anchor_x = int(np.clip(round(ideal_x), 0, gs - 1))
            anchor_y = int(np.clip(round(ideal_y), 0, gs - 1))

            # ── place base ─────────────────────────────────────────────
            base_pos = self._find_open_tile(
                rng,
                wall_grid,
                anchor_x,
                anchor_y,
                occupied,
            )
            if base_pos is None:
                raise RuntimeError(
                    f"Could not place base for team {t} near ({anchor_x}, {anchor_y})."
                )
            occupied.add(base_pos)
            base_locs.append(np.array(base_pos))

            # ── place agent near the base ──────────────────────────────
            agent_pos = self._find_reachable_tile(
                rng,
                wall_grid,
                (int(base_pos[0]), int(base_pos[1])),
                occupied,
            )
            if agent_pos is None:
                raise RuntimeError(
                    f"Could not place agent for team {t} near base "
                    f"at {base_pos} — grid may be too crowded."
                )
            occupied.add(agent_pos)
            agent_locs.append(np.array(agent_pos))

        return np.array(base_locs), np.array(agent_locs)

    @staticmethod
    def _find_open_tile(
        rng: np.random.Generator,
        wall_grid: np.ndarray,
        cx: int,
        cy: int,
        occupied: set[tuple[int, int]],
        max_radius: int | None = None,
    ) -> tuple[int, int] | None:
        """Spiral outward from ``(cx, cy)`` to find the nearest open tile.

        An *open* tile is one that is not fully enclosed (i.e. does NOT have
        walls on all four edges) and is not in ``occupied``.  Searches
        Chebyshev-distance shells at radius 0, 1, 2, … up to ``max_radius``
        (defaults to ``grid_size``).

        Within each shell the candidates are shuffled (via ``rng``) so that
        ties don't consistently resolve in the same direction.

        Returns ``None`` if no open tile is found within the search radius.
        """
        gs = wall_grid.shape[0]
        limit = max_radius if max_radius is not None else gs

        for r in range(limit):
            if r == 0:
                candidates = [(cx, cy)]
            else:
                candidates = []
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        if max(abs(dx), abs(dy)) != r:
                            continue  # Chebyshev-distance shell
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < gs and 0 <= ny < gs:
                            candidates.append((nx, ny))

            # shuffle for fairness
            rng.shuffle(candidates)

            for pos in candidates:
                # A tile is blocked only when ALL four edges are walled.
                # After the >>4 shift, wall bits live in the lower nibble.
                _ALL = 0x0F  # bits 0-3: RIGHT | BOTTOM | LEFT | TOP
                if (wall_grid[pos[0], pos[1]] & _ALL) != _ALL and pos not in occupied:
                    return pos

        return None

    @staticmethod
    def _find_reachable_tile(
        rng: np.random.Generator,
        wall_grid: np.ndarray,
        start: tuple[int, int],
        occupied: set[tuple[int, int]],
    ) -> tuple[int, int] | None:
        """BFS from *start* traversing only wall-free edges.

        Returns the nearest unoccupied reachable tile, shuffled within each
        BFS distance level for fairness.  Guarantees no structural wall lies
        between *start* and the returned tile.
        """
        gs = wall_grid.shape[0]
        if start not in occupied:
            return start

        visited: set[tuple[int, int]] = {start}
        frontier: list[tuple[int, int]] = [start]

        while frontier:
            nxt: list[tuple[int, int]] = []
            for x, y in frontier:
                tile_val = int(wall_grid[x, y])
                for d in Direction:
                    if get_bit(tile_val, d.value):
                        continue
                    mv = d.movement
                    nx, ny = x + int(mv[0]), y + int(mv[1])
                    if not (0 <= nx < gs and 0 <= ny < gs):
                        continue
                    if get_bit(int(wall_grid[nx, ny]), (d.value + 2) % 4):
                        continue
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        nxt.append((nx, ny))
            rng.shuffle(nxt)
            for pos in nxt:
                if pos not in occupied:
                    return pos
            frontier = nxt

        return None

    def _ensure_base_destructible_walls(
        self,
        wall_grid: np.ndarray,
        base_locations: np.ndarray,
        min_count: int = 7,
        radius: int = 1,
    ) -> None:
        """Guarantee each base has at least *min_count* destructible wall edges nearby.

        Collects all wall edges whose midpoint falls within Chebyshev *radius*
        of the base tile.  If fewer than *min_count* are already flagged
        destructible, the closest non-destructible candidates are promoted
        (bits 4–7 set) until the quota is met.  Only mutates ``wall_grid``.
        """
        gs = self.grid_size
        for base_pos in base_locations:
            bx, by = int(base_pos[0]), int(base_pos[1])

            # Collect unique wall edges in the neighbourhood.
            seen: set = set()
            nearby: list[tuple] = []  # (midpoint_chebyshev_dist, edge_canonical)
            for x in range(max(0, bx - radius), min(gs, bx + radius + 1)):
                for y in range(max(0, by - radius), min(gs, by + radius + 1)):
                    tile = int(wall_grid[x, y])
                    for d in Direction:
                        if not get_bit(tile, d.value):
                            continue
                        mv = d.movement
                        nx, ny = x + int(mv[0]), y + int(mv[1])
                        if not (0 <= nx < gs and 0 <= ny < gs):
                            continue
                        edge = tuple(sorted(((x, y), (nx, ny))))
                        if edge in seen:
                            continue
                        seen.add(edge)
                        (ax, ay), (ex, ey) = edge
                        mid_dist = max(abs((ax + ex) / 2 - bx), abs((ay + ey) / 2 - by))
                        nearby.append((mid_dist, edge))

            # Partition into already-destructible and candidates.
            destr_count = 0
            candidates: list[tuple] = []
            for dist, edge in nearby:
                (ax, ay), (ex, ey) = edge
                dx, dy = ex - ax, ey - ay
                d = 0 if dx == 1 else 1 if dy == 1 else 2 if dx == -1 else 3
                if get_bit(int(wall_grid[ax, ay]), d + 4):
                    destr_count += 1
                else:
                    candidates.append((dist, edge))

            need = min_count - destr_count
            if need <= 0:
                continue

            candidates.sort()
            for _, edge in candidates[:need]:
                (ax, ay), (ex, ey) = edge
                dx, dy = ex - ax, ey - ay
                d = 0 if dx == 1 else 1 if dy == 1 else 2 if dx == -1 else 3
                wall_grid[ax, ay] = np.uint8(int(wall_grid[ax, ay]) | (1 << (d + 4)))
                wall_grid[ex, ey] = np.uint8(
                    int(wall_grid[ex, ey]) | (1 << ((d + 2) % 4 + 4))
                )

    def _clear_base_area(
        self,
        wall_grid: np.ndarray,
        walls: set,
        base_locations: np.ndarray,
        radius: int = 1,
    ) -> None:
        """Remove all structural wall edges incident to cells within *radius* of each base.

        Clears both internal walls (both endpoints inside the box) and perimeter
        walls (one endpoint inside, one outside), so every side of the clearance
        zone has at least one open entry point from the surrounding maze.

        Mutates *wall_grid* and *walls* in-place (caller must pass copies if the
        originals are cached).
        """
        gs = self.grid_size
        for base_pos in base_locations:
            bx, by = int(base_pos[0]), int(base_pos[1])
            x0 = max(0, bx - radius)
            x1 = min(gs - 1, bx + radius)
            y0 = max(0, by - radius)
            y1 = min(gs - 1, by + radius)
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    for direction in Direction:
                        mv = direction.movement
                        nx, ny = x + int(mv[0]), y + int(mv[1])
                        if not (0 <= nx < gs and 0 <= ny < gs):
                            continue
                        bit = direction.value
                        if not get_bit(int(wall_grid[x, y]), bit):
                            continue
                        clear = ~(1 << bit) & ~(1 << (bit + 4)) & 0xFF
                        wall_grid[x, y] = np.uint8(int(wall_grid[x, y]) & clear)
                        opp_bit = (bit + 2) % 4
                        opp_clear = ~(1 << opp_bit) & ~(1 << (opp_bit + 4)) & 0xFF
                        wall_grid[nx, ny] = np.uint8(int(wall_grid[nx, ny]) & opp_clear)
                        walls.discard(tuple(sorted(((x, y), (nx, ny)))))

    def _deconflict_spawn_tiles(
        self,
        static_entities: list[StaticEntitySpec],
        starting_locations: np.ndarray,
        base_locations: np.ndarray,
    ) -> list[StaticEntitySpec]:
        """Remove static entities that share a tile with any spawn point."""
        excluded: set[tuple[int, int]] = set()
        for loc in starting_locations:
            excluded.add((int(loc[0]), int(loc[1])))
        for loc in base_locations:
            excluded.add((int(loc[0]), int(loc[1])))

        return [
            spec
            for spec in static_entities
            if (int(spec.position[0]), int(spec.position[1])) not in excluded
        ]
