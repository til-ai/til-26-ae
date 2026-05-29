"""
entities/registry.py — Indexed container for O(1) entity lookups.

Imports: base.py, static.py, dynamic.py, protocols.py.
"""

from collections.abc import Iterable
from typing import Callable, Iterator, TypeVar, Generic, Protocol

from til_environment.entities.base import Entity, EntityStatus
from til_environment.entities.protocols import (
    Attacker,
    Defender,
    Vision,
)
from til_environment.entities.static import (
    Mission,
    Recon,
    Resource,
)
from til_environment.entities.dynamic import Agent, Base, Bomb


# Query builder for lazy, composable filtering
T = TypeVar("T", bound=Entity)


class Query(Generic[T]):
    """
    Extremely overengineered query class for composable entity filtering. 
    Very cool as an idea, but completely unnecessary.
    Actually no it do make the code read nicer so yippee
    """
    def __init__(self, registry: "EntityRegistry", ids: Iterable[str] | None = None):
        self._registry = registry
        self._ids: set[str] | None = set(ids) if ids is not None else None
        self._filters: list[Callable[[T], bool]] = []

    def type(self, cls: type) -> "Query[T]":
        ids = self._registry.type_index.get(cls, set())
        return Query(self._registry, ids if self._ids is None else self._ids & ids)

    def team(self, team: int | None) -> "Query[T]":
        ids = self._registry.team_index.get(team, set())
        return Query(self._registry, ids if self._ids is None else self._ids & ids)

    def status(self, status: EntityStatus) -> "Query[T]":
        ids = self._registry.status_index.get(status, set())
        return Query(self._registry, ids if self._ids is None else self._ids & ids)

    def at(self, x: int, y: int) -> "Query[T]":
        ids = self._registry.pos_index[x][y]
        return Query(self._registry, ids if self._ids is None else self._ids & ids)

    def filter(self, fn: Callable[[T], bool]) -> "Query[T]":
        self._filters.append(fn)
        return self

    def all(self) -> list[T]:
        ids = sorted(
            self._ids if self._ids is not None else set(self._registry._entities.keys())
        )
        entities = [
            self._registry._entities[eid]
            for eid in ids
            if eid in self._registry._entities
        ]
        for f in self._filters:
            entities = list(filter(f, entities))
        return entities

    def first(self) -> T | None:
        all_ = self.all()
        return all_[0] if all_ else None

    def __iter__(self) -> Iterator[T]:
        return iter(self.all())

    def __len__(self) -> int:
        return len(self.all())

    def ids(self) -> set[str]:
        return set(e.entity_id for e in self.all())

    def __repr__(self):
        return f"<Query n={len(self)} filters={len(self._filters)} ids={self._ids}>"


EntityRegistryQuery = Query


class EntityRegistry:
    """Indexed store of all entities on the board."""

    EXCLUDED_TYPES = {object, Generic, Protocol}

    def __init__(self, grid_size: int = 0) -> None:
        self.grid_size = grid_size
        self._entities: dict[str, Entity] = {}
        self.pos_index: list[list[set[str]]] = [
            [set() for _ in range(grid_size)] for _ in range(grid_size)
        ]
        self.status_index: dict[EntityStatus, set[str]] = {
            EntityStatus.ACTIVE: set(),
            EntityStatus.DESTROYED: set(),
        }
        self.team_index: dict[int | None, set[str]] = {}
        self.type_index: dict[type, set[str]] = {}

    def query(self) -> Query[Entity]:
        return Query(self)

    def _index_add(self, entity: Entity) -> None:
        eid = entity.entity_id
        self.pos_index[int(entity.position[0])][int(entity.position[1])].add(eid)
        self.status_index[entity.status].add(eid)
        self.team_index.setdefault(entity.team, set()).add(eid)
        for cls in type(entity).__mro__:
            if cls not in self.EXCLUDED_TYPES:
                self.type_index.setdefault(cls, set()).add(eid)

    def _index_remove(self, entity: Entity) -> None:
        eid = entity.entity_id
        self.pos_index[int(entity.position[0])][int(entity.position[1])].discard(eid)
        self.status_index.get(entity.status, set()).discard(eid)
        self.team_index.get(entity.team, set()).discard(eid)
        for cls in type(entity).__mro__:
            if cls not in self.EXCLUDED_TYPES:
                self.type_index.get(cls, set()).discard(eid)

    def _on_position_changed(self, entity: Entity, old_pos: tuple[int, int]) -> None:
        eid = entity.entity_id
        self.pos_index[old_pos[0]][old_pos[1]].discard(eid)
        new_x, new_y = int(entity.position[0]), int(entity.position[1])
        self.pos_index[new_x][new_y].add(eid)

    def _on_destroyed(self, entity: Entity) -> None:
        eid = entity.entity_id
        self.status_index[EntityStatus.ACTIVE].discard(eid)
        self.status_index[EntityStatus.DESTROYED].add(eid)

    # -- CRUD ---------------------------------------------------------------

    def add(self, entity: Entity) -> None:
        if entity.entity_id in self._entities:
            raise ValueError(f"Duplicate entity_id: {entity.entity_id}")
        self._entities[entity.entity_id] = entity
        entity._registry = self
        self._index_add(entity)

    def remove(self, entity_id: str) -> Entity | None:
        entity = self._entities.pop(entity_id, None)
        if entity is not None:
            self._index_remove(entity)
            entity._registry = None
        return entity

    def get(self, entity_id: str) -> Entity | None:
        return self._entities.get(entity_id)

    def __contains__(self, entity_id: str) -> bool:
        return entity_id in self._entities

    def __len__(self) -> int:
        return len(self._entities)

    def __iter__(self):
        return iter(self._entities.values())

    # -- positional lookup (O(1)) ------------------------------------------

    def at(self, x: int, y: int) -> list[Entity]:
        """Return all alive entities at grid position (x, y)."""
        return [
            e
            for eid in sorted(self.pos_index[x][y])
            if (e := self._entities.get(eid)) is not None and e.alive
        ]

    # -- filtered views -----------------------------------------------------

    def _materialize(self, ids: Iterable[str]) -> list[Entity]:
        """Resolve entity ids to entities in deterministic (sorted) order.

        Single chokepoint for set→list conversion: index sets have no stable
        iteration order, so every accessor that fans out from an id-set routes
        through here to guarantee reproducible ordering. Stale ids are dropped.
        """
        return [self._entities[eid] for eid in sorted(ids) if eid in self._entities]

    def _alive_of_type_ids(self, cls: type) -> list[str]:
        type_ids = self.type_index.get(cls, set())
        alive_ids = self.status_index[EntityStatus.ACTIVE]
        return sorted(eid for eid in (type_ids & alive_ids) if eid in self._entities)

    def _alive_of_type(self, cls: type) -> list[Entity]:
        type_ids = self.type_index.get(cls, set())
        alive_ids = self.status_index[EntityStatus.ACTIVE]
        return self._materialize(type_ids & alive_ids)

    def _alive_of_type_team(self, cls: type, team: int | None) -> list[Entity]:
        type_ids = self.type_index.get(cls, set())
        alive_ids = self.status_index[EntityStatus.ACTIVE]
        team_ids = self.team_index.get(team, set())
        return self._materialize(type_ids & alive_ids & team_ids)

    def _alive_of_types(self, *classes: type) -> list[Entity]:
        alive_ids = self.status_index[EntityStatus.ACTIVE]
        union: set[str] = set()
        for cls in classes:
            union |= self.type_index.get(cls, set())
        return self._materialize(union & alive_ids)

    def _alive_of_types_team(self, team: int | None, *classes: type) -> list[Entity]:
        alive_ids = self.status_index[EntityStatus.ACTIVE]
        team_ids = self.team_index.get(team, set())
        union: set[str] = set()
        for cls in classes:
            union |= self.type_index.get(cls, set())
        return self._materialize(union & alive_ids & team_ids)

    def by_type(self, cls: type) -> list[Entity]:
        return self._materialize(self.type_index.get(cls, set()))

    def by_team(self, team: int) -> list[Entity]:
        return self._materialize(self.team_index.get(team, set()))

    def by_type_and_team(self, cls: type, team: int) -> list[Entity]:
        type_ids = self.type_index.get(cls, set())
        team_ids = self.team_index.get(team, set())
        return self._materialize(type_ids & team_ids)

    # -- concrete-type accessors -------------------------------------------

    def agents(self, team: int | None = None) -> list[Agent]:
        if team is None:
            return self._alive_of_type(Agent)  # type: ignore[return-value]
        return self._alive_of_type_team(Agent, team)  # type: ignore[return-value]

    def bases(self, team: int | None = None) -> list[Base]:
        if team is None:
            return self._alive_of_type(Base)  # type: ignore[return-value]
        return self._alive_of_type_team(Base, team)  # type: ignore[return-value]

    def bombs(self, team: int | None = None) -> list[Bomb]:
        if team is None:
            return self._alive_of_type(Bomb)  # type: ignore[return-value]
        return self._alive_of_type_team(Bomb, team)  # type: ignore[return-value]

    def missions(self) -> list[Mission]:
        return self._alive_of_type(Mission)  # type: ignore[return-value]

    def recons(self) -> list[Recon]:
        return self._alive_of_type(Recon)  # type: ignore[return-value]

    def resources(self) -> list[Resource]:
        return self._alive_of_type(Resource)  # type: ignore[return-value]

    def alive(self) -> list[Entity]:
        return self._materialize(self.status_index[EntityStatus.ACTIVE])

    # -- protocol-keyed trait accessors ------------------------------------

    def attackers(self, team: int | None = None) -> list[Attacker]:
        if team is None:
            return self._alive_of_type(Attacker)  # type: ignore[return-value]
        return self._alive_of_type_team(Attacker, team)  # type: ignore[return-value]

    def defenders(self, team: int | None = None) -> list[Defender]:
        if team is None:
            return self._alive_of_type(Defender)  # type: ignore[return-value]
        return self._alive_of_type_team(Defender, team)  # type: ignore[return-value]

    def vision_providers(self, team: int | None = None) -> list[Vision]:
        if team is None:
            return self._alive_of_type(Vision)  # type: ignore[return-value]
        return self._alive_of_type_team(Vision, team)  # type: ignore[return-value]

    def clear(self) -> None:
        for entity in self._entities.values():
            entity._registry = None
        self._entities.clear()
        for col in self.pos_index:
            for cell in col:
                cell.clear()
        self.status_index = {EntityStatus.ACTIVE: set(), EntityStatus.DESTROYED: set()}
        self.team_index.clear()
        self.type_index.clear()
