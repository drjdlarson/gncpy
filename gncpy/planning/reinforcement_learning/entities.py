class Entity:
    """Elements in a game.

    Attributes
    ----------
    """

    __slots__ = ('_active', '_id', '_tag', '_components')

    def __init__(self, e_id, tag):
        """Initialize an object.

        This should not be called outside of the :class:`.EntityManager` class.

        Parameters
        ----------
        e_id : int
            Unique ID number for the entity.
        tag : string
            Type of entity.
        """
        self._active = True
        self._id = e_id
        self._tag = tag

        self._components = {}

    @property
    def active(self):
        """Flag indicating if the entity is alive."""
        return self._active

    @property
    def tag(self):
        """Read only tag of for the entity."""
        return self._tag

    @property
    def id(self):
        """Read only unique id of the entity."""
        return self._id

    def destroy(self):
        """Handles destruction of the entity."""
        self._active = False

    def has_component(self, comp):
        return comp.__name__ in self._components

    def get_component(self, comp):
        return self._components[comp.__name__]

    def add_component(self, comp, **kwargs):
        component = comp(**kwargs)
        self._components[comp.__name__] = component
        return component


class EntityManager:
    """Handles creation and deletion of entities."""

    __slots__ = ('_entities', '_entities_to_add', '_entity_map', '_total_entities')

    def __init__(self):
        self._entities = []
        self._entities_to_add = []
        self._entity_map = {}
        self._total_entities = 0

    def _remove_dead_entities(self, vec):
        e_to_rm = []
        for ii, e in enumerate(vec):
            if not e.active:
                e_to_rm.append(ii)

        for ii in e_to_rm[::-1]:
            del vec[ii]

    def update(self):
        """Updates the list of entities.

        Should be called once per timestep. Adds new entities to the list and
        and removes dead ones.

        Returns
        -------
        None.
        """
        for e in self._entities_to_add:
            self._entities.append(e)
            if e.tag not in self._entity_map:
                self._entity_map[e.tag] = []
            self._entity_map[e.tag].append(e)
        self._entities_to_add = []

        self._remove_dead_entities(self._entities)

        for tag, ev in self._entity_map.items():
            self._remove_dead_entities(ev)

    def add_entity(self, tag):
        """Creates a new entity.

        The entity is queued to be added. It is not part of the entity list
        until after the update function has been called.

        Parameters
        ----------
        tag : string
            Tag to identify the type of entity.

        Returns
        -------
        e : :class:`.Entity`
            Reference to the created entity.
        """
        self._total_entities += 1
        e = Entity(self._total_entities, tag)
        self._entities_to_add.append(e)

        return e

    def get_entities(self, tag=None):
        """Return a list of references to entities.

        Can also get all entities with a given tag. Note that changing entities
        returned by this function modifies the entities managed by this class.

        Parameters
        ----------
        tag : string, optional
            If provided only return entities with this tag. The default is None.

        Returns
        -------
        list
            Each element is an :class:`.Entity`.
        """
        if tag is None:
            return self._entities
        else:
            if tag in self._entity_map:
                return self._entity_map[tag]
            else:
                return []
