import enum


@enum.unique
class EventType(enum.Enum):
    """Define the different types of events in the game."""

    HAZARD = enum.auto()
    DEATH = enum.auto()
    TARGET = enum.auto()
    WALL = enum.auto()
