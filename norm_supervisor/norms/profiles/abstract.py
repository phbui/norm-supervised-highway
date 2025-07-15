from abc import ABC

from norm_supervisor.norms.abstract import AbstractNorm, AbstractConstraint
from norm_supervisor.norms.constraints import CollisionConstraint, LaneChangeCollisionConstraint
from norm_supervisor.norms.norms import LanePreference

class AbstractNormProfile(ABC):
    """Abstract class for norm profiles."""
    # NOTE: The collision threshold must be greater than the policy period (1/policy_frequency).
    COLLISION_THRESHOLD = 1.0                # Minimum TTC (s)

    # Constants for the norm profile
    TARGET_SPEED_RANGE: tuple[float, float]  # Target speed range in m/s
    TARGET_FOLLOWING_DISTANCE: float         # Target following distance in meters
    BRAKING_THRESHOLD: float                 # Minimum TTC (s) for braking behavior
    LANE_PREFERENCE: LanePreference          # Preferred lane (LEFT, RIGHT, NONE)
    
    def __init__(self, norms: list[AbstractNorm]):
        """Initialize the norm profile with hard constraints and norms."""
        self.norms = norms
        self.constraints: list[AbstractConstraint] = [
            CollisionConstraint(self.collision_threshold),
            LaneChangeCollisionConstraint(self.collision_threshold),
        ]

    def __init_subclass__(cls):
        required_constants = [
            'TARGET_SPEED_RANGE',
            'TARGET_FOLLOWING_DISTANCE',
            'BRAKING_THRESHOLD',
            'LANE_PREFERENCE'
        ]
        for const in required_constants:
            if not hasattr(cls, const):
                raise NotImplementedError(f"{cls.__name__} is missing required constant: {const}")
    
    @property
    def collision_threshold(self) -> float:
        return type(self).COLLISION_THRESHOLD

    @property
    def target_speed_range(self) -> tuple[float, float]:
        return type(self).TARGET_SPEED_RANGE

    @property
    def target_following_distance(self) -> float:
        return type(self).TARGET_FOLLOWING_DISTANCE

    @property
    def braking_threshold(self) -> float:
        return type(self).BRAKING_THRESHOLD

    @property
    def lane_preference(self) -> LanePreference:
        return type(self).LANE_PREFERENCE
