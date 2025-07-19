from norm_supervisor.norms.norms import (
    SpeedNorm,
    TailgatingNorm,
    BrakingNorm,
    LaneChangeTailgatingNorm,
    LaneChangeBrakingNorm, 
    LaneKeepingNorm
)
from norm_supervisor.consts import VEHICLE_LENGTH
from norm_supervisor.norms.norms import LanePreference
from norm_supervisor.norms.profiles.abstract import AbstractNormProfile

class EfficientDrivingProfile(AbstractNormProfile):
    """Profile for efficient driving norms."""
    TARGET_SPEED_RANGE        = (0, 30)             # Target speed range in m/s
    TARGET_FOLLOWING_DISTANCE = VEHICLE_LENGTH * 2  # Target following distance in m
    BRAKING_THRESHOLD         = 2                   # Minimum TTC (s)
    LANE_PREFERENCE           = LanePreference.LEFT # Preferred lane for efficient driving
    
    def __init__(self):
        """Initialize the efficient driving profile with norms."""
        super().__init__(
            target_speed_range=type(self).TARGET_SPEED_RANGE,
            target_following_distance=type(self).TARGET_FOLLOWING_DISTANCE,
            braking_threshold=type(self).BRAKING_THRESHOLD,
            lane_preference=type(self).LANE_PREFERENCE
        )
