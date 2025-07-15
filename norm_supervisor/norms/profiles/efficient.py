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
    TARGET_SPEED_RANGE        = (25, 30)            # Target speed range in m/s
    TARGET_FOLLOWING_DISTANCE = VEHICLE_LENGTH * 2  # Target following distance in m
    BRAKING_THRESHOLD         = 2                   # Minimum TTC (s)
    LANE_PREFERENCE           = LanePreference.LEFT # Preferred lane for efficient driving
    
    def __init__(self):
        """Initialize the cautious driving profile with constraints and norms."""
        super().__init__(
            norms=[
                SpeedNorm(weight=4, target_speed_range=self.target_speed_range),
                TailgatingNorm(weight=4,safe_distance=self.target_following_distance),
                BrakingNorm(weight=5, min_ttc=self.braking_threshold),
                LaneChangeTailgatingNorm(weight=4, safe_distance=self.target_following_distance),
                LaneChangeBrakingNorm(weight=5, min_ttc=self.braking_threshold),
                LaneKeepingNorm(weight=3, lane_preference=self.lane_preference)
            ]
        )
