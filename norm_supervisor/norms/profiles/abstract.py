from abc import ABC

from norm_supervisor.consts import VEHICLE_LENGTH

from norm_supervisor.norms.abstract import AbstractNorm, AbstractConstraint
from norm_supervisor.norms.constraints import (
    CollisionConstraint,
    LaneChangeCollisionConstraint,
    SafetyEnvelopeConstraint,
    LaneChangeSafetyEnvelopeConstraint,
)
from norm_supervisor.norms.norms import (
    SpeedNorm,
    BrakingNorm,
    LaneChangeBrakingNorm,
    TailgatingNorm,
    LaneChangeTailgatingNorm,
    LanePreference,
    LaneKeepingNorm
)

class AbstractNormProfile(ABC):
    """Abstract class for norm profiles."""
    COLLISION_THRESHOLD = 1.0                # Minimum TTC (s)
    SAFETY_ENVELOPE     = VEHICLE_LENGTH     # Minimum distance (m)
    
    def __init__(
        self,
        target_speed_range: tuple[float, float],
        target_following_distance: float,
        braking_threshold: float,
        lane_preference: LanePreference
    ):
        """Initialize the norm profile with hard constraints and norms.
        
        :param target_speed_range: target speed range in m/s
        :param target_following_distance: target following distance in meters
        :param braking_threshold: minimum TTC (s) for braking norm
        :param lane_preference: preferred lane (LEFT, RIGHT, NONE)
        """

        self.collision_threshold: float              = type(self).COLLISION_THRESHOLD
        self.safety_envelope: float                  = type(self).SAFETY_ENVELOPE
        self.target_speed_range: tuple[float, float] = target_speed_range
        self.target_following_distance: float        = target_following_distance
        self.braking_threshold: float                = braking_threshold
        self.lane_preference: LanePreference         = lane_preference

        # Initialize constraints with common thresholds
        self.constraints: list[AbstractConstraint] = [
            CollisionConstraint(min_ttc=self.collision_threshold),
            LaneChangeCollisionConstraint(min_ttc=self.collision_threshold),
            SafetyEnvelopeConstraint(safe_distance=self.safety_envelope),
            LaneChangeSafetyEnvelopeConstraint(safe_distance=self.safety_envelope),
        ]

        # Initialize norms with the specified thresholds using a standard weighting scheme
        self.norms: list[AbstractNorm] = [
            SpeedNorm(
                target_speed_range=self.target_speed_range,
                weight=1
            ),
            TailgatingNorm(
                safe_distance=self.target_following_distance,
                weight=1
            ),
            BrakingNorm(
                min_ttc=self.braking_threshold,
                weight=1
            ),
            LaneChangeTailgatingNorm(
                safe_distance=self.target_following_distance,
                weight=1
            ),
            LaneChangeBrakingNorm(
                min_ttc=self.braking_threshold,
                weight=1
            ),
            LaneKeepingNorm(
                lane_preference=self.lane_preference,
                weight=1
            )
        ]
