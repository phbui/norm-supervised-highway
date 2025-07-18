from enum import Enum

from highway_env.envs.common.action import Action
from highway_env.road.road import LaneIndex
from highway_env.vehicle.controller import MDPVehicle

from norm_supervisor.consts import ACTION_STRINGS
from norm_supervisor.norms.abstract import AbstractNorm
from norm_supervisor.norms.constraints import (
    SafetyEnvelopeConstraint,
    LaneChangeSafetyEnvelopeConstraint
)
from norm_supervisor.norms.prediction import get_next_speed, get_next_lane_index
import norm_supervisor.metrics as metrics

class SpeedNorm(AbstractNorm):
    """Norm for enforcing a target speed range."""
    def __init__(self, target_speed_range: tuple[float, float], weight: int = 1):
        """Initialize the speed norm with a weight and a target speed range."""
        if target_speed_range[0] > target_speed_range[1]:
            raise ValueError("Target speed range must be a tuple of (min_speed, max_speed) where "
                             "min_speed <= max_speed.")

        super().__init__(
            violating_actions=[
                ACTION_STRINGS["FASTER"],
                ACTION_STRINGS["SLOWER"],
                ACTION_STRINGS["IDLE"],
                ACTION_STRINGS["LANE_LEFT"],
                ACTION_STRINGS["LANE_RIGHT"]
            ],
            weight=weight
        )
        self.min_speed = target_speed_range[0]
        self.max_speed = target_speed_range[1]

    @staticmethod
    def evaluate_criterion(vehicle: MDPVehicle, action: Action) -> float:
        """Return the next speed of the ego vehicle."""
        return get_next_speed(vehicle, action)
    
    def is_violating_action(self, vehicle: MDPVehicle, action: Action) -> bool:
        """Check if the action produces a speed outside of the target speed range."""
        if action not in self.violating_actions:
            return False

        speed = self.evaluate_criterion(vehicle, action)
        return speed < self.min_speed or speed > self.max_speed
    
    def __str__(self):
        return "SpeedNorm"

class BrakingNorm(AbstractNorm):
    """Norm constraint for avoiding sudden braking."""
    def __init__(self, min_ttc: float, weight: int = 1):
        """Initialize the braking norm with a weight and a minimum TTC."""
        super().__init__(
            violating_actions=[
                ACTION_STRINGS["FASTER"],
                # SLOWER can still be norm-violating if a safe lane change is possible
                ACTION_STRINGS["SLOWER"],
                ACTION_STRINGS["IDLE"],
                # If the lane change is disallowed, the action is effectively IDLE
                ACTION_STRINGS["LANE_LEFT"],
                ACTION_STRINGS["LANE_RIGHT"]
            ],
            weight=weight
        )
        self.min_ttc = min_ttc

    @staticmethod
    def evaluate_criterion(
        vehicle: MDPVehicle,
        action: Action,
        lane_index: LaneIndex = None,
        check_rear: bool = False
    ) -> float:
        """Return the TTC between the ego vehicle and the leading or following vehicle."""
        next_speed = get_next_speed(vehicle, action)
        ttc_front, ttc_rear = metrics.calculate_neighbour_ttcs(vehicle, lane_index, next_speed)
        return ttc_rear if check_rear else ttc_front

    def is_violating_action(
        self,
        vehicle: MDPVehicle,
        action: Action,
        lane_index: LaneIndex = None,
        check_rear: bool = False
    ) -> bool:
        """Check if the action produces or worsens a braking violation."""
        if action not in self.violating_actions:
            return False
        
        # If the action results in a lane change, this norm is not violated
        if get_next_lane_index(vehicle, action) != vehicle.target_lane_index:
            return False
        
        ttc = self.evaluate_criterion(vehicle, action, lane_index, check_rear)
        return ttc < self.min_ttc

    def __str__(self):
        return "BrakingNorm"

class LaneChangeBrakingNorm(BrakingNorm):
    """Norm constraint for avoiding sudden braking due to lane changes."""
    def __init__(self, min_ttc: float, weight: int = 1):
        """Initialize the braking constraint with a weight and a minimum TTC."""
        super().__init__(min_ttc=min_ttc, weight=weight)
        self.violating_actions = [
            ACTION_STRINGS["LANE_LEFT"],
            ACTION_STRINGS["LANE_RIGHT"]
        ]
    
    def is_violating_action(self, vehicle: MDPVehicle, action: Action) -> bool:
        """Check if the action produces or worsens a braking violation for lane changes."""
        if action not in self.violating_actions:
            return False
        
        # Return False if the action does not result in a lane change
        next_lane_index = get_next_lane_index(vehicle, action)
        if next_lane_index == vehicle.target_lane_index:
            return False

        ttc_front = super().evaluate_criterion(vehicle, action, next_lane_index, False)
        ttc_rear = super().evaluate_criterion(vehicle, action, next_lane_index, True)
        return ttc_front < self.min_ttc or ttc_rear < self.min_ttc

    def __str__(self):
        return "LaneChangeBrakingNorm"

class TailgatingNorm(SafetyEnvelopeConstraint, AbstractNorm):
    """Norm constraint for enforcing a safe following distance."""
    def __init__(self, safe_distance: float, weight: int = 1):
        """Initialize the tailgating norm with a weight and a safe distance."""
        SafetyEnvelopeConstraint.__init__(self, safe_distance=safe_distance)
        AbstractNorm.__init__(self, violating_actions=self.violating_actions, weight=weight)
        
    def __str__(self):
        return "TailgatingNorm"
    
class LaneChangeTailgatingNorm(LaneChangeSafetyEnvelopeConstraint, AbstractNorm):
    """Norm constraint for enforcing a safe following distance during lane changes."""
    def __init__(self, safe_distance: float, weight: int = 1):
        """Initialize the lane change tailgating norm with a weight and a safe distance."""
        SafetyEnvelopeConstraint.__init__(self, safe_distance=safe_distance)
        AbstractNorm.__init__(self, violating_actions=self.violating_actions, weight=weight)
        
    def __str__(self):
        return "LaneChangeTailgatingNorm"

class LanePreference(Enum):
    """Enum for lane preferences."""
    LEFT  = 'left'
    RIGHT = 'right'
    NONE  = 'none'

class LaneKeepingNorm(AbstractNorm):
    """Norm constraint for enforcing lane keeping."""
    def __init__(self, lane_preference: LanePreference, weight: int = 1):
        """Initialize the lane keeping norm with a weight."""
        super().__init__(
            violating_actions=[
                ACTION_STRINGS["FASTER"],
                ACTION_STRINGS["SLOWER"],
                ACTION_STRINGS["IDLE"],
                ACTION_STRINGS["LANE_LEFT"],
                ACTION_STRINGS["LANE_RIGHT"]
            ],
            weight=weight
        )
        self.lane_preference = lane_preference

    @staticmethod
    def evaluate_criterion(vehicle: MDPVehicle, action: Action) -> LaneIndex:
        """Return the next lane index after applying the action."""
        return get_next_lane_index(vehicle, action)

    def is_violating_action(self, vehicle: MDPVehicle, action: Action) -> bool:
        """Check if the action results in a lane change outside of the preferred lanes."""
        if action not in self.violating_actions or self.lane_preference == LanePreference.NONE:
            return False
        
        _from, _to, next_lane_id = self.evaluate_criterion(vehicle, action)
        all_lanes = vehicle.road.network.graph[_from][_to]
        middle_lane = len(all_lanes) // 2

        # If there is an odd number of lanes, the middle lane is always compliant
        if len(all_lanes) % 2 == 1 and next_lane_id == middle_lane:
            return False
        
        if self.lane_preference == LanePreference.LEFT:
            return next_lane_id >= middle_lane
        elif self.lane_preference == LanePreference.RIGHT:
            return next_lane_id < middle_lane
        else:
            raise ValueError(f"Unknown lane preference: {self.lane_preference}")

    def __str__(self):
        return "LaneKeepingNorm"
