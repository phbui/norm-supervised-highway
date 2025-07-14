from enum import Enum
import numpy as np

from highway_env.envs.common.action import Action
from highway_env.road.road import LaneIndex
from highway_env.vehicle.controller import MDPVehicle

from norm_supervisor.consts import ACTION_STRINGS
from norm_supervisor.norms.abstract import AbstractNorm
from norm_supervisor.norms.prediction import get_next_speed, get_next_lane_index
import norm_supervisor.metrics as metrics

class LanePreference(Enum):
    """Enum for lane preferences."""
    LEFT  = 'left'
    RIGHT = 'right'
    NONE  = 'none'

class SpeedNorm(AbstractNorm):
    """Norm for enforcing a target speed range."""
    def __init__(self, weight: int, target_speed_range: tuple[float, float]):
        """Initialize the speed norm with a weight and a target speed range."""
        if target_speed_range[0] > target_speed_range[1]:
            raise ValueError("Target speed range must be a tuple of (min_speed, max_speed) where "
                             "min_speed <= max_speed.")

        super().__init__(
            weight,
            [
                ACTION_STRINGS["IDLE"],
                ACTION_STRINGS["FASTER"],
                ACTION_STRINGS["SLOWER"]
            ]
        )
        self.min_speed = target_speed_range[0]
        self.max_speed = target_speed_range[1]

    def evaluate_criterion(self, vehicle: MDPVehicle, action: Action) -> float:
        """Return the next speed of the ego vehicle."""
        return get_next_speed(vehicle, action)
    
    def is_violating_action(self, vehicle: MDPVehicle, action: Action) -> bool:
        """Check if the action produces a speed outside of the target speed range."""
        if action not in self.violating_actions:
            return False
        speed = self.evaluate_criterion(vehicle, action)
        return speed < self.min_speed or speed > self.max_speed
    
    def __str__(self):
        return "Speeding"

class TailgatingNorm(AbstractNorm):
    """Norm constraint for enforcing a safe following distance."""
    def __init__(self, weight: int, safe_distance: float):
        """Initialize the tailgating norm with a weight and environment parameters."""
        super().__init__(
            weight,
            [
                ACTION_STRINGS["IDLE"],
                ACTION_STRINGS["FASTER"]
            ]
        )
        self.safe_distance = safe_distance

    def evaluate_criterion(
        self,
        vehicle: MDPVehicle,
        lane_index: LaneIndex = None,
        check_rear: bool = False
    ) -> float:
        """Return the distance to the vehicle ahead or behind."""
        v_front, v_rear = vehicle.road.neighbour_vehicles(vehicle, lane_index)
        v_to_check = v_rear if check_rear else v_front
        if v_to_check is not None:
            return v_to_check.position[0] - vehicle.position[0] - MDPVehicle.LENGTH
        return np.inf
    
    def is_violating_action(
            self,
            vehicle: MDPVehicle,
            action: Action,
            lane_index: LaneIndex = None,
            check_rear: bool = False
    ) -> bool:
        """Check if the action produces or worsens a tailgating violation."""
        if action not in self.violating_actions:
            return False
        
        distance = self.evaluate_criterion(vehicle, lane_index, check_rear)
        return distance < self.safe_distance
        
    def __str__(self):
        return "Tailgating"

class BrakingNorm(AbstractNorm):
    """Norm constraint for avoiding sudden braking."""
    def __init__(self, weight: int, min_ttc: float):
        """Initialize the braking norm with a weight and a minimum TTC."""
        super().__init__(
            weight,
            [
                ACTION_STRINGS["FASTER"],
                ACTION_STRINGS["IDLE"]
            ]
        )
        self.min_ttc = min_ttc

    def evaluate_criterion(
        self,
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
        
        ttc = self.evaluate_criterion(vehicle, action, lane_index, check_rear)
        return ttc < self.min_ttc

    def __str__(self):
        return "Braking"

class LaneChangeTailgatingNorm(TailgatingNorm):
    """Norm constraint for enforcing safe distances for lane changes."""
    def __init__(self, weight: int, safe_distance: float):
        """Initialize the lane change tailgating norm with a weight and environment parameters."""
        super().__init__(weight, safe_distance)
        self.violating_actions = [
            ACTION_STRINGS["LANE_LEFT"],
            ACTION_STRINGS["LANE_RIGHT"]
        ]

    def is_violating_action(self, vehicle: MDPVehicle, action: Action) -> bool:
        """Check if the action produces or worsens a tailgating violation for lane changes."""
        if action not in self.violating_actions:
            return False
        
        # Return False if the action does not result in a lane change
        next_lane_index = get_next_lane_index(vehicle, action)
        if next_lane_index == vehicle.target_lane_index:
            return False

        distance_front = super().evaluate_criterion(vehicle, next_lane_index, False)
        distance_rear = super().evaluate_criterion(vehicle, next_lane_index, True)
        return distance_front < self.safe_distance or distance_rear < self.safe_distance

    def __str__(self):
        return "LaneChangeTailgating"

class LaneChangeBrakingNorm(BrakingNorm):
    """Norm constraint for avoiding sudden braking due to lane changes."""
    def __init__(self, weight: int, min_ttc: float):
        """Initialize the braking constraint with a weight and a minimum TTC."""
        super().__init__(weight, min_ttc)
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
        return "LaneChangeBraking"

class LaneKeepingNorm(AbstractNorm):
    """Norm constraint for enforcing lane keeping."""
    def __init__(self, weight: int, lane_preference: LanePreference):
        """Initialize the lane keeping norm with a weight."""
        super().__init__(
            weight,
            [
                ACTION_STRINGS["LANE_LEFT"],
                ACTION_STRINGS["LANE_RIGHT"]
            ]
        )
        self.lane_preference = lane_preference

    def evaluate_criterion(self, vehicle: MDPVehicle, action: Action) -> LaneIndex:
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
        return "LaneKeeping"
