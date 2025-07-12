import numpy as np

from highway_env.envs.common.action import Action, DiscreteMetaAction
from highway_env.road.road import LaneIndex
from highway_env.vehicle.controller import MDPVehicle

from norm_supervisor.consts import ACTION_STRINGS
from norm_supervisor.norms.abstract import AbstractNorm
from norm_supervisor.norms.prediction import get_next_speed, get_next_lane_index
import norm_supervisor.metrics as metrics

class SpeedingNorm(AbstractNorm):
    """Norm constraint for enforcing the speed limit."""
    def __init__(self, weight: int, speed_limit: float):
        """Initialize the speed limit norm with a weight and a speed limit."""
        super().__init__(weight, [ACTION_STRINGS["IDLE"], ACTION_STRINGS["FASTER"]])
        self.speed_limit = speed_limit

    def evaluate_criterion(self, vehicle: MDPVehicle, action: Action) -> float:
        """Return the next speed of the ego vehicle."""
        return get_next_speed(vehicle, action)
    
    def is_violating_action(self, vehicle: MDPVehicle, action: Action) -> bool:
        """Check if the action produces or worsens a speed limit violation."""
        if action not in self.violating_actions:
            return False
        return self.evaluate_criterion(vehicle, action) > self.speed_limit
    
    def __str__(self):
        return "Speeding"

class TailgatingNorm(AbstractNorm):
    """Norm constraint for enforcing a safe following distance."""
    def __init__(self, weight: int, action_type: DiscreteMetaAction, simulation_frequency: float):
        """Initialize the tailgating norm with a weight and environment parameters."""
        super().__init__(
            weight,
            [
                ACTION_STRINGS["IDLE"],
                ACTION_STRINGS["FASTER"],
                ACTION_STRINGS["SLOWER"],  # experimental: slower action can still be tailgating vs lane change
            ]
        )
        self.action_type = action_type
        self.simulation_frequency = simulation_frequency

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
        
        safe_distance = metrics.calculate_safe_distance(
            get_next_speed(vehicle, action),
            self.action_type,
            self.simulation_frequency
        )
        return self.evaluate_criterion(vehicle, lane_index, check_rear) < safe_distance
        
    def __str__(self):
        return "Tailgating"

class BrakingNorm(AbstractNorm):
    """Norm constraint for avoiding sudden braking."""
    def __init__(self, weight: int, min_ttc: float = 1.5):
        """Initialize the braking norm with a weight and a minimum TTC."""
        super().__init__(weight, [ACTION_STRINGS["FASTER"], ACTION_STRINGS["IDLE"]])
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
        return self.evaluate_criterion(vehicle, action, lane_index, check_rear) < self.min_ttc

    def __str__(self):
        return "Braking"

class LaneChangeTailgatingNorm(TailgatingNorm):
    """Norm constraint for enforcing safe distances for lane changes."""
    def __init__(self, weight: int, action_type: DiscreteMetaAction, simulation_frequency: float):
        """Initialize the lane change tailgating norm with a weight and environment parameters."""
        super().__init__(weight, action_type, simulation_frequency)
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

        safe_distance = metrics.calculate_safe_distance(
            get_next_speed(vehicle, action),
            self.action_type,
            self.simulation_frequency
        )
        return (super().evaluate_criterion(vehicle, next_lane_index, False) < safe_distance
                or super().evaluate_criterion(vehicle, next_lane_index, True) < safe_distance)

    def __str__(self):
        return "LaneChangeTailgating"

class LaneChangeBrakingNorm(BrakingNorm):
    """Norm constraint for avoiding sudden braking due to lane changes."""
    def __init__(self, weight: int, min_ttc: float = 1.5):
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

        return (super().evaluate_criterion(vehicle, action, next_lane_index, False) < self.min_ttc
                or super().evaluate_criterion(vehicle, action, next_lane_index, True) < self.min_ttc)

    def __str__(self):
        return "LaneChangeBraking"
