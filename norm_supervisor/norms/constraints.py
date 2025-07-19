import numpy as np
from typing import Optional

from highway_env.envs.common.action import Action
from highway_env.road.road import LaneIndex
from highway_env.vehicle.controller import MDPVehicle

from norm_supervisor.consts import ACTION_STRINGS
from norm_supervisor.norms.abstract import AbstractConstraint
from norm_supervisor.norms.prediction import get_next_speed, get_next_lane_index
import norm_supervisor.metrics as metrics

class CollisionConstraint(AbstractConstraint):
    """Constraint for prohibiting collisions."""
    def __init__(self, min_ttc: float):
        """Initialize the collision constraint with a minimum TTC."""
        super().__init__(
            [
                ACTION_STRINGS["FASTER"],
                # SLOWER can still be norm-violating if a safe lane change is possible
                ACTION_STRINGS["SLOWER"],
                ACTION_STRINGS["IDLE"],
                # If the lane change is disallowed, the action is effectively IDLE
                ACTION_STRINGS["LANE_LEFT"],
                ACTION_STRINGS["LANE_RIGHT"]
            ]
        )
        self.min_ttc = min_ttc

    @staticmethod
    def evaluate_criterion(
        vehicle: MDPVehicle,
        action: Action,
        lane_index: Optional[LaneIndex] = None,
        check_rear: bool = False
    ) -> float:
        """Return the TTC between the ego vehicle and the leading or following vehicle."""
        # Conservative speed estimation for hard constraint satisfaction
        next_speed = max(vehicle.speed, get_next_speed(vehicle, action))
        ttc_front, ttc_rear = metrics.calculate_neighbour_ttcs(vehicle, lane_index, next_speed)
        return ttc_rear if check_rear else ttc_front

    def is_violating_action(
        self,
        vehicle: MDPVehicle,
        action: Action,
        lane_index: Optional[LaneIndex] = None,
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
        return "CollisionConstraint"

class LaneChangeCollisionConstraint(CollisionConstraint):
    """Constraint for prohibiting collisions during lane changes."""
    def __init__(self, min_ttc: float):
        """Initialize the lane change collision constraint with a minimum TTC."""
        super().__init__(min_ttc)
        self.violating_actions = [
            ACTION_STRINGS["LANE_LEFT"],
            ACTION_STRINGS["LANE_RIGHT"]
        ]
    
    def is_violating_action(self, vehicle: MDPVehicle, action: Action) -> bool:
        """Check if the action violates the lane change collision constraint."""
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
        return "LaneChangeCollisionConstraint"

class SafetyEnvelopeConstraint(AbstractConstraint):
    """Constraint for enforcing the safety envelope of the vehicle"""
    def __init__(self, safe_distance: float):
        """Initialize the safety envelope constraint with a safe distance."""
        super().__init__(
            violating_actions=[
                ACTION_STRINGS["FASTER"],
                ACTION_STRINGS["IDLE"],
                # If the lane change is disallowed, the action is effectively IDLE
                ACTION_STRINGS["LANE_LEFT"],
                ACTION_STRINGS["LANE_RIGHT"]
            ]
        )
        self.safe_distance = safe_distance

    @staticmethod
    def evaluate_criterion(
        vehicle: MDPVehicle,
        lane_index: LaneIndex = None,
        check_rear: bool = False
    ) -> float:
        """Return the distance to the vehicle ahead or behind."""
        v_front, v_rear = vehicle.road.neighbour_vehicles(vehicle, lane_index)
        v_to_check = v_rear if check_rear else v_front
        if v_to_check is not None:
            return abs(v_to_check.position[0] - vehicle.position[0] - MDPVehicle.LENGTH)
        return np.inf
    
    def is_violating_action(
            self,
            vehicle: MDPVehicle,
            action: Action,
            lane_index: LaneIndex = None,
            check_rear: bool = False
    ) -> bool:
        """Check if the action violates the safety envelope."""
        if action not in self.violating_actions:
            return False
        
        # If the action results in a lane change, this norm is not violated
        if get_next_lane_index(vehicle, action) != vehicle.target_lane_index:
            return False
        
        distance = self.evaluate_criterion(vehicle, lane_index, check_rear)
        return distance < self.safe_distance
        
    def __str__(self):
        return "SafetyEnvelopeConstraint"

class LaneChangeSafetyEnvelopeConstraint(SafetyEnvelopeConstraint):
    """Constraint for enforcing the safety envelope of the vehicle during lane changes"""
    def __init__(self, safe_distance: float):
        """Initialize the lane change safety envelope constraint with a safe distance."""
        super().__init__(safe_distance=safe_distance)
        self.violating_actions = [
            ACTION_STRINGS["LANE_LEFT"],
            ACTION_STRINGS["LANE_RIGHT"]
        ]

    def is_violating_action(self, vehicle: MDPVehicle, action: Action) -> bool:
        """Check if the action violates the safety envelope for lane changes."""
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
        return "LaneChangeSafetyEnvelopeConstraint"
