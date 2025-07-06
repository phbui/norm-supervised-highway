import numpy as np
from typing import Optional, Protocol

from highway_env.envs.common.action import Action, DiscreteMetaAction
from highway_env.road.road import Road, LaneIndex
from highway_env.vehicle.controller import MDPVehicle

import norm_supervisor.metrics as metrics
from norm_supervisor.norms.abstract import AbstractNorm

ACTION_STRINGS = {val: key for key, val in DiscreteMetaAction.ACTIONS_ALL.items()}

def get_next_speed(vehicle: MDPVehicle, action: Action = ACTION_STRINGS["IDLE"]) -> float:
    """Return the next speed of the vehicle based on the action.
    
    :param vehicle: the vehicle to get the next speed for
    :param action: the action to predict speed for. If None, defaults to IDLE
    """   
    # Lane changes often require a speed increase
    if action in [ACTION_STRINGS["FASTER"], ACTION_STRINGS["LANE_LEFT"], ACTION_STRINGS["LANE_RIGHT"]]:
        speed_index_change = 1
    elif action == ACTION_STRINGS["SLOWER"]:
        speed_index_change = -1
    else:
        speed_index_change = 0
    
    # NOTE: For speed control logic, refer to `highway_env.vehicle.controller.MDPVehicle.act()`.
    target_speed_index = vehicle.speed_to_index(vehicle.speed) + speed_index_change
    target_speed_index = int(
        np.clip(target_speed_index, 0, vehicle.target_speeds.size - 1)
    )
    target_speed = vehicle.index_to_speed(target_speed_index)
    return target_speed

class SpeedingNorm(AbstractNorm):
    """Norm constraint for enforcing the speed limit."""
    def __init__(self, weight: int, speed_limit: float):
        """Initialize the speed limit constraint with a threshold.
        
        :param speed_limit: the maximum allowed speed
        """
        super().__init__(
        weight,
        [
            ACTION_STRINGS["IDLE"],
            ACTION_STRINGS["FASTER"]
        ])
        self.speed_limit = speed_limit

    def evaluate_criterion(self, vehicle: MDPVehicle) -> float:
        """Return the speed of the ego vehicle."""
        return vehicle.speed
    
    def is_violating_state(self, vehicle: MDPVehicle) -> bool:
        """Check if the vehicle is exceeding the speed limit."""
        return self.evaluate_criterion(vehicle) > self.speed_limit
    
    def is_violating_action(self, action: Action, vehicle: MDPVehicle) -> bool:
        """Check if the action produces or worsens a speed limit violation."""
        if action not in self.violating_actions:
            return False
        
        # Get the predicted speed for this action
        target_speed = get_next_speed(vehicle, action)
        return target_speed > self.speed_limit
    
    def __str__(self):
        """To string function."""
        return "Speeding"

class TailgatingNorm(AbstractNorm):
    """Norm constraint for enforcing a safe following distance."""
    def __init__(self, weight: int, road: Road, action_type: DiscreteMetaAction, simulation_frequency: float):
        """Initialize the tailgating constraint with a threshold.

        :param road: the road object to check for tailgating violations
        :param action_type: the action type of the ego vehicle
        :param simulation_frequency: the frequency at which the simulation is running (in Hz)
        """
        super().__init__(
        weight,
        violating_actions=[
            ACTION_STRINGS["IDLE"],
            ACTION_STRINGS["FASTER"],
            ACTION_STRINGS["SLOWER"], # experimental: slower action can still be tailgating vs lane change
        ])
        self.road                 = road
        self.action_type          = action_type
        self.simulation_frequency = simulation_frequency

    def evaluate_criterion(self, vehicle: MDPVehicle, lane_index: LaneIndex = None) -> float:
        """Return the distance to the vehicle ahead."""
        v_front, _ = self.road.neighbour_vehicles(vehicle, lane_index)
        if v_front is not None:
            return v_front.position[0] - vehicle.position[0] - MDPVehicle.LENGTH
        return np.inf

    def is_violating_state(
        self,
        vehicle: MDPVehicle,
        lane_index: LaneIndex = None,
        speed: Optional[float] = None
    ) -> bool:
        """Check if the vehicle is violating the safe distance to the vehicle ahead.
        
        :param vehicle: the vehicle to check
        :param speed: the speed to check. If None, the current speed of the vehicle is used
        """
        speed = speed or vehicle.velocity[0]
        safe_distance = metrics.calculate_safe_distance(
            speed, self.action_type, self.simulation_frequency
        )
        return self.evaluate_criterion(vehicle, lane_index) < safe_distance
    
    def is_violating_action(
        self,
        action: Action,
        vehicle: MDPVehicle,
        lane_index: LaneIndex = None
    ) -> bool:
        """Check if the action produces or worsens a tailgating violation."""
        if action not in self.violating_actions:
            return False
        
        # Get the predicted speed for this action
        target_speed = get_next_speed(vehicle, action)
        return self.is_violating_state(vehicle, lane_index, target_speed)
        
    def __str__(self):
        """To string function."""
        return "Tailgating"

class BrakingNorm(AbstractNorm):
    """Norm constraint for avoiding sudden braking."""
    def __init__(self, weight: int, road: Road, min_ttc: float = 1.5):
        """Initialize the braking constraint with a threshold.
        
        :param road: the road object to check for braking violations
        :param min_ttc: the minimum allowed TTC value
        """
        super().__init__(
        weight,
        [
            ACTION_STRINGS["SLOWER"]
        ])
        self.road    = road
        self.min_ttc = min_ttc

    def evaluate_criterion(self, vehicle: MDPVehicle, lane_index: LaneIndex = None, next_speed: Optional[float] = None) -> float:
        """Return the TTC between the ego vehicle and the rear following vehicle."""
        _, ttc_rear = metrics.calculate_neighbour_ttcs(vehicle, self.road, lane_index, next_speed ) 
        return ttc_rear
    
    def is_violating_state(self, vehicle: MDPVehicle, lane_index: LaneIndex = None, next_speed: Optional[float] = None) -> bool:
        """Check if the ego vehicle is within the minimum TTC to the vehicle ahead."""
        return self.evaluate_criterion(vehicle, lane_index, next_speed) < self.min_ttc
    
    def is_violating_action(self, action: Action, vehicle: MDPVehicle, lane_index: LaneIndex = None) -> bool:
        """Check if the action produces or worsens a braking violation."""
        if action not in self.violating_actions:
            return False
        
        # Get the predicted speed for this action
        target_speed = get_next_speed(vehicle, action)
        return self.is_violating_state(vehicle, lane_index, target_speed)

    def __str__(self):
        """To string function."""
        return "Braking"

class CollisionNorm(AbstractNorm):
    """Norm constraint for avoiding collisions."""
    def __init__(self, weight: int, road: Road, min_ttc: float = 1.5):
        """Initialize the collision constraint with a threshold.

        :param road: the road object to check for collision violations
        :param min_ttc: the minimum allowed TTC value
        """
        super().__init__(
            weight,
            [
                ACTION_STRINGS["LANE_LEFT"],
                ACTION_STRINGS["IDLE"],
                ACTION_STRINGS["LANE_RIGHT"],
                ACTION_STRINGS["FASTER"],
                ACTION_STRINGS["SLOWER"] 
                # testing: might still need failsafe slowdown if every action triggers collision
                # technically, slowing down can cause a collision if the vehicle behind is too close so need to account for this
            ]
        )
        self.road    = road
        self.min_ttc = min_ttc

    def evaluate_criterion(self, vehicle: MDPVehicle, lane_index: LaneIndex = None, next_speed: Optional[float] = None) -> tuple[float, float]:
        """Return the TTC between the ego vehicle and the vehicle ahead."""
        ttc_front, ttc_rear = metrics.calculate_neighbour_ttcs(vehicle, self.road, lane_index, next_speed) 
        return (ttc_front, ttc_rear)
        
        
    
    def is_violating_state(self, vehicle: MDPVehicle, lane_index: LaneIndex = None, next_speed: Optional[float] = None) -> bool:
        """Check if the ego vehicle is within the minimum TTC to the vehicle ahead."""
        # return self.evaluate_criteria(vehicle, lane_index) < self.min_ttc
        ttc_front, ttc_rear = self.evaluate_criterion(vehicle, lane_index, next_speed) 
        return ttc_front < self.min_ttc or ttc_rear < self.min_ttc

    def is_violating_action(self, action: Action, vehicle: MDPVehicle, lane_index: LaneIndex = None) -> bool:
        """Check if the action produces or worsens a collision violation."""
        if action not in self.violating_actions:
            return False
        
        # Get the predicted speed for this action
        target_speed = get_next_speed(vehicle, action)
        # DEBUG breakout logic #######
        x =  self.is_violating_state(vehicle, lane_index, target_speed)

        if action == ACTION_STRINGS["LANE_RIGHT"] and x:
            print("LANE RIGHT VIOLATION", x, )
            self.is_violating_state(vehicle, lane_index, target_speed)
        
        return x
        ##############

    def __str__(self):
        """To string function."""
        return "Collision"
    
class LaneChangeNormProtocol(Protocol):
    """Protocol for lane change norm constraints."""
    road: Road

    def is_violating_action(self, action: Action, vehicle: MDPVehicle) -> bool:
        """Check if the action produces a lane change violation."""
        pass

    def is_violating_state(
            self,
            vehicle: MDPVehicle,
            lane_index: LaneIndex = None,
            next_speed: Optional[float] = None
    ) -> bool:
        """Check if the current state violates the norm."""
        pass

    def __str__(self):
        """To string function."""
        pass

class LaneChangeNormMixin:
    """Mixin class for shared lane change norm logic."""
    def is_violating_action(
            self: LaneChangeNormProtocol,
            action: Action,
            vehicle: MDPVehicle
    ) -> bool:
        """Check if the action produces a lane change violation."""
        # TODO: getting violations for lane changes that are not possible, need to account for this
        if action == ACTION_STRINGS["LANE_RIGHT"]:
            _from, _to, _id = vehicle.target_lane_index
            target_lane_index = (
                _from,
                _to,
                np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1),
            )
            if self.road.network.get_lane(target_lane_index).is_reachable_from(
                vehicle.position
            ):
                target_lane_index = target_lane_index
        elif action == ACTION_STRINGS["LANE_LEFT"]:
            _from, _to, _id = vehicle.target_lane_index
            target_lane_index = (
                _from,
                _to,
                np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1),
            )
            if self.road.network.get_lane(target_lane_index).is_reachable_from(
                vehicle.position
            ):
                target_lane_index = target_lane_index
        else:
            # This condition should never be reached, since there is no reason to check a lane change
            # norm violation if the action is not a lane change
            return False # TODO: this does get called from count_and_weigh_norm_violations

        # Simply check if the current state would violate the norm in the target lane
        # The parent class's is_violating_state method already handles both front and rear vehicles
        next_speed = get_next_speed(vehicle, action)
        return self.is_violating_state(vehicle, target_lane_index, next_speed)
    
    def __str__(self):
        """To string function."""
        return "LaneChange"

class LaneChangeTailgatingNorm(LaneChangeNormMixin, TailgatingNorm):
    """Norm constraint for enforcing safe distances for lane changes."""
    def __init__(self, weight: int, road: Road, action_type: DiscreteMetaAction, simulation_frequency: float):
        """Initialize the lane change tailgating constraint with a threshold.

        :param road: the road object to check for lane change violations
        :param min_ttc: the minimum allowed TTC value
        """
        super().__init__(weight, road, action_type, simulation_frequency)
        self.violating_actions = [
            ACTION_STRINGS["LANE_LEFT"],
            ACTION_STRINGS["LANE_RIGHT"]
        ]

    def __str__(self):
        """To string function."""
        return "LaneChangeTailgating"

class LaneChangeBrakingNorm(LaneChangeNormMixin, BrakingNorm):
    """Norm constraint for avoiding sudden braking due to lane changes."""
    def __init__(self, weight: int, road: Road, min_ttc: float = 1.5):
        """Initialize the braking constraint with a threshold.

        :param road: the road object to check for braking violations
        :param min_ttc: the minimum allowed TTC value
        """
        super().__init__(weight, road, min_ttc)
        self.violating_actions = [
            ACTION_STRINGS["LANE_LEFT"],
            ACTION_STRINGS["LANE_RIGHT"]
        ]

    def __str__(self):
        """To string function."""
        return "LaneChangeBraking"

class LaneChangeCollisionNorm(LaneChangeNormMixin, CollisionNorm):
    """Norm constraint for avoiding collisions due to lane changes."""
    def __init__(self, weight: int, road: Road, min_ttc: float = 0.5):
        """Initialize the collision constraint with a threshold.

        :param road: the road object to check for collision violations
        :param min_ttc: the minimum allowed TTC value
        """
        super().__init__(weight, road, min_ttc)
        self.violating_actions = [
            ACTION_STRINGS["LANE_LEFT"],
            ACTION_STRINGS["LANE_RIGHT"]
        ]

    def __str__(self):
        """To string function."""
        return "LaneChangeCollision"
