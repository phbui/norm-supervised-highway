import numpy as np
from typing import Optional, Protocol

from abc import ABC, abstractmethod
from highway_env.envs.common.action import Action, DiscreteMetaAction
from highway_env.road.road import Road, LaneIndex
from highway_env.vehicle.controller import MDPVehicle

import metrics

ACTION_STRINGS = {val: key for key, val in DiscreteMetaAction.ACTIONS_ALL.items()}

class MetricsDrivenNorm(ABC):
    """Abstract base class for metric-driven norm constraints.
    
    Note that these norms assume perfect knowledge of the state of the environment, and rely on 
    access to the underlying vehicle model to evaluate the criteria.
    """
    def __init__(self, weight: int, violating_actions: list[Action]):
        """Initialize the norm constraint with a threshold.

        :param violating_actions: dict of potentially norm-violating actions to associate predicates
            for evaluating whether an action is norm-violating
        """
        self.weight = weight
        self.violating_actions = violating_actions

    @abstractmethod
    def evaluate_criteria(self, vehicle: MDPVehicle, *args, **kwargs) -> float:
        """Evaluate the criteria for the norm constraint.

        :param vehicle: the vehicle for which to evaluate the criteria.

        :return: the value of the criteria.
        """
        pass

    @abstractmethod
    def is_violating_state(self, vehicle: MDPVehicle, *args, **kwargs) -> bool:
        """Check if the provided observation violates the norm constraint.
        
        :param vehicle: the vehicle to check.

        :return: True if the norm constraint is violations_dict, False otherwise.
        """
        pass

    def is_violating_action(self, action: Action, vehicle: MDPVehicle, *args, **kwargs) -> bool:
        """Check if the provided action violates the norm constraint.

        If the current state already violates this norm and the action reduces the severity of the 
        violation, the action is not considered norm-violating.
        
        :param action: the action to check.
        :param vehicle: the vehicle to check.

        :return: True if the provided action is norm-violating, False otherwise.
        """
        if action not in self.violating_actions:
            return False
        return self.is_violating_state(vehicle, *args, **kwargs)

    @staticmethod
    def get_next_speed(vehicle: MDPVehicle) -> float:
        """Return the next speed of the vehicle if the agent chooses to go faster."""
        # NOTE: The following speed control logic is copied from the HighwayEnv code.
        # Refer to: `highway_env.vehicle.controller.MDPVehicle.act()`
        target_speed_index = vehicle.speed_to_index(vehicle.speed) + 1
        target_speed_index = int(
            np.clip(vehicle.speed_index, 0, vehicle.target_speeds.size - 1)
        )
        target_speed = vehicle.index_to_speed(target_speed_index)
        return target_speed
    
    def __str__(self):
        """To string function."""
        pass

class SpeedingNorm(MetricsDrivenNorm):
    """Norm constraint for enforcing the speed limit."""
    def __init__(self, weight: int, speed_limit: float):
        """Initialize the speed limit constraint with a threshold.
        
        :param speed_limit: the maximum allowed speed
        """
        super().__init__(
        weight,
        [
            ACTION_STRINGS["LANE_LEFT"],
            ACTION_STRINGS["IDLE"],
            ACTION_STRINGS["LANE_LEFT"],
            ACTION_STRINGS["FASTER"]
        ])
        self.speed_limit = speed_limit

    def evaluate_criteria(self, vehicle: MDPVehicle) -> float:
        """Return the speed of the ego vehicle."""
        return vehicle.speed
    
    def is_violating_state(self, vehicle: MDPVehicle) -> bool:
        """Check if the vehicle is exceeding the speed limit."""
        return self.evaluate_criteria(vehicle) > self.speed_limit
    
    def is_violating_action(self, action: Action, vehicle: MDPVehicle) -> bool:
        """Check if the action produces or worsens a speed limit violation."""
        if action not in self.violating_actions:
            return False
        # Check if the action causes the vehicle to exceed the speed limit
        if action == ACTION_STRINGS["FASTER"]:
            target_speed = super().get_next_speed(vehicle)
            return target_speed > self.speed_limit
        # Check if the vehicle is already speeding and the action is not to slow down
        return self.is_violating_state(vehicle)
    
    def __str__(self):
        """To string function."""
        return "Speeding"

class TailgatingNorm(MetricsDrivenNorm):
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
            ACTION_STRINGS["FASTER"]
        ])
        self.road                 = road
        self.action_type          = action_type
        self.simulation_frequency = simulation_frequency

    def evaluate_criteria(self, vehicle: MDPVehicle, lane_index: LaneIndex = None) -> float:
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
        return self.evaluate_criteria(vehicle, lane_index) < safe_distance
    
    def is_violating_action(
        self,
        action: Action,
        vehicle: MDPVehicle,
        lane_index: LaneIndex = None
    ) -> bool:
        """Check if the action produces or worsens a tailgating violation."""
        if action not in self.violating_actions:
            return False
        # Check if the action causes the vehicle to violate the safe following distance
        if action == ACTION_STRINGS["FASTER"]:
            target_speed = super().get_next_speed(vehicle)
            return self.is_violating_state(vehicle, lane_index, target_speed)
        # Check if the vehicle is already tailgating and not slowing down or changing lanes
        return self.is_violating_state(vehicle, lane_index)
        
    def __str__(self):
        """To string function."""
        return "Tailgating"

class BrakingNorm(MetricsDrivenNorm):
    """Norm constraint for avoiding sudden braking."""
    def __init__(self, weight: int, road: Road, min_ttc: float = 1.5):
        """Initialize the braking constraint with a threshold.
        
        :param road: the road object to check for braking violations
        :param min_ttc: the minimum allowed TTC value
        """
        super().__init__(
        weight,
        [
            ACTION_STRINGS["IDLE"],
            ACTION_STRINGS["FASTER"]
        ])
        self.road    = road
        self.min_ttc = min_ttc

    def evaluate_criteria(self, vehicle: MDPVehicle, lane_index: LaneIndex = None) -> float:
        """Return the TTC between the ego vehicle and the vehicle ahead."""
        ttc_front, _ = metrics.calculate_neighbour_ttcs(vehicle, self.road, lane_index)
        return ttc_front
    
    def is_violating_state(self, vehicle: MDPVehicle, lane_index: LaneIndex = None) -> bool:
        """Check if the ego vehicle is within the minimum TTC to the vehicle ahead."""
        return self.evaluate_criteria(vehicle, lane_index) < self.min_ttc
    
    def __str__(self):
        """To string function."""
        return "Braking"

class LaneChangeNormProtocol(Protocol):
    """Protocol for lane change norm constraints."""
    road: Road

    def is_violating_action(self, action: Action, vehicle: MDPVehicle) -> bool:
        """Check if the action produces a lane change violation."""
        pass

    def __str__(self):
        """To string function."""
        pass

class LaneChangeNormMixin:
    """Mixin class for shared lane change norm logic."""
    def is_violating_action(self: LaneChangeNormProtocol, action: Action, vehicle: MDPVehicle):
        """Check if the action produces a lane change violation."""
        # NOTE: The following lane change index logic is copied directly from the HighwayEnv code.
        # Refer to: `highway_env.vehicle.controller.ControlledVehicle.act()`
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
            return False

        # Check for safe distance violations to the vehicle ahead and the vehicle behind
        _, v_rear = self.road.neighbour_vehicles(vehicle, target_lane_index)
        if v_rear is not None:
            return (super().is_violating_action(action, vehicle, target_lane_index)
                    or super().is_violating_action(action, v_rear, target_lane_index))
        # If there is no vehicle behind, only check for violation to the vehicle ahead
        return super().is_violating_action(action, vehicle, target_lane_index)
    
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
