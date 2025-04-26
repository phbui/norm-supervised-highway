import numpy as np

from abc import ABC, abstractmethod
from highway_env.envs.common.action import Action, DiscreteMetaAction
from highway_env.road.road import Road
from highway_env.vehicle.controller import MDPVehicle

import metrics

ACTION_STRINGS = {val: key for key, val in DiscreteMetaAction.ACTIONS_ALL.items()}

class MetricsDrivenNorm(ABC):
    """Abstract base class for metric-driven norm constraints.
    
    Note that these norms assume perfect knowledge of the state of the environment, and rely on 
    access to the underlying vehicle model to evaluate the criteria.
    """
    def __init__(self, threshold: float = None):
        """Initialize the norm constraint with a threshold.

        :param threshold: The threshold value, above which the constraint is violated.
        """
        self.threshold = threshold

    @abstractmethod
    def evaluate_criteria(self, vehicle: MDPVehicle) -> float:
        """Evaluate the criteria for the norm constraint.

        :param vehicle: the vehicle for which to evaluate the criteria.

        :return: the value of the criteria.
        """
        pass

    @abstractmethod
    def is_violating_state(self, vehicle: MDPVehicle) -> bool:
        """Check if the provided observation violates the norm constraint.
        
        :param vehicle: the vehicle to check.

        :return: True if the norm constraint is violated, False otherwise.
        """
        pass

    @abstractmethod
    def is_violating_action(self, action: Action, vehicle: MDPVehicle) -> bool:
        """Check if the provided action violates the norm constraint.

        If the current state already violates this norm and the action reduces the severity of the 
        violation, the action is not considered norm-violating.
        
        :param action: the action to check.
        :param vehicle: the vehicle to check.

        :return: True if the provided action is norm-violating, False otherwise.
        """
        pass

class SpeedingNorm(MetricsDrivenNorm):
    """Norm constraint for enforcing the speed limit."""
    def __init__(self, speed_limit: float):
        """Initialize the speed limit constraint with a threshold.
        
        :param speed_limit: the maximum allowed speed
        """
        super().__init__(speed_limit)

    def evaluate_criteria(self, vehicle: MDPVehicle) -> float:
        """Return the speed of the ego vehicle."""
        return vehicle.speed
    
    def is_violating_state(self, vehicle: MDPVehicle) -> bool:
        """Check if the vehicle is exceeding the speed limit."""
        return self.evaluate_criteria(vehicle) > self.threshold
    
    def is_violating_action(self, action: Action, vehicle: MDPVehicle) -> bool:
        """Check if the action produces or worsens a speed limit violation."""
        # Check if the action causes the vehicle to exceed the speed limit
        if action == ACTION_STRINGS["FASTER"]:
            target_speed_index = vehicle.speed_to_index(vehicle.speed) + 1
            target_speed = vehicle.index_to_speed(target_speed_index)
            return target_speed > self.threshold
        # Check if the vehicle is already speeding and the action is not to slow down
        return self.is_violating_state(vehicle) and action != ACTION_STRINGS["SLOWER"]

class TailgatingNorm(MetricsDrivenNorm):
    """Norm constraint for enforcing a safe following distance.
    
    :param road: the road object to check for tailgating violations
    :param action_type: the action type of the ego vehicle
    :param simulation_frequency: the frequency at which the simulation is running (in Hz)
    """
    def __init__(self, road: Road, action_type: DiscreteMetaAction, simulation_frequency: float):
        super().__init__()
        self.road                 = road
        self.action_type          = action_type
        self.simulation_frequency = simulation_frequency

    def evaluate_criteria(self, vehicle: MDPVehicle) -> float:
        """Return the distance to the vehicle ahead."""
        v_front, _ = self.road.neighbour_vehicles(vehicle)
        if v_front is not None:
            return v_front.position[0] - vehicle.position[0] - MDPVehicle.LENGTH
        return np.inf

    def is_violating_state(self, vehicle: MDPVehicle, speed: float = None) -> bool:
        """Check if the vehicle is violating the safe distance to the vehicle ahead.
        
        :param vehicle: the vehicle to check
        :param speed: the speed to check. If None, the current speed of the vehicle is used
        """
        if speed is None:
            speed = vehicle.velocity[0]
        safe_distance \
            = metrics.calculate_safe_distance(speed, self.action_type, self.simulation_frequency)
        return self.evaluate_criteria(vehicle) < safe_distance
    
    def is_violating_action(self, action: Action, vehicle: MDPVehicle) -> bool:
        """Check if the action produces or worsens a tailgating violation."""
        # Check if the action causes the vehicle to violate the safe following distance
        if action == ACTION_STRINGS["FASTER"]:
            target_speed_index = vehicle.speed_to_index(vehicle.speed) + 1
            target_speed = vehicle.index_to_speed(target_speed_index)
            return self.is_violating_state(vehicle, target_speed)
        # Check if the vehicle is already tailgating and not slowing down or changing lanes
        return self.is_violating_state(vehicle) and action == ACTION_STRINGS["IDLE"]

class BrakingNorm(MetricsDrivenNorm):
    """Norm constraint for avoiding sudden braking."""
    def __init__(self, road: Road, min_ttc: float = 1.5):
        """Initialize the braking constraint with a threshold.
        
        :param road: the road object to check for braking violations
        :param min_ttc: the minimum allowed TTC value
        """
        super().__init__(min_ttc)
        self.road = road

    def evaluate_criteria(self, vehicle: MDPVehicle) -> float:
        """Return the TTC between the ego vehicle and the vehicle ahead."""
        ttc_front, _ = metrics.calculate_neighbor_ttcs(vehicle, self.road)
        return ttc_front
    
    def is_violating_state(self, vehicle: MDPVehicle) -> bool:
        """Check if the ego vehicle is within the minimum TTC to the vehicle ahead."""
        return self.evaluate_criteria(vehicle) < self.threshold
    
    def is_violating_action(self, action: Action, vehicle: MDPVehicle) -> bool:
        """Check if the action produces or worsens a braking violation."""
        return (self.is_violating_state(vehicle)
                and action in [ACTION_STRINGS["FASTER"], ACTION_STRINGS["IDLE"]])
