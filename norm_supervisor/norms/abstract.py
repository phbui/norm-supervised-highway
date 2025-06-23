from abc import ABC, abstractmethod
from highway_env.envs.common.action import Action
from highway_env.vehicle.kinematics import Vehicle

class AbstractNorm(ABC):
    """Abstract base class for norms.

    Note that these norms may rely on the underlying vehicle model to evaluate violations.
    """
    def __init__(self, weight: int, violating_actions: list[Action]) -> None:
        """Initialize the norm with a weight and a list of potentially violating actions.

        :param weight: the norm weight, used for prioritization.
        :param violating_actions: list of potentially norm-violating actions.
        """
        self.weight            = weight
        self.violating_actions = violating_actions

    @abstractmethod
    def evaluate_criterion(self, vehicle: Vehicle, *args, **kwargs) -> float:
        """Evaluate the criterion for the norm constraint.

        :param vehicle: the vehicle for which to evaluate the criterion.
        :return: the value of the criterion.
        """
        pass

    @abstractmethod
    def is_violating_state(self, vehicle: Vehicle, *args, **kwargs) -> bool:
        """Check if the current state of the vehicle violates the norm.
        
        :param vehicle: the vehicle to check.
        :return: True if the norm is violated, False otherwise.
        """
        pass

    def is_violating_action(self, action: Action, vehicle: Vehicle, *args, **kwargs) -> bool:
        """Check if the provided action violates the norm in the vehicle's current state.

        If the current state already violates this norm and the action reduces the severity of the 
        violation, the action is not considered norm-violating.
        
        :param action: the action to check.
        :param vehicle: the vehicle to check.
        :return: True if the provided action is norm-violating, False otherwise.
        """
        if action not in self.violating_actions:
            return False
        return self.is_violating_state(vehicle, *args, **kwargs)
    
    def __str__(self):
        """To string function."""
        pass
