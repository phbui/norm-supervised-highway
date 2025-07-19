from abc import ABC, abstractmethod
from highway_env.envs.common.action import Action
from highway_env.vehicle.kinematics import Vehicle

class AbstractConstraint(ABC):
    """Abstract base class for constraints."""

    def __init__(self, violating_actions: list[Action]) -> None:
        """Initialize the constraint with a list of potentially violating actions.

        :param violating_actions: list of potentially constraint-violating actions.
        """
        if not violating_actions:
            raise ValueError("Norm must have at least one violating action.")

        self.violating_actions = violating_actions

    @staticmethod
    @abstractmethod
    def evaluate_criterion(vehicle: Vehicle, *args, **kwargs) -> any:
        """Evaluate the criterion for the norm constraint.

        :param vehicle: the vehicle for which to evaluate the criterion.
        :return: the value of the criterion.
        """
        pass

    @abstractmethod
    def is_violating_action(self, vehicle: Vehicle, action: Action, *args, **kwargs) -> bool:
        """Check if the provided action violates the norm in the vehicle's current state.

        If the current state already violates this norm and the action reduces the severity of the 
        violation, the action is not considered norm-violating.
        
        :param vehicle: the vehicle to check.
        :param action: the action to check.
        :return: True if the provided action is norm-violating, False otherwise.
        """
        pass
    
    def __str__(self):
        """To string function."""
        pass

class AbstractNorm(AbstractConstraint):
    """Abstract base class for norms, represented as weighted constraints."""

    def __init__(self, violating_actions: list[Action], weight: int = 1) -> None:
        """Initialize the norm with a weight and a list of potentially violating actions.

        :param weight: the norm weight, used for prioritization.
        :param violating_actions: list of potentially norm-violating actions.
        """
        super().__init__(violating_actions)

        if weight < 0:
            raise ValueError("Norm weight must be non-negative.")

        self.weight = weight
