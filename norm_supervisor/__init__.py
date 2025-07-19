"""
Norm Supervisor Package

A package for implementing normative supervision in the HighwayEnv driving environment.
"""

from .supervisor import Supervisor, PolicyAugmentMethod
from .metrics import calculate_ttc, calculate_neighbour_ttcs, calculate_exposure
from .consts import ACTION_STRINGS, VEHICLE_LENGTH

__version__ = "0.1.0"
__all__ = [
    "Supervisor",
    "PolicyAugmentMethod",
    "calculate_ttc",
    "calculate_neighbour_ttcs",
    "calculate_exposure",
    "ACTION_STRINGS",
    "VEHICLE_LENGTH"
]
