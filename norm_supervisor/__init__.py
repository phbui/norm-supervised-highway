"""
Norm Supervisor Package

A package for implementing normative supervision in the HighwayEnv driving environment.
"""

from .supervisor import Supervisor, PolicyAugmentMode, PolicyAugmentMethod
from .metrics import (
    calculate_ttc,
    calculate_neighbour_ttcs,
    calculate_tet,
    calculate_safe_distance,
    calculate_safety_score
)
from .consts import ACTION_STRINGS, VEHICLE_LENGTH

__version__ = "0.1.0"
__all__ = [
    "Supervisor",
    "PolicyAugmentMode", 
    "PolicyAugmentMethod",
    "calculate_ttc",
    "calculate_neighbour_ttcs",
    "calculate_tet",
    "calculate_safe_distance",
    "calculate_safety_score",
    "ACTION_STRINGS",
    "VEHICLE_LENGTH"
]
