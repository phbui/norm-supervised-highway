from typing import Optional

import numpy as np
import numpy.typing as npt

from highway_env.road.road import LaneIndex
from highway_env.vehicle.kinematics import Vehicle

def calculate_ttc(v_front: Vehicle, v_rear: Vehicle) -> float:
    """
    Calculate the Time-To-Collision (TTC) between two vehicles, assuming straight lanes.

    TTC is the amount of time it would take for the following vehicle to collide with the leading
    vehicle if both vehicles continue at their current velocities.

    :param v_front: the leading vehicle
    :param v_rear: the following vehicle
    :return: the TTC value in seconds, or np.inf if there is no projected collision
    """
    if v_front is None or v_rear is None:
        return np.inf
    if v_rear.position[0] >= v_front.position[0]:
        raise ValueError("The following vehicle must be behind the leading vehicle.")
    
    # check if vehicle is overlapping positions (for lane change, detects if positions are overlapping ie. collision)
    if v_front.position[0] - v_rear.position[0] <= Vehicle.LENGTH:
        return 0.0
    
    if v_rear.velocity[0] <= v_front.velocity[0]:
        return np.inf

    # TTC = (x_front - x_rear - length) / (vx_rear - vx_front)
    return ((v_front.position[0] - v_rear.position[0] - Vehicle.LENGTH)
            / (v_rear.velocity[0] - v_front.velocity[0]))

def calculate_neighbour_ttcs(
    vehicle: Vehicle,
    lane_index: Optional[LaneIndex] = None,
    next_speed: Optional[float] = None
) -> tuple[float, float]:
    """
    Calculate the TTCs for neighboring vehicles of the given vehicle, assuming straight lanes.

    :param vehicle: the vehicle for which to compute neighbor TTC values
    :param lane_index: optional lane index to check (if None, uses vehicle's current lane)
    :return: the TTC values for the leading vehicle and following vehicle.
    """
    if next_speed is None:
        vehicle_to_test = vehicle
    else:
        # Create a temporary vehicle with the next speed to calculate TTCs
        vehicle_to_test = Vehicle(
            road=vehicle.road,
            position=vehicle.position,
            heading=vehicle.heading,
            speed=next_speed,
            predition_type=vehicle.prediction_type
        )
    v_front, v_rear = vehicle.road.neighbour_vehicles(vehicle, lane_index)
    ttc_front = calculate_ttc(v_front, vehicle_to_test)
    ttc_rear  = calculate_ttc(vehicle_to_test, v_rear)
    return (ttc_front, ttc_rear)

def calculate_exposure(
    sample_history: npt.ArrayLike,
    sample_frequency: float,
    threshold: float
) -> float:
    """
    Calculate the time exposure for the provided metric using the specified threshold

    This method returns the cumulative duration for which the value of the provided metric remains
    lower than a certain threshold.

    :param sample_history: a list of samples
    :param sample_frequency: the sample frequency
    :param threshold: the threshold for exposure

    :return: the time exposure in inverted frequency units
    """
    sample_history = np.asarray(sample_history, dtype=np.float64)
    if sample_history.ndim != 1:
        raise ValueError("Sample history must be a 1D array of numerics.")
    # time exposure = sum(beta * period) where beta = 1 if value < threshold else 0
    period = 1 / sample_frequency
    return period * (sample_history < threshold).sum()

def calculate_mean_under(
    sample_history: npt.ArrayLike,
    threshold: float
) -> float:
    """
    Calculate the mean value for the provided metric under the specified threshold

    :param sample_history: a list of samples
    :param threshold: the threshold for exposure
    """
    sample_history = np.asarray(sample_history, dtype=np.float64)
    if sample_history.ndim != 1:
        raise ValueError("Sample history must be a 1D array of numerics.")
    mask = sample_history <= threshold
    if not any(mask):
        return np.nan
    return sample_history[mask].mean()
