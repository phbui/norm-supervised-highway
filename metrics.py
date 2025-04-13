import numpy as np
import numpy.typing as npt

from highway_env.road.road import Road
from highway_env.vehicle.kinematics import Vehicle

def ttc(v_front: Vehicle, v_rear: Vehicle) -> float:
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
    # There is no collision course if the following vehicle is slower than the leading vehicle
    if v_rear.velocity[0] <= v_front.velocity[0]:
        return np.inf
    # TTC = (x_front - x_rear - length) / (vx_rear - vx_front)
    return ((v_front.position[0] - v_rear.position[0] - Vehicle.LENGTH)
            / (v_rear.velocity[0] - v_front.velocity[0]))

def neighbor_ttcs(vehicle: Vehicle, road: Road) -> tuple[float, float]:
    """
    Calculate the TTCs for neighboring vehicles of the given vehicle, assuming straight lanes.

    Example usage::

        env = gymnasium.make("highway-fast-v0")
        ttc_front, ttc_rear = neighbor_ttcs(env.unwrapped.vehicle, env.unwrapped.road)

    :param vehicle: the vehicle for which to compute neighbor TTC values
    :param road: the corresponding road object from the game environment

    :return: the TTC values for the leading vehicle and following vehicle. Returns None if there is
        no projected collision between the vehicles.
    """
    if vehicle not in road.vehicles:
        raise ValueError("The given vehicle is not driving on the given road.")
    v_front, v_rear = road.neighbour_vehicles(vehicle)
    return (ttc(v_front, vehicle), ttc(vehicle, v_rear))

def tet(ttc_history: npt.ArrayLike, sample_frequency: float, ttc_threshold: float = 2.0) -> float:
    """
    Calculate the Time Exposed Time-To-Collision (TET) from a list of TTC values.

    TET is the cumulative duration for which the TTC remains lower than a certain threshold.

    :param ttc_history: a list of TTC values (in seconds)
    :param sample_frequency: the frequency at which the TTC values were sampled (in Hz)
    :param ttc_threshold: the threshold for TTC values to be considered as "exposed" (in seconds)

    :return: the TET value in seconds
    """
    ttc_history = np.asarray(ttc_history, dtype=np.float64)
    if ttc_history.ndim != 1:
        raise ValueError("TTC history must be a 1D array of floats.")
    # TET = sum(beta * step_size) where beta = 1 if ttc < ttc_threshold else 0
    step_size = 1 / sample_frequency
    return step_size * (ttc_history < ttc_threshold).sum()
