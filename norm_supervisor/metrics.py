import numpy as np
import numpy.typing as npt

from highway_env.envs.common.action import DiscreteMetaAction
from highway_env.road.road import LaneIndex
from highway_env.vehicle.controller import ControlledVehicle
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
    lane_index: LaneIndex = None,
    next_speed: float = None
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
            speed=next_speed,
        )
    v_front, v_rear = vehicle.road.neighbour_vehicles(vehicle, lane_index)
    ttc_front = calculate_ttc(v_front, vehicle_to_test)
    ttc_rear  = calculate_ttc(vehicle_to_test, v_rear)   
    return (ttc_front, ttc_rear)

def calculate_tet(ttc_history: npt.ArrayLike, simulation_frequency: float,
                  ttc_threshold: float = 2.0) -> float:
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
    step_size = 1 / simulation_frequency
    return step_size * (ttc_history < ttc_threshold).sum()

def calculate_safe_distance(speed: float, action_type: DiscreteMetaAction,
                            simulation_frequency: float) -> float:
    """
    Calculate the safe longitudinal distance for the ego vehicle.

    Implements the simplified metric from Zhao et al. (2020) assuming that the two vehicles are
    traveling in the same direction.

    :param speed: the longitudinal speed of the ego vehicle
    :param action_type: the DiscreteMetaAction action type of the ego vehicle
    :param simulation_frequency: the frequency at which the simulation is running (in Hz)

    :return: the safe longitudinal distance in meters
    """
    target_speeds = np.sort(action_type.target_speeds)
    speed_intervals = np.diff(target_speeds)
    # Maximum and minimum acceleration values permitted by the proportional speed controller
    acc_max = ControlledVehicle.KP_A * speed_intervals.max()
    decc_min = ControlledVehicle.KP_A * speed_intervals.min()
    step_size = 1 / simulation_frequency

    alpha = 0.5 * acc_max + 0.5 * acc_max ** 2 / decc_min
    beta = speed * ( 1 + acc_max / decc_min)
    gamma = 0.5 * speed ** 2 / decc_min
    return alpha * step_size ** 2 + beta * step_size + gamma 

def calculate_safety_score(
        distance: float,
        safe_distance: float,
        reward: float = 1.0,
        penalty: float = 1.0
    ) -> float:
    """
    Calculate the safety score as defined by Zhao et al. (2020).

    :param reward: the user-specified reward to achieve extra margin from the minimum safe distance
    :param penalty: the user-specified penalty of unit distance less than the minimum safe distance
    :param distance: the current distance between vehicles
    :param safe_distance: the safe distance

    :return: the safety score
    """
    # Safety score is not defined if there is no leading vehicle
    if distance == np.inf:
        return np.nan
    elif distance > safe_distance:
        return reward * (distance - safe_distance)
    else:
        return penalty * (distance - safe_distance)
