import numpy as np

from highway_env.envs.common.action import Action, DiscreteMetaAction
from highway_env.road.road import LaneIndex
from highway_env.vehicle.controller import MDPVehicle

def get_next_speed(vehicle: MDPVehicle, action: Action) -> float:
    """Return the next speed of the vehicle based on the action."""
    # Lane changes often require a speed increase
    if DiscreteMetaAction.ACTIONS_ALL[action] in ["FASTER", "LANE_LEFT", "LANE_RIGHT"]:
        speed_index_delta = 1
    elif DiscreteMetaAction.ACTIONS_ALL[action] == "SLOWER":
        speed_index_delta = -1
    else:
        speed_index_delta = 0
    
    # NOTE: For speed control logic, refer to `highway_env.vehicle.controller.MDPVehicle.act()`.
    target_speed_index = vehicle.speed_to_index(vehicle.speed) + speed_index_delta
    target_speed_index = int(
        np.clip(target_speed_index, 0, vehicle.target_speeds.size - 1)
    )
    target_speed = vehicle.index_to_speed(target_speed_index)
    return target_speed

def get_next_lane_index(vehicle: MDPVehicle, action: Action) -> LaneIndex:
    """Return the next lane index of the vehicle based on the action."""
    if DiscreteMetaAction.ACTIONS_ALL[action] not in ["LANE_LEFT", "LANE_RIGHT"]:
        return vehicle.target_lane_index

    _from, _to, _id = vehicle.target_lane_index
    lane_delta = 1 if DiscreteMetaAction.ACTIONS_ALL[action] == "LANE_RIGHT" else -1
    # Get target lane, clipping to stay in the valid range
    target_lane_index = (
        _from,
        _to,
        np.clip(_id + lane_delta, 0, len(vehicle.road.network.graph[_from][_to]) - 1),
    )
    # Return no-op if target lane is not reachable
    if not vehicle.road.network.get_lane(target_lane_index).is_reachable_from(vehicle.position):
        return vehicle.target_lane_index
    return target_lane_index
