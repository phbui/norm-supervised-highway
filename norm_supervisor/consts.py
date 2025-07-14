from highway_env.envs.common.action import DiscreteMetaAction
from highway_env.vehicle.kinematics import Vehicle

ACTION_STRINGS: dict[str, int] = {val: key for key, val in DiscreteMetaAction.ACTIONS_ALL.items()}
VEHICLE_LENGTH = Vehicle.LENGTH
