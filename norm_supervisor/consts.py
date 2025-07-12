from highway_env.envs.common.action import DiscreteMetaAction

ACTION_STRINGS: dict[str, int] = {val: key for key, val in DiscreteMetaAction.ACTIONS_ALL.items()}
