import numpy as np
import torch

from stable_baselines3 import DQN
from highway_env.envs.common.abstract import Observation
from highway_env.envs.common.action import Action
from highway_env.envs.highway_env import HighwayEnv
from highway_env.envs.common.action import DiscreteMetaAction

import norms

class Supervisor:
    """Supervisor class for enforcing metrics-driven norms in the HighwayEnv environment."""
    ACTIONS_ALL        = DiscreteMetaAction.ACTIONS_ALL
    ACTION_STRINGS     = norms.ACTION_STRINGS
    SPEED_THRESHOLD    = 30   # Speed limit (m/s)
    BRAKING_THRESHOLD  = 1.5  # Minimum TTC (s)

    def __init__(self, env_unwrapped: HighwayEnv, env_config: dict, verbose=False):
        """Initialize the supervisor with the environment and configuration.

        :param env_unwrapped: the unwrapped HighwayEnv environment
        :param env_config: the environment configuration
        :param verbose: whether to print debug information
        """
        self.env_unwrapped = env_unwrapped
        self.env_config    = env_config
        self.verbose       = verbose
        self.reset_norms()

    def reset_norms(self):
        """Reset the norms for the current environment.
        
        This method must be called every time the environment is reset to update the norm checkers
        with the new state of the environment.
        """
        self.norms: list[norms.MetricsDrivenNorm] = [
            norms.SpeedingNorm(self.SPEED_THRESHOLD),
            norms.TailgatingNorm(
                self.env_unwrapped.road,
                self.env_unwrapped.action_type,
                self.env_config["simulation_frequency"]
            ),
            norms.BrakingNorm(self.env_unwrapped.road),
            norms.LaneChangeTailgatingNorm(
                self.env_unwrapped.road,
                self.env_unwrapped.action_type,    
                self.env_config["simulation_frequency"]
            ),
            norms.LaneChangeBrakingNorm(self.env_unwrapped.road)
        ]

    def count_state_norm_violations(self) -> int:
        """Return the number of norm violations for the given state."""
        violations = 0
        for norm in self.norms:
            # Skip lane change norms, since they only apply to action violations
            if isinstance(norm, norms.LaneChangeNormProtocol):
                continue
            elif norm.is_violating_state(self.env_unwrapped.vehicle):
                violations += 1
        return violations
    
    def count_action_norm_violations(self, action: Action) -> int:
        """Return the number of norm violations for a given action."""
        violations = 0
        for norm in self.norms:
            if norm.is_violating_action(action, self.env_unwrapped.vehicle):
                violations += 1
        return violations

    def decide_action(self, model: DQN, obs: Observation) -> Action:
        """Decide which action the agent should take to comply with norms.
        
        Manually selects an action which reduces the number of norm violations.

        :param model: DQN model
        :param obs: observation from the environment

        :return: norm-compliant action selection
        """ 
        selected_action, _ = model.predict(obs, deterministic=True)
        selected_action    = int(selected_action)
        violations         = self.count_action_norm_violations(selected_action)
        if self.verbose:
            print(f"Original action: {self.ACTIONS_ALL[selected_action]} | Norm violations: {violations}")
        if violations > 0:
            available_actions = self.env_unwrapped.action_type.get_available_actions()
            # If the agent's selected action violates any norms, manually select a compliant action
            if self.ACTION_STRINGS["SLOWER"] in available_actions:
                action_override = self.ACTION_STRINGS["SLOWER"]
                new_violations  = self.count_action_norm_violations(action_override)
            else:
                action_override = self.ACTION_STRINGS["IDLE"]
                new_violations  = self.count_action_norm_violations(action_override)
            # Override the selected action if it reduces the number of violations
            if new_violations < violations:
                selected_action = action_override
                violations      = new_violations
        if self.verbose:
            print(f"New action: {self.ACTIONS_ALL[selected_action]} | Norm Violations: {violations}")
        return selected_action, violations
        
    def _decide_action(self, model: DQN, obs: Observation) -> tuple[Action, int]:
        """Decide which action the agent should take to comply with norms.
        
        Selects the action with the highest q-value from the network which also minimizes the number
        of norm violations. This method is not currently in use because it significantly increases
        the number of collisions.

        :param model: DQN model
        :param obs: observation from the environment

        :return: norm-compliant action selection
        """
        obs_tensor: torch.Tensor = model.policy.obs_to_tensor(obs)[0]
        q_values: torch.Tensor   = model.policy.q_net(obs_tensor)
        if self.verbose:
            print(f"Original action selection: {self.ACTIONS_ALL[q_values.argmax()]}")
        
        # Sort actions in descending order, so the first action is the most promising
        _, sorted_idx_tensor = torch.sort(q_values, descending=True)
        sorted_actions       = sorted_idx_tensor.tolist()[0]
        violations           = dict(zip(sorted_actions, [0] * len(sorted_actions)))
        
        available_actions = self.env_unwrapped.action_type.get_available_actions()
        selected_action   = None
        for action in sorted_actions:
            # Skip invalid actions and remove from dict
            if action not in available_actions:
                del violations[action]
                continue

            violations[action] = self.count_action_norm_violations(action)

            if violations[action] == 0:
                selected_action = action
                break
        
        # If there are no norm-compliant actions, select the best action which minimizes violations
        if selected_action is None:
            selected_action = min(violations, key=violations.get)

        if self.verbose:
            print(f"New action selection: {self.ACTIONS_ALL[selected_action]}")

        return selected_action, violations[selected_action]
    
    @staticmethod
    def print_obs(obs: Observation):
        """Print kinematic observation data for the ego vehicle and all other present vehicles."""
        _, ego_x, ego_y, ego_vx, ego_vy = obs[0]
        print(f"Ego Vehicle: x={ego_x}, y={ego_y}, vx={ego_vx}, vy={ego_vy}")

        for presence, x, y, vx, vy in obs[1:]:
            if presence:
                print(f"Vehicle: x={x}, y={y}, vx={vx}, vy={vy}")
