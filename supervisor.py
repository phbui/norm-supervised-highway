import numpy as np
import torch
import torch.nn.functional as F

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
            norms.SpeedingNorm(
                weight=5, 
                speed_limit=self.SPEED_THRESHOLD
            ),
            norms.TailgatingNorm(
                weight=5,
                road=self.env_unwrapped.road,
                action_type=self.env_unwrapped.action_type,
                simulation_frequency=self.env_config["simulation_frequency"]
            ),
            norms.BrakingNorm(
                weight=5,
                road=self.env_unwrapped.road
            ),
            norms.LaneChangeTailgatingNorm(
                weight=5,
                road=self.env_unwrapped.road,
                action_type=self.env_unwrapped.action_type,    
                simulation_frequency=self.env_config["simulation_frequency"]
            ),
            norms.LaneChangeBrakingNorm(
                weight=5,
                road=self.env_unwrapped.road
            )
        ]

    def count_state_norm_violations(self) -> int:
        """Return the number of norm violations for the given state."""
        violations = 0
        violations_dict = {norm: 0 for norm in self.norms}
        for norm in self.norms:
            # Skip lane change norms, since they only apply to action violations
            if isinstance(norm, norms.LaneChangeNormProtocol):
                continue
            elif norm.is_violating_state(self.env_unwrapped.vehicle):
                violations += 1
                violations_dict[norm] += 1

        return violations, violations_dict
    
    def count_action_norm_violations(self, action: Action) -> int:
        """Return the number of norm violations for a given action."""
        violations = 0
        violations_dict = {str(norm): 0 for norm in self.norms}
        for norm in self.norms:
            if norm.is_violating_action(action, self.env_unwrapped.vehicle):
                violations += 1
                violations_dict[str(norm)] += 1

        return violations, violations_dict

    def get_action_probs(self, model: DQN, obs: np.ndarray) -> np.ndarray:
        device = getattr(model, "device",
                         torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            q_tensor     = model.q_net(obs_tensor)            # shape (1, n_actions)
            probs_tensor = F.softmax(q_tensor, dim=-1)        # still on device

        action_probs = probs_tensor.squeeze(0).cpu().numpy()  # shape (n_actions,)

        if self.verbose:
            print("Action Probabilities:")
            for i, prob in enumerate(action_probs):
                print(f"  {self.ACTIONS_ALL[i]}: {prob:.3f}")

        return action_probs
    
    def get_supervised_action(self, selected_action, action_probs, violations, violations_dict):
        """
        Given raw action_probs (softmax over Q), 
        penalize norm-violating actions,
        select via ε-greedy
        """
        if self.verbose:
            print(f"Original action: {self.ACTIONS_ALL[selected_action]} | Norm violations: {violations}")

        if violations > 0:
            # Weighing likelihood of potential future actions by possibility of commiting more norm violations
            n = len(self.ACTIONS_ALL)
            violation_counts = np.zeros(n, dtype=int)
            violations_dicts = {a: 0 for a in range(n)}
            for a in range(n):
                # Instead of count_action_nrom_violations, we could look at the weight of norms and their violations here
                v, v_d = self.count_action_norm_violations(a)
                violation_counts[a] = v
                violations_dicts[a] = v_d

            # Fewer violations -> bigger weight
            weights = 1.0 / (violation_counts + 1)
            adj = action_probs * weights
            adj_sum = adj.sum()
            if adj_sum > 0:
                adj /= adj_sum
            else:
                # fallback to uniform if something weird happens
                adj = np.ones(n) / n

            if self.verbose:
                print("Adjusted action probs:")
                for i, p in enumerate(adj):
                    print(f"  {self.ACTIONS_ALL[i]}: {p:.3f} (viol={violation_counts[i]})")

            selected_action = int(adj.argmax())

            violations = violation_counts[selected_action]
            violations_dict = violations_dicts[selected_action]
        
        if self.verbose:
            print(f"Chose: {self.ACTIONS_ALL[selected_action]} | Norm violations: {violations}")

        return selected_action, violations, violations_dict

    def decide_action(self, model: DQN, obs: Observation) -> Action:
        """Decide which action the agent should take to comply with norms.
        
        Manually selects an action which reduces the number of norm violations.

        :param model: DQN model
        :param obs: observation from the environment

        :return: norm-compliant action selection
        """ 
        selected_action, _          = model.predict(obs, deterministic=True)
        selected_action             = int(selected_action)
        violations, violations_dict = self.count_action_norm_violations(selected_action)
        action_probs = self.get_action_probs(model, obs)
        selected_action, violations, violations_dict = self.get_supervised_action(selected_action, action_probs, violations, violations_dict)

        return selected_action, violations, violations_dict
        
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
