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
    COLLISION_THRESHOLD = 0.5  # Minimum TTC (s)
    EPSILON            = 0.0

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

            norms.CollisionNorm(
                weight=500,
                self.env_unwrapped.road,
                self.COLLISION_THRESHOLD
            ),
            norms.LaneChangeTailgatingNorm(
                weight=5,
                road=self.env_unwrapped.road,
                action_type=self.env_unwrapped.action_type,    
                simulation_frequency=self.env_config["simulation_frequency"]
            ),
            norms.LaneChangeCollisionNorm(
                weight=500,
                road=self.env_unwrapped.road
            ),
            norms.LaneChangeBrakingNorm(
                weight=500,
                road=self.env_unwrapped.road
            )
        ]
    
    def count_and_weigh_norm_violations(self, action: Action) -> int:
        """Return the number of norm violations for a given action."""
        violations = 0
        violations_weight = 0
        violations_dict = {str(norm): 0 for norm in self.norms}
        violations_weight_dict = {str(norm): 0 for norm in self.norms}
        for norm in self.norms:
            if norm.is_violating_action(action, self.env_unwrapped.vehicle):
                violations += 1
                violations_dict[str(norm)] += 1
                violations_weight += norm.weight
                violations_weight_dict[str(norm)] += norm.weight

                # for debug 
                # print if collision norm is violated
                if str(norm) == "Collision":
                    print("Collision norm violated for action:", self.ACTIONS_ALL[action])
                elif str(norm) == "LaneChangeCollision":
                    print("LaneChangeCollision norm violated for action:", self.ACTIONS_ALL[action])

        return violations, violations_dict, violations_weight, violations_weight_dict

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
    
    def get_supervised_action(self, selected_action, action_probs, violations, violations_dict, violations_weight, violations_weight_dict):
        """
        Given raw action_probs (softmax over Q), 
        penalize norm-violating actions,
        select via ε-greedy
        """
        mode = "greedy"
        if self.verbose:
            print(f"Original action: {self.ACTIONS_ALL[selected_action]} | Norm violations: {violations}")

        selected_violations             = violations
        selected_violations_dict        = violations_dict
        selected_violations_weight      = violations_weight
        selected_violations_weight_dict = violations_weight_dict

        if violations > 0:
            # Weighing likelihood of potential future actions by possibility of commiting more norm violations
            n = len(self.ACTIONS_ALL)
            violation_counts = np.zeros(n, dtype=int)
            violations_dicts = {a: 0 for a in range(n)}
            violation_weights = np.zeros(n, dtype=int)
            violations_weights_dicts = {a: 0 for a in range(n)}
            for a in range(n):
                # Instead of count_action_nrom_violations, we could look at the weight of norms and their violations here
                v, v_d, vw, vw_d = self.count_and_weigh_norm_violations(a)
                violation_counts[a]        = v
                violations_dicts[a]        = v_d
                violation_weights[a]        = vw
                violations_weights_dicts[a] = vw_d

            # Bigger norm weight + violation -> lower action weight (lower chance of selection)
            action_weights = 1.0 / (violation_weights + 1)
            adj_prob = action_probs * action_weights
            adj_sum = adj_prob.sum()
            if adj_sum > 0:
                adj_prob /= adj_sum
            else:
                # fallback to uniform if something weird happens
                adj_prob = np.ones(n) / n

            if self.verbose:
                print("Adjusted action probs:")
                for i, p in enumerate(adj_prob):
                    print(f"  {self.ACTIONS_ALL[i]}: {p:.3f} (viol={violation_counts[i]})")

            selected_action = int(adj_prob.argmax())

            selected_violations = violation_counts[selected_action]
            selected_violations_dict = violations_dicts[selected_action]
            selected_violations_weight = violation_weights[selected_action]
            selected_violations_weight_dict = violations_weights_dicts[selected_action]
        
        if self.verbose:
            print(f"Chose: {self.ACTIONS_ALL[selected_action]} | Norm violations: {selected_violations} | Norm violation weight: {selected_violations_weight}")

        return selected_action, selected_violations, selected_violations_dict, selected_violations_weight, selected_violations_weight_dict

    def decide_action(self, model: DQN, obs: Observation) -> Action:
        """Decide which action the agent should take to comply with norms.
        
        Manually selects an action which reduces the number of norm violations.

        :param model: DQN model
        :param obs: observation from the environment

        :return: norm-compliant action selection
        """ 
        selected_action, _          = model.predict(obs, deterministic=True)
        selected_action             = int(selected_action)
        violations, violations_dict, violations_weight, violations_weight_dict = self.count_and_weigh_norm_violations(selected_action)
        action_probs = self.get_action_probs(model, obs)
        selected_action, violations, violations_dict, violations_weight, violations_weight_dict = self.get_supervised_action(selected_action, action_probs, violations, violations_dict, violations_weight, violations_weight_dict)

        return selected_action, violations, violations_dict, violations_weight, violations_weight_dict
        
    @staticmethod
    def print_obs(obs: Observation):
        """Print kinematic observation data for the ego vehicle and all other present vehicles."""
        _, ego_x, ego_y, ego_vx, ego_vy = obs[0]
        print(f"Ego Vehicle: x={ego_x}, y={ego_y}, vx={ego_vx}, vy={ego_vy}")

        for presence, x, y, vx, vy in obs[1:]:
            if presence:
                print(f"Vehicle: x={x}, y={y}, vx={vx}, vy={vy}")
