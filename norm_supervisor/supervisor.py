import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import entropy
from scipy.stats import wasserstein_distance

from stable_baselines3 import DQN
from highway_env.envs.common.abstract import Observation
from highway_env.envs.common.action import Action
from highway_env.envs.highway_env import HighwayEnv
from highway_env.envs.common.action import DiscreteMetaAction

from norm_supervisor.norms.abstract import AbstractNorm
import norm_supervisor.norms.norms as norms

class Supervisor:
    """Supervisor class for enforcing metrics-driven norms in the HighwayEnv environment."""
    ACTIONS_ALL = DiscreteMetaAction.ACTIONS_ALL

    SPEED_THRESHOLD     = 25   # Speed limit (m/s) TODO: base off of simulation frequency
    BRAKING_THRESHOLD   = 2.5  # Minimum TTC (s)
    COLLISION_THRESHOLD = 0.5  # Minimum TTC (s), placeholder

    def __init__(
        self,
        env_unwrapped: HighwayEnv,
        env_config: dict,
        kl_budget: float = 0.005,
        tol: float = 1e-6,
        max_iters: int = 100,
        verbose=False):
        """Initialize the supervisor with the environment and configuration.

        :param env_unwrapped: the unwrapped HighwayEnv environment
        :param env_config: the environment configuration
        :param kl_budget: maximum KL-divergence for the supervisory policy.
        :param verbose: whether to print debug information
        """
        self.env_unwrapped = env_unwrapped
        self.env_config    = env_config
        self.kl_budget     = kl_budget
        self.tol           = tol
        self.max_iters     = max_iters
        self.verbose       = verbose
        self.reset_norms()

    def reset_norms(self):
        """Reset the norms for the current environment.
        
        This method must be called every time the environment is reset to update the norm checkers
        with the new state of the environment.
        """
        self.contraints: list[AbstractNorm] = [
            norms.CollisionNorm(
                weight=None,
                min_ttc=self.COLLISION_THRESHOLD
            ),
            norms.LaneChangeCollisionNorm(
                weight=None,
                min_ttc=self.COLLISION_THRESHOLD,
            ),
        ]
        self.norms: list[AbstractNorm] = [
            norms.SpeedingNorm(
                weight=5, 
                speed_limit=self.SPEED_THRESHOLD
            ),
            norms.TailgatingNorm(
                weight=5,
                action_type=self.env_unwrapped.action_type,
                simulation_frequency=self.env_config["simulation_frequency"]
            ),
            norms.BrakingNorm(
                weight=5,
                min_ttc=self.BRAKING_THRESHOLD
            ),
            norms.LaneChangeTailgatingNorm(
                weight=50,
                action_type=self.env_unwrapped.action_type,    
                simulation_frequency=self.env_config["simulation_frequency"]
            ),
            norms.LaneChangeBrakingNorm(
                weight=50,
                min_ttc=self.BRAKING_THRESHOLD
            )
        ]
        self.norm_weights = {str(norm): norm.weight for norm in self.norms}
    
    def _is_permissible(self, action: Action) -> bool:
        """Checks if an action violates any hard constraints."""
        return not any(c.is_violating_action(self.env_unwrapped.vehicle, action)
                       for c in self.contraints)

    def _count_norm_violations(self, action: Action) -> dict[str, int]:
        """Return a dictionary mapping norms to violation counts."""
        violations_dict = {str(norm): 0 for norm in self.norms}
        for norm in self.norms:
            if norm.is_violating_action(action, self.env_unwrapped.vehicle):
                violations_dict[str(norm)] += 1
        return  violations_dict

    def _weight_norm_violations(self, violations_count: dict[str, int]) -> dict[str, int]:
        """Return a dictionary mapping norms to weighted violation counts."""
        return {norm: violations_count[norm] * self.norm_weights[norm] for norm in violations_count}

    def _get_model_policy(self, model: DQN, obs: np.ndarray) -> np.ndarray:
        """Return the action probabilities from the model for the given observation."""
        default_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        device = getattr(model, "device", default_device)

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_tensor     = model.q_net(obs_tensor)            # shape (1, n_actions)
            probs_tensor = F.softmax(q_tensor, dim=-1)        # still on device

        action_probs = probs_tensor.squeeze(0).cpu().numpy()  # shape (n_actions,)
        if self.verbose:
            print("Action Probabilities:")
            for action, prob in enumerate(action_probs):
                print(f"  {self.ACTIONS_ALL[action]}: {prob:.3f}")
        return action_probs
    
    def _get_safe_policy(self) -> np.ndarray:
        """Computes the maximally safe policy according to the norm base."""
        violation_tensor = torch.zeros(len(self.ACTIONS_ALL))
        for i, action in enumerate(self.ACTIONS_ALL):
            if self._is_permissible(action):
                violations_count = self._count_norm_violations(action)
                violation_tensor[i] = -sum(self._weight_norm_violations(violations_count).values())
            else:
                violation_tensor[i] = -torch.inf
            
        safe_policy = F.softmax(violation_tensor, dim=0)
        if self.verbose:
            print("Safe Policy Probabilities:")
            for action, prob in enumerate(safe_policy):
                print(f"  {self.ACTIONS_ALL[action]}: {prob:.3f}")
        return safe_policy.numpy()
    
    def _update_policy(self, model_policy: np.ndarray, safe_policy: np.ndarray) -> np.ndarray:
        """Estimate the safest policy for which the KL-divergence is less than the threshold.
        
        :param model_policy: action probabilities from the RL agent.
        :param safe_policy: maximally safe action probabilities.
        :return: updated policy that is a convex combination of the model and safe policies.
        """
        def calculate_policy_update(
            _model_policy: np.ndarray,
            _safe_policy: np.ndarray,
            _lambda: float
        ) -> np.ndarray:
            """Calculate the updated policy based on the model and safe policies."""
            updated_policy = _model_policy ** (1 - _lambda) * _safe_policy ** _lambda
            updated_policy /= sum(updated_policy)
            return updated_policy

        # Filter out impermissible actions
        permissible_mask = ~np.isclose(safe_policy, 0.0)
        model_policy_filtered = model_policy[permissible_mask]
        safe_policy_filtered  = safe_policy[permissible_mask]
        model_policy_filtered /= sum(model_policy_filtered)
        safe_policy_filtered  /= sum(safe_policy_filtered)
        if self.verbose:
            permissible_indices = np.nonzero(permissible_mask)[0]
            print("Filtered Model Policy Probabilities:")
            for action, prob in zip(permissible_indices, model_policy_filtered):
                print(f"  {self.ACTIONS_ALL[action]}: {prob:.3f}")

        low, high = 0.0, 1.0
        for _ in range(self.max_iters):
            mid = (low + high) / 2
            updated_policy_filtered \
                = calculate_policy_update(model_policy_filtered, safe_policy_filtered, mid)
            kl_value = entropy(model_policy_filtered, updated_policy_filtered, nan_policy='raise')

            if abs(kl_value - self.kl_budget) < self.tol:
                break  # Found a good estimate
            if kl_value > self.kl_budget:
                high = mid # Too far, zoom in on lower half
            else:
                low = mid # Not far enough, zoom in on top half

        # Return best estimate of the updated policy
        best_estimate = (low + high) / 2
        updated_policy_filterd \
            = calculate_policy_update(model_policy_filtered, safe_policy_filtered, best_estimate)
        updated_policy = np.zeros_like(model_policy)
        updated_policy[permissible_mask] = updated_policy_filterd
        if self.verbose:
            print(f"Best Estimate: {best_estimate:.3f}")
            print(f"Updated Policy KL-Divergence: {entropy(model_policy_filtered, updated_policy_filtered)}")
            print("Updated Policy Probabilities:")
            for action, prob in enumerate(updated_policy):
                print(f"  {self.ACTIONS_ALL[action]}: {prob:.3f}")
        return updated_policy

    def decide_action(self, model: DQN, obs: Observation) -> Action:
        """Decide which action the agent should take to comply with norms.
        
        Manually selects an action which reduces the number of norm violations.

        :param model: DQN model
        :param obs: observation from the environment

        :return: norm-compliant action selection
        """ 
        model_policy = self._get_model_policy(model, obs)
        safe_policy  = self._get_safe_policy()
        updated_policy = self._update_policy(model_policy, safe_policy)
        return updated_policy.argmax()
        
    @staticmethod
    def print_obs(obs: Observation):
        """Print kinematic observation data for the ego vehicle and all other present vehicles."""
        _, ego_x, ego_y, ego_vx, ego_vy = obs[0]
        print(f"Ego Vehicle: x={ego_x}, y={ego_y}, vx={ego_vx}, vy={ego_vy}")

        for presence, x, y, vx, vy in obs[1:]:
            if presence:
                print(f"Vehicle: x={x}, y={y}, vx={vx}, vy={vy}")


# notes
# when changing lanes, the ego vehicle acclerates first then moves to the new lane, 
# we need to check if the ego vehicle is colliding with the vehicle in the same lane before the lane change
# some collisions are caused by ego and a another vehicle switching into the same lane from opposite directions, this is a nuanced issue we might want to track and handle 
# might need different logic for lane chage with leading vs following vehicles