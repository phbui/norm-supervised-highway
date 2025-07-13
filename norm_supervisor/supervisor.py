from enum import Enum
from scipy.optimize import brentq
from scipy.stats import entropy
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from highway_env.envs.common.abstract import Observation
from highway_env.envs.common.action import Action
from highway_env.envs.common.action import DiscreteMetaAction
from highway_env.envs.highway_env import HighwayEnv
from stable_baselines3 import DQN

from norm_supervisor.norms.abstract import AbstractNorm
import norm_supervisor.norms.norms as norms

# Type alias for 1D array of floating points
FloatArray1D = npt.NDArray[np.float64] 

class SupervisorMode(Enum):
    """Enum for supervisor modes."""
    FILTER_ONLY   = "filter_only"   # Only filter out impermissible actions
    NAIVE_AUGMENT = "naive_augment" # Use naive update rule to augment the policy
    DEFAULT       = "default"       # Filter and augment the policy to minimize norm violation cost

class SupervisorMethod(Enum):
    """Enum for supervisor methods."""
    FIXED    = "fixed"    # Use a fixed beta value
    ADAPTIVE = "adaptive" # Adaptively calculate beta based on KL budget

class Supervisor:
    """Supervisor class for enforcing metrics-driven norms in the HighwayEnv environment."""
    ACTIONS_ALL = DiscreteMetaAction.ACTIONS_ALL
    SPEED_THRESHOLD     = 25   # Speed limit (m/s) TODO: base off of simulation frequency
    BRAKING_THRESHOLD   = 3    # Minimum TTC (s)
    # NOTE: The collision threshold must be greater than the policy period (1/policy_frequency).
    COLLISION_THRESHOLD = 0.5  # Minimum TTC (s)

    def __init__(
        self,
        env_unwrapped: HighwayEnv,
        env_config: dict,
        mode: str = SupervisorMode.DEFAULT.value,
        method: str = SupervisorMethod.ADAPTIVE.value,
        fixed_beta: float = 0.01,
        kl_budget: float = 0.005,
        eta_max: float = 10.0,
        eta_min: float = -10.0,
        tol: float = 1e-4,
        verbose=False
    ) -> None:
        """Initialize the supervisor with the environment and configuration.

        :param env_unwrapped: the unwrapped HighwayEnv environment.
        :param env_config: the environment configuration.
        :param mode: the mode of the supervisor ('filter_only', 'naive_augment', 'default').
        :param method: the method for computing the supervisory policy ('fixed', 'adaptive').
        :param fixed_beta: fixed beta value for the supervisory policy.
            This value is only used for the FIXED method.
        :param kl_budget: maximum KL-divergence for the supervisory policy.
            This value is only used for the ADPATIVE method.
        :param eta_max: maximum value for the KL-divergence hyperparameter (eta=log(beta)).
            This value is only used for the ADPATIVE method.
        :param eta_min: minimum value for the KL-divergence hyperparameter (eta=log(beta)).
            This value is only used for the ADPATIVE method.
        :param tol: tolerance for the KL-divergence estimation.
            This value is only used for the ADPATIVE method.
        :param verbose: whether to print debug information.
            This value is only used for the ADPATIVE method.
        """       
        self.env_unwrapped = env_unwrapped
        self.env_config    = env_config

        try:
            self.mode = SupervisorMode(mode.lower())
        except ValueError:
            raise ValueError(f"Invalid mode: {mode}. Expected one of "
                             f"{[m.value for m in SupervisorMode]}")

        try:
            self.method = SupervisorMethod(method.lower())
        except ValueError:
            raise ValueError(f"Invalid method: {method}. Expected one of "
                             f"{[m.value for m in SupervisorMethod]}")

        self.fixed_beta    = fixed_beta
        self.kl_budget     = kl_budget
        self.eta_max       = eta_max
        self.eta_min       = eta_min
        self.tol           = tol
        self.verbose       = verbose

        self.reset_norms()

    def reset_norms(self) -> None:
        """Reset the norms for the current environment.
        
        This method must be called every time the environment is reset to update the norm checkers
        with the new state of the environment.
        """
        self.contraints: list[AbstractNorm] = [
            norms.BrakingNorm(
                weight=None,
                min_ttc=self.COLLISION_THRESHOLD
            ),
            norms.LaneChangeBrakingNorm(
                weight=None,
                min_ttc=self.COLLISION_THRESHOLD,
            ),
        ]
        self.norms: list[AbstractNorm] = [
            norms.SpeedingNorm(
                weight=4,
                speed_limit=self.SPEED_THRESHOLD
            ),
            norms.TailgatingNorm(
                weight=4,
                action_type=self.env_unwrapped.action_type,
                simulation_frequency=self.env_config["simulation_frequency"]
            ),
            norms.BrakingNorm(
                weight=5,
                min_ttc=self.BRAKING_THRESHOLD
            ),
            norms.LaneChangeTailgatingNorm(
                weight=4,
                action_type=self.env_unwrapped.action_type,    
                simulation_frequency=self.env_config["simulation_frequency"]
            ),
            norms.LaneChangeBrakingNorm(
                weight=5,
                min_ttc=self.BRAKING_THRESHOLD
            )
        ]
        self.norm_weights = {str(norm): norm.weight for norm in self.norms}
    
    def is_permissible(self, action: Action) -> bool:
        """Checks if an action violates any hard constraints."""
        return not any(c.is_violating_action(self.env_unwrapped.vehicle, action)
                       for c in self.contraints)

    def count_norm_violations(self, action: Action) -> dict[str, int]:
        """Return a dictionary mapping norms to violation counts for the given action."""
        violations_dict = {str(norm): 0 for norm in self.norms}
        for norm in self.norms:
            if norm.is_violating_action(self.env_unwrapped.vehicle, action):
                violations_dict[str(norm)] += 1
        return  violations_dict

    def weight_norm_violations(self, violations_count: dict[str, int]) -> dict[str, int]:
        """Return a dictionary of weighted norm costs given a dictionary of norm violations."""
        return {norm: violations_count[norm] * self.norm_weights[norm] for norm in violations_count}
    
    def get_norm_violation_cost(self) -> FloatArray1D:
        """Return the norm violation cost vector for the current state."""
        norm_violation_cost = np.zeros(len(self.ACTIONS_ALL))
        for action in self.ACTIONS_ALL.keys():
            violations_count = self.count_norm_violations(action)
            weighted_violations = self.weight_norm_violations(violations_count)
            norm_violation_cost[action] = sum(weighted_violations.values())

        return norm_violation_cost

    def _get_model_policy(self, model: DQN, obs: FloatArray1D) -> FloatArray1D:
        """Return the action probabilities from the model for the given observation."""
        default_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        device = getattr(model, "device", default_device)

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_tensor     = model.q_net(obs_tensor)            # Shape: (1, num_actions)
            probs_tensor = F.softmax(q_tensor, dim=-1)        # Still on device

        action_probs = probs_tensor.squeeze(0).cpu().numpy()  # Shape: (num_actions,)
        if self.verbose:
            print("Model Policy Probabilities:")
            for action, prob in enumerate(action_probs):
                print(f"  {self.ACTIONS_ALL[action]}: {prob:.3f}")
        return action_probs
    
    def _filter_policy(self, policy: FloatArray1D) -> FloatArray1D:
        """Filter the given policy to only include permissible actions.
        
        This method assigns zero probability to impermissible actions and then renormalizes the
        distribution.
        """
        permissible_mask = [self.is_permissible(action) for action in self.ACTIONS_ALL]
        # If all actions are impermissible, return the original policy.
        if not any(permissible_mask):
            if self.verbose:
                print("All actions are impermissible! Returning original policy.")
            return policy
        policy_permissible = policy[permissible_mask]
        policy_permissible /= np.sum(policy_permissible)
        policy_filtered = np.full_like(policy, 0.0)
        policy_filtered[permissible_mask] = policy_permissible
        if self.verbose:
            print("Filtered Model Policy Probabilities:")
            for action, prob in enumerate(policy_filtered):
                print(f"  {self.ACTIONS_ALL[action]}: {prob:.3f}")

        return policy_filtered
    
    def _update_policy_supports(
        self,
        log_policy_supports: FloatArray1D,
        cost_supports: FloatArray1D,
        eta: float
    ) -> FloatArray1D:
        """Update the policy supports based on the hyperparameter eta.
        
        :param log_policy_supports: precomputed policy supports in log space.
        :param cost_supports: norm violation costs over the policy supports.
        :param eta: hyperparameter for the KL-divergence estimation, where eta = log(beta).
        :return: [policy * exp(cost / beta)] / Z (normalized).
        """
        logits = log_policy_supports - cost_supports / np.exp(eta)
        logits -= np.max(logits)
        updated_policy_supports = np.exp(logits)
        updated_policy_supports /= np.sum(updated_policy_supports)
        return updated_policy_supports
    
    def _augment_policy(self, policy: FloatArray1D) -> FloatArray1D:
        """Augment the given policy to minimizes the expected norm violation cost.

        The augmented policy minimizes the expected norm violation cost subject to a constraint on
        the KL-divergence from the original policy. When there are infinite solutions, the policy
        with the smallest KL-divergence is selected.
        """
        support_mask = ~np.isclose(policy, 0.0)
        policy_supports = policy[support_mask]
        if not np.isclose(policy_supports.sum(), 1.0):
            raise ValueError(f"Policy supports do not sum to 1: {policy_supports.sum():.4f}")

        # Cache cost vector for use in all computations.
        cost = self.get_norm_violation_cost()
        cost_supports = cost[support_mask]
        if self.verbose:
            print(f"Cost vector:")
            for action, cost in enumerate(cost):
                print(f"  {self.ACTIONS_ALL[action]}: {cost:.3f}")
        
        # Return the original policy if the cost function is uniform.
        if np.allclose(cost_supports, cost_supports[0]):
            if self.verbose:
                print(f"All actions are equally norm compliant! No augmentation needed.")
            return policy

        # Check if optimal policy is within the KL budget.
        min_cost_mask = np.isclose(cost_supports, np.min(cost_supports))
        pi_cost_supports = np.zeros_like(policy_supports)
        pi_cost_supports[min_cost_mask] = 1.0 / np.sum(min_cost_mask)
        updated_policy_supports = policy_supports * min_cost_mask
        updated_policy_supports /= np.sum(updated_policy_supports)
        kl_value = entropy(updated_policy_supports, policy_supports, nan_policy='raise')
        # Otherwise, saturate KL budget to minimize cost.
        if kl_value > self.kl_budget:
            # Cache log policy supports for use in all computations.
            log_policy_supports = np.log(policy_supports)
            def kl_gap(eta: float) -> float:
                """Return the difference between the KL-divergence induced by eta and the budget."""
                updated_policy_supports \
                    = self._update_policy_supports(log_policy_supports, cost_supports, eta)
                kl_value = entropy(updated_policy_supports, policy_supports, nan_policy='raise')
                return kl_value - self.kl_budget

            kl_gap_high = kl_gap(self.eta_min)
            kl_gap_low  = kl_gap(self.eta_max)
            if kl_gap_high * kl_gap_low > 0:
                raise ValueError(f"KL constraint not bracketed: [({self.eta_min}, {kl_gap_high:.4f}), "
                                f"({self.eta_max}, {kl_gap_low:.4f})]")

            if self.method == SupervisorMethod.FIXED:
                eta_star = np.log(self.fixed_beta)
            elif self.method == SupervisorMethod.ADAPTIVE:
                # Solve for eta using Brent's root-finding algorithm.
                eta_star, root_results = brentq(
                    f=kl_gap,
                    a=self.eta_min,
                    b=self.eta_max,
                    xtol=self.tol,
                    full_output=True
                )
                if not root_results.converged:
                    print(f"WARNING: KL-divergence estimation did not converge: {root_results.flag}")
            else:
                raise ValueError(f"Unknown supervisor method: {self.method}")

            # Update policy supports.
            updated_policy_supports \
                = self._update_policy_supports(log_policy_supports, cost_supports, eta_star)
            if self.verbose:
                print(f"Best Estimate: {eta_star:.3f}")

        # Construct updated policy from new supports.
        updated_policy = np.full_like(policy, 0.0)
        updated_policy[support_mask] = updated_policy_supports
        if self.verbose:
            print(f"Updated policy KL-divergence: {entropy(updated_policy_supports, policy_supports)}")
            print("Updated policy:")
            for action, prob in enumerate(updated_policy):
                print(f"  {self.ACTIONS_ALL[action]}: {prob:.3f}")
        return updated_policy
    
    def _augment_policy_naive(self, policy: FloatArray1D) -> FloatArray1D:
        """Naively augment the given policy to minimize the expected norm violation cost.
        
        This method simply multiplies the policy by 1/(1 + cost) to bias probabilities towards
        norm-compliant actions. It does not enforce any constraint on the KL-divergence.
        """
        cost = self.get_norm_violation_cost()
        updated_policy = policy * (1.0 / (1.0 + cost))
        updated_policy /= np.sum(updated_policy)
        if self.verbose:
            print("Naively Augmented Policy:")
            for action, prob in enumerate(updated_policy):
                print(f"  {self.ACTIONS_ALL[action]}: {prob:.3f}")
        return updated_policy

    def decide_action(self, model: DQN, obs: Observation) -> Action:
        """Decide which action the agent should take based on the updated policy.
        
        If the supervisor is in FILTER_ONLY mode, the model policy is filtered on hard constraints
        and the highest probability action is returned. If the supervisor is in DEFAULT mode, the 
        model policy is filtered and augmented for norm compliance, and the highest probability
        action is returned.

        :param model: DQN model.
        :param obs: observation from the environment.
        :return: final action selection.
        """ 
        model_policy = self._get_model_policy(model, obs)
        policy_filtered = self._filter_policy(model_policy)
        if self.mode == SupervisorMode.FILTER_ONLY:
            return policy_filtered.argmax()
        if self.mode == SupervisorMode.NAIVE_AUGMENT:
            augmented_policy = self._augment_policy_naive(policy_filtered)
        else: # SupervisorMode.DEFAULT
            augmented_policy = self._augment_policy(policy_filtered)
        return augmented_policy.argmax()
        
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