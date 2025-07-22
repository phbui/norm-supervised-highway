from collections import defaultdict
from enum import Enum
from typing import Optional
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

from norm_supervisor.norms.profiles.abstract import AbstractNormProfile
from norm_supervisor.norms.profiles.cautious import CautiousDrivingProfile
from norm_supervisor.norms.profiles.efficient import EfficientDrivingProfile
from norm_supervisor.consts import ACTION_STRINGS

# Type alias for 1D array of floating points
FloatArray1D = npt.NDArray[np.float64]
IntArray1D = npt.NDArray[np.int32]

class PolicyAugmentMethod(Enum):
    """Enum for supervisor methods."""
    NOP        = 'nop'        # No KL-divergence method applied.
    NAIVE      = 'naive'      # Naive augment rule
    FIXED      = 'fixed'      # Use a fixed beta value
    ADAPTIVE   = 'adaptive'   # Adaptively calculate beta based on KL budget
    PROJECTION = 'projection' # Project policy onto the minimum cost action space

class Supervisor:
    """Supervisor class for enforcing metrics-driven norms in the HighwayEnv environment."""
    ACTIONS_ALL = DiscreteMetaAction.ACTIONS_ALL
    PROFILES: dict[str, AbstractNormProfile] = {
        'cautious': CautiousDrivingProfile,
        'efficient': EfficientDrivingProfile
    }

    def __init__(
        self,
        env: HighwayEnv,
        profile_name: str,
        filter: bool = True,
        method: str = PolicyAugmentMethod.ADAPTIVE.value,
        fixed_beta: Optional[float] = 1.0,
        kl_budget: Optional[float] = 0.01,
        eta_max: Optional[float] = 10.0,
        eta_min: Optional[float] = -10.0,
        tol: Optional[float] = 1e-4,
        verbose=False
    ) -> None:
        """Initialize the supervisor with the environment and configuration.

        :param env: the unwrapped HighwayEnv environment.
        :profile_name: the name of the norm profile to use for the supervisor.
        :filter: whether to filter the model policy on hard constraints.
        :param method: the method for policy augmentation ('fixed', 'adaptive', 'naive', 'nop').
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
        # Ensure proper enum types can be constructed from string parameters  
        try:
            self.profile_name = profile_name.lower()
            self.profile: AbstractNormProfile = self.PROFILES[self.profile_name]()
        except KeyError:
            raise ValueError(f"Invalid profile: {profile_name}. Expected one of "
                             f"{[p for p in self.PROFILES.keys()]}")
        try:
            self.method = PolicyAugmentMethod(method.lower())
        except ValueError:
            raise ValueError(f"Invalid method: {method}. Expected one of "
                             f"{[m.value for m in PolicyAugmentMethod]}")
        
        # Ensure internal consistency of specified parameters
        if self.method == PolicyAugmentMethod.FIXED and fixed_beta is None:
            raise ValueError("fixed_beta must be provided for the 'fixed' method!")
        
        if self.method == PolicyAugmentMethod.ADAPTIVE and any(
            param is None for param in [kl_budget, eta_max, eta_min, tol]
        ):
            raise ValueError("kl_budget, eta_max, eta_min, and tol must be provided for the "
                             "'adaptive' method!")
        
        # Check for value ranges
        if self.method == PolicyAugmentMethod.ADAPTIVE and kl_budget <= 0:
            raise ValueError("kl_budget must be greater than zero for the 'adaptive' method!")

        if self.method == PolicyAugmentMethod.ADAPTIVE and tol <= 0:
            raise ValueError("tol must be greater than zero for the 'adaptive' method!")
        
        if self.method == PolicyAugmentMethod.FIXED and fixed_beta <= 0:
            raise ValueError("fixed_beta must be greater than zero for the 'adaptive' method!") 

        self.env        = env
        self.filter     = filter
        self.fixed_beta = fixed_beta
        self.kl_budget  = kl_budget
        self.eta_max    = eta_max
        self.eta_min    = eta_min
        self.tol        = tol
        self.verbose    = verbose

        self.reset_norms()

    def reset_norms(self) -> None:
        """Reset the norms for the current environment.
        
        This method must be called every time the environment is reset to update the norm checkers
        with the new state of the environment.
        """
        self.norms      = self.profile.norms
        self.contraints = self.profile.constraints
        # Create a dictionary mapping norm names to their weights for easy access
        self.norm_weights = {str(norm): norm.weight for norm in self.norms}

    def count_constraint_violations(self, action: Action) -> dict[str, bool]:
        """Return a dictionary mapping constraints to violation boolean for the given action."""
        violations_dict = {str(c): False for c in self.contraints}
        for constraint in self.contraints:
            if constraint.is_violating_action(self.env.vehicle, action):
                violations_dict[str(constraint)] = True

        return violations_dict
    
    def get_constraint_violation_cost(self) -> IntArray1D:
        """Return the constraint violation cost vector for the current state."""
        constraint_violation_cost = np.zeros(len(self.ACTIONS_ALL))
        for action in self.ACTIONS_ALL.keys():
            violations_count = self.count_constraint_violations(action)
            constraint_violation_cost[action] = sum(violations_count.values())

        return constraint_violation_cost
    
    def is_permissible(self, action: Action) -> bool:
        """Checks if an action violates any hard constraints."""
        return not any(c.is_violating_action(self.env.vehicle, action)
                       for c in self.contraints)

    def count_norm_violations(self, action: Action) -> dict[str, bool]:
        """Return a dictionary mapping norms to violation counts for the given action."""
        violations_dict = {str(norm): 0 for norm in self.norms}
        for norm in self.norms:
            if norm.is_violating_action(self.env.vehicle, action):
                violations_dict[str(norm)] = True
        return  violations_dict

    def weight_norm_violations(self, violations_count: dict[str, bool]) -> dict[str, int]:
        """Return a dictionary of weighted norm costs given a dictionary of norm violations."""
        return {norm: violations_count[norm] * self.norm_weights[norm] for norm in violations_count}
    
    def get_norm_violation_cost(self) -> IntArray1D:
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
    
    def _filter_policy(self, policy: FloatArray1D) -> Optional[FloatArray1D]:
        """Filter the given policy to only include permissible actions.
        
        This method assigns zero probability to impermissible actions and then renormalizes the
        distribution.
        """
        constraint_violation_cost = self.get_constraint_violation_cost()
        permissible_mask = [cost == 0 for cost in constraint_violation_cost]
        if all(permissible_mask):
            if self.verbose:
                print("All actions are permissible! Returning original policy.")
            return policy

        # If all actions are impermissible, filter on the minimum constraint cost action set
        if not any(permissible_mask):
            cost_min = min(constraint_violation_cost)
            permissible_mask = [cost == cost_min for cost in constraint_violation_cost]
        
        policy_permissible = policy[permissible_mask]
        policy_permissible /= np.sum(policy_permissible)
        policy_filtered = np.full_like(policy, 0.0)
        policy_filtered[permissible_mask] = policy_permissible
        if self.verbose:
            print("Filtered Model Policy Probabilities:")
            for action, prob in enumerate(policy_filtered):
                print(f"  {self.ACTIONS_ALL[action]}: {prob:.3f}")

        return policy_filtered
    
    def _update_policy_support(
        self,
        log_policy_support: FloatArray1D,
        cost_support: FloatArray1D,
        eta: float
    ) -> FloatArray1D:
        """Update the policy support based on the hyperparameter eta.
        
        :param log_policy_support: precomputed policy support in log space.
        :param cost_support: norm violation costs over the policy support.
        :param eta: hyperparameter for the KL-divergence estimation, where eta = log(beta).
        :return: [policy * exp(cost / beta)] / Z (normalized).
        """
        logits = log_policy_support - cost_support / np.exp(eta)
        logits -= np.max(logits)
        updated_policy_support = np.exp(logits)
        updated_policy_support /= np.sum(updated_policy_support)
        return updated_policy_support
    
    def _augment_policy(self, policy: FloatArray1D) -> FloatArray1D:
        """Augment the given policy to minimizes the expected norm violation cost.

        The augmented policy is computed using either the 'fixed' or 'adaptive' method. When in
        'fixed' mode, the provided value for beta is used to compute the new policy. When in
        'adaptive' mode, the value for beta is computed to satisfy the KL-divergence constraint.
        """
        support_mask = ~np.isclose(policy, 0.0)
        policy_support = policy[support_mask]
        if not np.isclose(policy_support.sum(), 1.0):
            raise ValueError(f"Policy support do not sum to 1: {policy_support.sum():.4f}")

        # Cache cost vector for use in all computations
        cost = self.get_norm_violation_cost()
        cost_support = cost[support_mask]
        if self.verbose:
            print(f"Cost vector:")
            for action, cost in enumerate(cost):
                print(f"  {self.ACTIONS_ALL[action]}: {cost:.3f}")
        
        # Return the original policy if the cost function is uniform
        if np.allclose(cost_support, cost_support[0]):
            if self.verbose:
                print(f"All actions are equally norm compliant! No augmentation needed.")
            return policy

        # Compute greedy projection onto the minimum cost action space
        if self.method in [PolicyAugmentMethod.ADAPTIVE, PolicyAugmentMethod.PROJECTION]:
            min_cost_mask = np.isclose(cost_support, np.min(cost_support))
            pi_cost_support = np.zeros_like(policy_support)
            pi_cost_support[min_cost_mask] = 1.0 / np.sum(min_cost_mask)
            updated_policy_support = policy_support * min_cost_mask
            updated_policy_support /= np.sum(updated_policy_support)
            kl_value = entropy(updated_policy_support, policy_support, nan_policy='raise')
            # Alwaus return the greedy policy for the PROJECTION method or for the ADAPTIVE method
            # if the KL budget is satisfied.
            if (self.method == PolicyAugmentMethod.PROJECTION
                or (self.method == PolicyAugmentMethod.ADAPTIVE and kl_value <= self.kl_budget)):
                # Construct updated policy from new support
                updated_policy = np.full_like(policy, 0.0)
                updated_policy[support_mask] = updated_policy_support
                if self.verbose:
                    print(f"Updated policy KL-divergence: {entropy(updated_policy_support, policy_support)}")
                    print("Updated policy:")
                    for action, prob in enumerate(updated_policy):
                        print(f"  {self.ACTIONS_ALL[action]}: {prob:.3f}")
                return updated_policy

        # Cache log policy support for use in subsequent computations
        log_policy_support = np.log(policy_support)

        # Normalize cost support for numerical stability
        # NOTE: This is especially important for the fixed beta method
        normalized_cost_support = cost_support / np.sum(cost_support)

        # Used fixed beta provided by the user
        if self.method == PolicyAugmentMethod.FIXED:
            eta_star = np.log(self.fixed_beta)
        # Or solve for eta=log(beta) using Brent's root-finding algorithm
        elif self.method == PolicyAugmentMethod.ADAPTIVE:
            def kl_gap(eta: float) -> float:
                """Return the difference between the KL-divergence induced by eta and the budget."""
                updated_policy_support \
                    = self._update_policy_support(log_policy_support, normalized_cost_support, eta)
                kl_value = entropy(updated_policy_support, policy_support, nan_policy='raise')
                return kl_value - self.kl_budget

            # Brent's method requires the function to be bracketed over the provided interval
            kl_gap_high = kl_gap(self.eta_min)
            kl_gap_low  = kl_gap(self.eta_max)
            if kl_gap_high * kl_gap_low > 0:
                raise ValueError(f"KL constraint not bracketed: [({self.eta_min}, {kl_gap_high:.4f}), "
                                f"({self.eta_max}, {kl_gap_low:.4f})]")
            
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

        # Update policy support
        updated_policy_support \
            = self._update_policy_support(log_policy_support, normalized_cost_support, eta_star)
        if self.verbose:
            print(f"Best Estimate: {eta_star:.3f}")

        # Construct updated policy from new support
        updated_policy = np.full_like(policy, 0.0)
        updated_policy[support_mask] = updated_policy_support
        if self.verbose:
            print(f"Updated policy KL-divergence: {entropy(updated_policy_support, policy_support)}")
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
        cost_sum = np.nansum(cost)   # ignore NaNs in cost if any
        if cost_sum == 0 or not np.isfinite(cost_sum):
            # fallback: e.g., uniform distribution or leave as zeros
            normalized_cost = np.zeros_like(cost)
        else:
            normalized_cost = cost / cost_sum
        updated_policy = policy * (1.0 / (1.0 + normalized_cost))
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
        policy = self._get_model_policy(model, obs)

        # Conditionally filter hard constraints from the model policy
        if self.filter:
            policy = self._filter_policy(policy)
        
        # Augment the policy based on the supervisor mode
        if self.method == PolicyAugmentMethod.NOP:
            augmented_policy = policy
        elif self.method == PolicyAugmentMethod.NAIVE:
            augmented_policy = self._augment_policy_naive(policy)
        elif self.method in [
            PolicyAugmentMethod.ADAPTIVE,
            PolicyAugmentMethod.FIXED,
            PolicyAugmentMethod.PROJECTION
        ]:
            augmented_policy = self._augment_policy(policy)
        else:
            raise ValueError(f"Unknown supervisor method: {self.method}")
        
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