#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time

import gymnasium
import torch
from stable_baselines3 import DQN
from highway_env.envs.highway_env import HighwayEnv

from norm_supervisor.norms.constraints import SafetyEnvelopeConstraint
from norm_supervisor.supervisor import Supervisor, PolicyAugmentMethod
import norm_supervisor.metrics as metrics

# Add the scripts directory to the path so we can import from test.py
sys.path.append(os.path.dirname(__file__))
from test import CONFIGS, METHOD_MAPPING

BASE_SEED = 239

def debug_collision(profile, method, value, filter, episode_seed=0):
    """Debug a single episode with visualization."""
    
    # Load model and environment config
    model_path = os.path.join("models", CONFIGS['default']['model_file'])
    env_config_path = os.path.join("configs/environment", CONFIGS['default']['env_config'])
    
    print(f"Loading model from {model_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DQN.load(model_path, device=device)
    model.set_random_seed(BASE_SEED)
    
    print(f"Loading environment config from {env_config_path}...")
    with open(env_config_path, 'r') as f:
        env_config_dict = json.load(f)
    
    # Override rendering properties for debugging
    env_config_dict.update({
        "render_agent": True,
        "show_trajectories": True,
        "offscreen_rendering": False
    })
    
    # Create environment with human rendering
    env = gymnasium.make("highway-v0", render_mode="human", config=env_config_dict)
    env_unwrapped: HighwayEnv = env.unwrapped
    
    # Create supervisor
    supervisor_method = METHOD_MAPPING[method].value
    fixed_beta = value if method == 'fixed' else None
    kl_budget = value if method == 'adaptive' else None
    
    supervisor = Supervisor(
        env=env_unwrapped,
        profile_name=profile,
        filter=filter,
        method=supervisor_method,
        fixed_beta=fixed_beta,
        kl_budget=kl_budget,
        verbose=True  # Enable verbose output
    )
    
    # Reset environment
    obs, _ = env.reset(seed=episode_seed)
    supervisor.reset_norms()
    
    print(f"Starting debug episode with seed {episode_seed}")
    print(f"Profile: {profile}, Method: {method}")
    
    step = 0
    done = truncated = False
    collision_occurred = False
    
    while not (done or truncated):
        print(f"\n--- Step {step} ---")
        
        # Get model action
        action, _ = model.predict(obs, deterministic=True)
        action = action.item()
        print(f"Model action: {Supervisor.ACTIONS_ALL[action]}")
        
        # Check constraint violations for original action
        constraint_violations = supervisor.count_constraint_violations(action)
        print(f"Constraint violations for original action: {constraint_violations}")
        
        # Apply supervisor if needed
        if method != 'nop' or filter:
            new_action = supervisor.decide_action(model, obs)
            print(f"Supervisor action: {Supervisor.ACTIONS_ALL[new_action]}")
            
            new_constraint_violations = supervisor.count_constraint_violations(new_action)
            print(f"Constraint violations for supervisor action: {new_constraint_violations}")
            
            action = new_action
        else:
            print("No supervisor applied")
        
        # Calculate TTC for debugging
        ttcs = metrics.calculate_neighbour_ttcs(env_unwrapped.vehicle)
        print(f"TTC front: {ttcs[0]:.3f}, TTC rear: {ttcs[1]:.3f}")
        distance = SafetyEnvelopeConstraint.evaluate_criterion(env_unwrapped.vehicle)
        print(f"Following distance: {distance:.3f}")
        
        # Take action
        obs, _, done, truncated, info = env.step(action)
        
        # Check for collision
        if done or truncated:
            if info.get("crashed", False):
                print(f"*** CRASHED AFTER ACTION: {Supervisor.ACTIONS_ALL[action]} ***")
                collision_occurred = True
            else:
                print("Episode ended without crash")
        
        # Slow down for visualization
        time.sleep(0.5)  # 0.5 second delay between steps
        
        step += 1
    
    print(f"Episode finished after {step} steps")
    
    # Replay the last 5 frames visually (no console output) only if collision occurred
    if collision_occurred:
        print(f"\nReplaying last 5 frames visually...")
        
        # Recreate environment for replay
        obs_replay, _ = env.reset(seed=episode_seed)
        supervisor.reset_norms()
        supervisor.verbose = False # Disable verbosity during replay
        replay_step = 0
        for replay_step in range(step):
            action_replay, _ = model.predict(obs_replay, deterministic=True)
            if method != 'nop' or filter:
                action_replay = supervisor.decide_action(model, obs)                
            obs_replay, _, done_replay, truncated_replay, _ = env.step(action_replay.item())
            if done_replay or truncated_replay:
                break
            if replay_step >= (step - 10):
                time.sleep(2.0)
        
    env.close()

def main():
    parser = argparse.ArgumentParser(description="Debug collision with visualization")
    parser.add_argument('--profile', choices=['cautious', 'efficient'], required=True,
                       help='Driving profile to use')
    parser.add_argument('--method', choices=['nop', 'naive', 'adaptive', 'fixed'], 
                       required=True, help='Supervisor method')
    parser.add_argument('--value', type=float, 
                       help='Value for adaptive/fixed methods (required for adaptive/fixed methods)')
    parser.add_argument('--filter', dest='filter', action='store_true',
                       help='Enable supervisor filtering (default: True)')
    parser.add_argument('--no-filter', dest='filter', action='store_false',
                       help='Disable supervisor filtering')
    parser.set_defaults(filter=True)
    parser.add_argument('--episode_seed', type=int, default=0,
                       help='Seed for the episode')
    
    args = parser.parse_args()
    
    # Validate method and value for adaptive/fixed methods
    if args.method in ['adaptive', 'fixed'] and not args.value:
        parser.error("--value is required for adaptive/fixed methods")
    
    debug_collision(
        profile=args.profile,
        method=args.method,
        value=args.value,
        filter=args.filter,
        episode_seed=args.episode_seed
    )

if __name__ == "__main__":
    main() 