#!/usr/bin/env python3
import argparse
import csv
import json
import numpy as np
import os
import sys
from collections import defaultdict

import gymnasium
from stable_baselines3 import DQN
from highway_env.envs.highway_env import HighwayEnv

from norm_supervisor.supervisor import Supervisor, SupervisorMode, SupervisorMethod
import norm_supervisor.norms.norms as norms
import norm_supervisor.metrics as metrics

# Configuration mappings
CONFIGS = {
    '2L5V': {
        'model_file': 'trained_30_duration_2_lanes_5_vehicles.zip',
        'env_config': 'train_30_duration_2_lanes_5_vehicles.json',
        'lanes': 2
    },
    '4L20V': {
        'model_file': 'trained_30_duration_4_lanes_20_vehicles.zip',
        'env_config': 'train_30_duration_4_lanes_20_vehicles.json',
        'lanes': 4
    }
}

MODE_MAPPING = {
    'unsupervised': SupervisorMode.NOP,
    'filter_only': SupervisorMode.FILTER_ONLY,
    'naive_augment': SupervisorMode.NAIVE_AUGMENT,
    'default': SupervisorMode.DEFAULT
}

METHOD_MAPPING = {
    'adaptive': SupervisorMethod.ADAPTIVE,
    'fixed': SupervisorMethod.FIXED,
    'nop': SupervisorMethod.NOP
}

BASE_SEED = 239


class ExperimentConfig:
    """Configuration for experiment parameters."""
    
    def __init__(self, args: argparse.Namespace):
        self.profile = args.profile
        self.mode = args.mode
        self.method = args.method
        self.value = args.value
        self.model_config = args.model
        self.env_config = args.env
        self.num_experiments = args.experiments
        self.num_episodes = args.episodes
        self.output_file = args.output
        
        # Validate configuration
        self._validate()
    
    def _validate(self):
        """Validate the experiment configuration."""
        if self.mode == 'default':
            if not self.method:
                raise ValueError("--method is required when --mode is 'default'")
            if self.method in ['adaptive', 'fixed'] and not self.value:
                raise ValueError("--value is required for adaptive/fixed methods")
        
        # Check if model file exists
        model_path = os.path.join("models", CONFIGS[self.model_config]['model_file'])
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found: {model_path}")
        
        # Check if environment config exists
        env_config_path = os.path.join("configs/environment", CONFIGS[self.env_config]['env_config'])
        if not os.path.exists(env_config_path):
            raise ValueError(f"Environment config not found: {env_config_path}")


class EpisodeMetrics:
    """Collects metrics for a single episode."""
    
    def __init__(self, lane_count: int):
        self.episode_length = 0
        self.collision = False
        self.ttc_history = []
        self.following_distances = []
        self.speeds = []

        self.lane_times = {f'lane_{i}': 0 for i in range(lane_count)}
        
        self.norm_violations            = defaultdict(int)
        self.constraint_violations      = defaultdict(int)
        self.norm_violation_rates       = defaultdict(float) # Per time step violation rate
        self.constraint_violation_rates = defaultdict(float) # Per time step violation rate
        
        self.cost              = 0.0 # Weighted norm violation cost
        self.avoided_cost      = 0.0 # Avoided weighted norm violation cost
        self.cost_rate         = 0.0 # Weighted norm violation cost per timestep
        self.avoided_cost_rate = 0.0 # Avoided weighted norm violation cost per timestep
    
    def add_timestep(self, ttc: float, following_distance: float, speed: float, lane_index: int,
                     norm_violations: dict[str, bool], constraint_violations: dict[str, bool],
                     cost: float, avoided_cost: float):
        """Add data from a single timestep."""
        self.episode_length += 1
        self.ttc_history.append(ttc)
        self.following_distances.append(following_distance)
        self.speeds.append(speed)
        self.lane_times[f'lane_{lane_index}'] += 1
        
        for norm, flag in norm_violations.items():
            self.norm_violations[norm] += flag
        
        for constraint, flag in constraint_violations.items():
            self.constraint_violations[constraint] += flag
        
        self.cost += cost
        self.avoided_cost += avoided_cost
    
    def _safe_nanmean(self, data):
        """Calculate nanmean safely, avoiding empty slice warnings."""
        if not data or all(np.isnan(x) for x in data):
            return np.nan
        return np.nanmean(data)
    
    def finalize(self):
        """Calculate final metrics for the episode."""
        self.mean_ttc = self._safe_nanmean(self.ttc_history)
        self.mean_following_distance = self._safe_nanmean(self.following_distances)
        self.mean_speed = self._safe_nanmean(self.speeds)
        
        # Normalize rates by episode length
        if self.episode_length > 0:
            self.cost_rate = self.cost / self.episode_length
            self.avoided_cost_rate = self.avoided_cost / self.episode_length
            
            for lane, time in self.lane_times.items():
                self.lane_times[lane] = time / self.episode_length

            for norm in self.norm_violations:
                self.norm_violation_rates[norm] = self.norm_violations[norm] / self.episode_length
            
            for constraint in self.constraint_violations:
                self.constraint_violation_rates[constraint] \
                    = self.constraint_violations[constraint] / self.episode_length


class ExperimentResults:
    """Aggregates results across episodes for a single experiment."""
    
    def __init__(self, experiment_id: int, config: ExperimentConfig):
        self.experiment_id = experiment_id
        self.config = config
        self.episode_metrics = []
    
    def add_episode(self, metrics: EpisodeMetrics):
        """Add metrics from a completed episode."""
        self.episode_metrics.append(metrics)
    
    def get_experiment_means(self) -> dict[str, float]:
        """Calculate mean values across all episodes in this experiment."""
        if not self.episode_metrics:
            return {}
        
        # Calculate means
        episode_lengths = [ep.episode_length for ep in self.episode_metrics]
        collision_count = sum(1 for ep in self.episode_metrics if ep.collision)
        mean_ttcs = [ep.mean_ttc for ep in self.episode_metrics if not np.isnan(ep.mean_ttc)]
        mean_following_distances = [ep.mean_following_distance for ep in self.episode_metrics
                                    if not np.isnan(ep.mean_following_distance)]
        mean_speeds = [ep.mean_speed for ep in self.episode_metrics if not np.isnan(ep.mean_speed)]

        # Aggregate lane times
        lane_count = CONFIGS[self.config.env_config]['lanes']
        all_lane_times = defaultdict(list)
        for ep in self.episode_metrics:
            for lane, time in ep.lane_times.items():
                all_lane_times[lane].append(time)
        
        # Aggregate norm violations
        all_norm_violation_rates = defaultdict(list)
        for ep in self.episode_metrics:
            for norm, count in ep.norm_violation_rates.items():
                all_norm_violation_rates[norm].append(count)

        # Aggregate constraint violations
        all_constraint_violation_rates = defaultdict(list)
        for ep in self.episode_metrics:
            for constraint, count in ep.constraint_violation_rates.items():
                all_constraint_violation_rates[constraint].append(count)
        
        # Aggregate constraint violations
        cost_rates = [ep.cost_rate for ep in self.episode_metrics]
        avoided_cost_rates = [ep.avoided_cost_rate for ep in self.episode_metrics]
        
        # Build base results
        results = {
            'experiment_id': self.experiment_id,
            'profile': self.config.profile,
            'mode': self.config.mode,
            'method': self.config.method or '',
            'value': float(self.config.value) if self.config.value is not None else 0.0,
            'num_episodes': len(self.episode_metrics),
            'episode_length_mean': np.mean(episode_lengths),
            'collision_count': collision_count,
            'mean_ttc': np.nanmean(mean_ttcs) if mean_ttcs else np.nan,
            'mean_following_distance': np.nanmean(mean_following_distances) if mean_following_distances else np.nan,
            'mean_speed': np.nanmean(mean_speeds) if mean_speeds else np.nan,
            'speed_violations_rate_mean': np.mean(all_norm_violation_rates.get('Speeding', [0])),
            'tailgating_violations_rate_mean': np.mean(all_norm_violation_rates.get('Tailgating', [0])),
            'braking_violations_rate_mean': np.mean(all_norm_violation_rates.get('Braking', [0])),
            'lane_change_tailgating_violations_rate_mean': np.mean(all_norm_violation_rates.get('LaneChangeTailgating', [0])),
            'lane_change_braking_violations_rate_mean': np.mean(all_norm_violation_rates.get('LaneChangeBraking', [0])),
            'collision_violations_rate_mean': np.mean(all_constraint_violation_rates.get('Collision', [0])),
            'lane_change_collision_violations_rate_mean': np.mean(all_constraint_violation_rates.get('LaneChangeCollision', [0])),
            'cost_rate_mean': np.mean(cost_rates),
            'avoided_cost_rate_mean': np.mean(avoided_cost_rates),
        }
        
        # Add lane times based on actual lane count
        for i in range(lane_count):
            results[f'lane_{i}_time_mean'] = np.mean(all_lane_times.get(f'lane_{i}', [0]))
        
        return results


class CSVWriter:
    """Handles CSV file writing with incremental updates."""
    
    def __init__(self, output_file: str, lane_count: int):
        self.output_file = output_file
        self.lane_count = lane_count
        self.fieldnames = self._get_fieldnames()
        self._create_file()
    
    def _get_fieldnames(self) -> list[str]:
        """Get CSV field names based on lane count."""
        base_fields = [
            'experiment_id', 'profile', 'mode', 'method', 'value',
            'num_episodes', 'episode_length_mean', 'collision_count', 'mean_ttc',
            'mean_following_distance', 'mean_speed'
        ]
        
        # Add lane fields
        lane_fields = [f'lane_{i}_time_mean' for i in range(self.lane_count)]
        
        # Add violation fields
        violation_fields = [
            'speed_violations_rate_mean', 'tailgating_violations_rate_mean',
            'braking_violations_rate_mean', 'lane_change_tailgating_violations_rate_mean',
            'lane_change_braking_violations_rate_mean', 'collision_violations_rate_mean',
            'lane_change_collision_violations_rate_mean', 'cost_rate_mean',
            'avoided_cost_rate_mean'
        ]
        
        return base_fields + lane_fields + violation_fields
    
    def _create_file(self):
        """Create the CSV file with headers."""
        try:
            with open(self.output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
        except Exception as e:
            raise RuntimeError(f"Failed to create output file {self.output_file}: {e}")
    
    def write_experiment(self, results: dict[str, float]):
        """Write a single experiment's results to the CSV file."""
        try:
            with open(self.output_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(results)
        except Exception as e:
            raise RuntimeError(f"Failed to write to output file {self.output_file}: {e}")


class ExperimentRunner:
    """Main experiment runner class."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.lane_count = CONFIGS[config.env_config]['lanes']
        self.csv_writer = CSVWriter(config.output_file, self.lane_count)
        
        # Load model and environment config
        self.model = self._load_model()
        self.env_config = self._load_env_config()
    
    def _load_model(self) -> DQN:
        """Load the DQN model."""
        model_path = os.path.join("models", CONFIGS[self.config.model_config]['model_file'])
        print(f"Loading model from {model_path}...")
        model = DQN.load(model_path)
        model.set_random_seed(BASE_SEED)
        return model
    
    def _load_env_config(self) -> dict:
        """Load environment configuration."""
        env_config_path = os.path.join("configs/environment", CONFIGS[self.config.env_config]['env_config'])
        print(f"Loading environment config from {env_config_path}...")
        with open(env_config_path, 'r') as f:
            return json.load(f)
    
    def _create_supervisor(self, env: HighwayEnv) -> Supervisor:
        """Create supervisor with appropriate configuration."""
        if self.config.mode == 'default':
            supervisor_mode = MODE_MAPPING[self.config.mode].value
            supervisor_method = METHOD_MAPPING[self.config.method].value
            fixed_beta = self.config.value if self.config.method == 'fixed' else None
            kl_budget = self.config.value if self.config.method == 'adaptive' else None
        else:
            supervisor_mode = MODE_MAPPING[self.config.mode].value
            supervisor_method = SupervisorMethod.NOP.value
            fixed_beta = None
            kl_budget = None
        
        return Supervisor(
            env=env,
            profile_name=self.config.profile,
            mode=supervisor_mode,
            method=supervisor_method,
            fixed_beta=fixed_beta,
            kl_budget=kl_budget,
            verbose=False
        )
    
    def run_experiment(self, experiment_id: int) -> ExperimentResults:
        """Run a single experiment."""
        print(f"\nStarting experiment {experiment_id + 1}/{self.config.num_experiments}")
        
        # Create environment
        env = gymnasium.make("highway-fast-v0", render_mode="rgb_array", config=self.env_config)
        env_unwrapped: HighwayEnv = env.unwrapped
        
        # Create supervisor
        supervisor = self._create_supervisor(env_unwrapped)
        
        # Create results container
        results = ExperimentResults(experiment_id, self.config)
        
        # Run episodes
        for episode in range(self.config.num_episodes):
            episode_seed = BASE_SEED * (10 ** len(str(abs(self.config.num_experiments)))) + experiment_id
            episode_seed = episode_seed * (10 ** len(str(abs(self.config.num_episodes)))) + episode
            
            obs, _ = env.reset(seed=episode_seed)
            supervisor.reset_norms()
            
            # Create metrics collector for this episode
            episode_metrics = EpisodeMetrics(self.lane_count)
            
            # Create tailgating norm for following distance calculation
            tailgating_norm = norms.TailgatingNorm(weight=None, safe_distance=None)
            
            done = truncated = False
            while not (done or truncated):
                # Get model action
                action, _ = self.model.predict(obs, deterministic=True)
                action = action.item()
                
                # Count norm violations for original action
                violations_dict = supervisor.count_norm_violations(action)
                violations_weight_dict = supervisor.weight_norm_violations(violations_dict)
                violations_weight = sum(violations_weight_dict.values())
                
                # Count constraint violations for original action
                constraint_violations_dict = supervisor.count_constraint_violations(action)
                
                # Apply supervisor if needed
                if self.config.mode != 'unsupervised':
                    new_action = supervisor.decide_action(self.model, obs)
                    new_violations_dict = supervisor.count_norm_violations(new_action)
                    new_violations_weight_dict = supervisor.weight_norm_violations(new_violations_dict)
                    new_violations_weight = sum(new_violations_weight_dict.values())

                    new_constraint_violations_dict = supervisor.count_constraint_violations(new_action)
                    
                    avoided_violations_weight = violations_weight - new_violations_weight
                    action = new_action

                    # Set measures to the values for the new action
                    violations_dict = new_violations_dict
                    violations_weight_dict = new_violations_weight_dict
                    violations_weight = new_violations_weight
                    constraint_violations_dict = new_constraint_violations_dict
                else:
                    avoided_violations_weight = 0.0
                
                # Calculate metrics
                ttcs = metrics.calculate_neighbour_ttcs(env_unwrapped.vehicle)
                ttc = ttcs[0] if ttcs[0] != np.inf else np.nan
                following_distance = tailgating_norm.evaluate_criterion(env_unwrapped.vehicle)
                following_distance = following_distance if following_distance != np.inf else np.nan
                speed = env_unwrapped.vehicle.speed
                _, _, lane_index = env_unwrapped.vehicle.lane_index
                
                # Add timestep data
                episode_metrics.add_timestep(
                    ttc=ttc,
                    following_distance=following_distance,
                    speed=speed,
                    lane_index=lane_index,
                    norm_violations=violations_dict,
                    constraint_violations=constraint_violations_dict,
                    cost=violations_weight,
                    avoided_cost=avoided_violations_weight
                )
                
                # Take action
                obs, _, done, truncated, info = env.step(action)
                
                # Check for collision
                if done or truncated:
                    if info.get("crashed", False):
                        episode_metrics.collision = True
            
            # Finalize episode metrics
            episode_metrics.finalize()
            results.add_episode(episode_metrics)
            
            # Print progress every 10 episodes
            if (episode + 1) % 10 == 0:
                print(f"  Completed episode {episode + 1}/{self.config.num_episodes}")
        
        env.close()
        return results
    
    def run_all_experiments(self):
        """Run all experiments and write results incrementally."""
        print(f"Starting experimental run:")
        print(f"  Profile: {self.config.profile}")
        print(f"  Mode: {self.config.mode}")
        print(f"  Method: {self.config.method or 'N/A'}")
        print(f"  Value: {self.config.value or 'N/A'}")
        print(f"  Model: {self.config.model_config}")
        print(f"  Environment: {self.config.env_config}")
        print(f"  Experiments: {self.config.num_experiments}")
        print(f"  Episodes per experiment: {self.config.num_episodes}")
        print(f"  Output: {self.config.output_file}")
        
        for experiment_id in range(self.config.num_experiments):
            try:
                results = self.run_experiment(experiment_id)
                experiment_means = results.get_experiment_means()
                self.csv_writer.write_experiment(experiment_means)
                
                print(f"Experiment {experiment_id + 1} completed:")
                print(f"  Collisions: {experiment_means['collision_count']}")
                print(f"  Mean TTC: {experiment_means['mean_ttc']:.3f}")
                print(f"  Mean speed: {experiment_means['mean_speed']:.2f} m/s")
                
            except Exception as e:
                import traceback
                print(f"Error in experiment {experiment_id + 1}: {e}")
                print(f"Full traceback:")
                traceback.print_exc()
                raise
        
        print(f"\nAll experiments completed. Results written to {self.config.output_file}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run norm-supervised highway driving experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--profile', choices=['cautious', 'efficient'], required=True,
                       help='Driving profile to use')
    parser.add_argument('--mode', choices=['unsupervised', 'filter_only', 'naive_augment', 'default'], 
                       required=True, help='Supervisor mode')
    parser.add_argument('--method', choices=['adaptive', 'fixed'], 
                       help='Supervisor method (required for default mode)')
    parser.add_argument('--value', type=float, 
                       help='Value for adaptive/fixed methods (required for default mode)')
    parser.add_argument('--model', choices=['2L5V', '4L20V'], required=True,
                       help='Model configuration')
    parser.add_argument('--env', choices=['2L5V', '4L20V'], required=True,
                       help='Environment configuration')
    parser.add_argument('--experiments', type=int, default=5,
                       help='Number of experiments to run')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of episodes per experiment')
    parser.add_argument('--output', required=True,
                       help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Validate method and value for default mode
    if args.mode == 'default':
        if not args.method:
            parser.error("--method is required when --mode is 'default'")
        if args.method in ['adaptive', 'fixed'] and not args.value:
            parser.error("--value is required for adaptive/fixed methods")
    
    return args


def main():
    """Main function."""
    try:
        args = parse_arguments()
        config = ExperimentConfig(args)
        runner = ExperimentRunner(config)
        runner.run_all_experiments()
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(f"Full traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
