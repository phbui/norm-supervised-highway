import os
import json
import gymnasium
import highway_env
from stable_baselines3 import DQN
import numpy as np
from supervisor import Supervisor
import pygame
import time
import random
from gymnasium.envs.registration import register
from highway_env.envs.highway_env import HighwayEnv
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.controller import MDPVehicle
from collections import defaultdict
from supervisor import Supervisor

BASE_SEED = 239

def list_files(directory, extension=".json"):
    return sorted([f for f in os.listdir(directory) if f.endswith(extension)])

def list_models(models_dir="models"):
    return list_files(models_dir, ".zip")

def get_output_filename(default="results.txt"):
    fname = input(f"Enter output filename [{default}]: ").strip() + '.txt'
    return fname if fname else default

def count_by_presence(a_dict, b_dict):
    for norm, delta in b_dict.items():
        a_dict[norm] = a_dict.get(norm, 0) + delta
    return a_dict

def average_by_presence(dicts):
    """Average value per key in dict list."""
    sums   = defaultdict(float)
    counts = defaultdict(int)

    for d in dicts:
        for k, v in d.items():
            sums[k]   += v
            counts[k] += 1

    return {k: sums[k] / counts[k] for k in sums}

class LaneKeepingVehicle(ControlledVehicle):
    """A vehicle that simply keeps its lane and speed without IDM."""
    
    def __init__(self, road, position, heading=0, speed=0):
        super().__init__(road, position, heading, speed)
        self.fixed_speed = speed  # Store the desired fixed speed
        
    def act(self, action=None):
        """Just maintain lane and speed."""
        action = {'steering': 0, 'acceleration': self.maintain_speed()}
        super().act(action)  # Actually apply the action
        
    def maintain_speed(self):
        """Calculate acceleration to maintain target speed."""
        # Simple proportional controller to maintain speed
        return 0.1 * (self.fixed_speed - self.speed)
        
    @classmethod
    def create_from(cls, vehicle):
        """Create a new vehicle of this class from an existing vehicle."""
        v = cls(vehicle.road, vehicle.position, vehicle.heading, vehicle.speed)
        return v


class CustomHighwayEnv(HighwayEnv):
    """
    A customized highway driving environment with controlled starting positions.
    """
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction"
            },
            "lanes_count": 4,
            "vehicles_count": 20,
            "duration": 200,
            "initial_spacing": 2,
            "collision_reward": -1,
            "reward_speed_range": [20, 30],
            "simulation_frequency": 15,
            "policy_frequency": 1,
            "other_vehicles_type": "LaneKeepingVehicle",
            "screen_width": 600,
            "screen_height": 150,
            "centering_position": [0.3, 0.5],
            "scaling": 5.5,
            "show_trajectories": True,
            "render_agent": False,
            
            # Road and Environment
            "initial_lane_id": 1,  # Lane 1 (0-indexed)
            "initial_position": 30,  # Adjusted to position ego vehicle in traffic
            
            "frequency": 50,
            
            # Vehicle Configuration
            "vehicles_count": 24,  # More vehicles for chessboard pattern
            "controlled_vehicles": 1,
            "ego_spacing": 2,
            "vehicles_density": 1.5,
            
            # Custom Vehicle Placement
            "custom_vehicle_placement": True,
            "vehicle_positions": [
                # Ego vehicle will be in lane 1 (position specified elsewhere in config)
                
                # Slow vehicle directly in front of ego in lane 1 (middle lane)

                
                # Fast vehicle approaching from behind in lane 0 (left lane)
                {"lane": 1, "position": 75, "speed": 22, "vehicle_type": "slow_car"},
                
                # Vehicle ahead in lane 0 but with gap that looks inviting
                {"lane": 0, "position": 10, "speed": 25, "vehicle_type": "car"},
                
                # Vehicles in lane 2 (right lane) creating a similar situation
                {"lane": 2, "position": 10, "speed": 25, "vehicle_type": "car"},

                
                # Extra vehicles to create more complex traffic pattern
            ],
            
            # Other Parameters
            "offroad_terminal": True
        })
        return config

    def _create_vehicles(self) -> None:
        """Create vehicles with a chess board pattern to force lane switching."""
        # Create the ego vehicle (controlled vehicle)
        ego_vehicle = Vehicle.make_on_lane(
            self.road,
            lane_index=("0", "1", self.config["initial_lane_id"]),
            longitudinal=self.config["initial_position"],
            speed=25  # Initial speed of ego vehicle
        )
        
        # Convert to controlled vehicle type
        ego_vehicle = self.action_type.vehicle_class(
            self.road, ego_vehicle.position, ego_vehicle.heading, ego_vehicle.speed
        )
        
        self.controlled_vehicles = [ego_vehicle]
        self.road.vehicles.append(ego_vehicle)

        if self.config.get("custom_vehicle_placement", False):
            for vehicle_data in self.config["vehicle_positions"]:
                lane = vehicle_data["lane"]
                position = vehicle_data["position"]
                speed = vehicle_data["speed"]
                vehicle_type = vehicle_data.get("vehicle_type", "car")  # Default to car
                
                # Create basic vehicle
                lane_index = ("0", "1", lane)
                
                # Create a LaneKeepingVehicle instead of a standard vehicle
                vehicle = LaneKeepingVehicle(
                    self.road,
                    position=self.road.network.get_lane(lane_index).position(position, 0),
                    heading=self.road.network.get_lane(lane_index).heading_at(position),
                    speed=speed
                )
                vehicle.fixed_speed = speed  # Set the fixed speed
                vehicle.target_lane_index = lane_index  # Set target lane
                
                # Set vehicle type and appearance
                if vehicle_type == "slow_car":
                    # Slow car - might be a bit wider to make passing harder
                    vehicle.LENGTH = 5.0
                    vehicle.WIDTH = 2.2
                    vehicle.fixed_speed = speed  # Already slow from config
                elif vehicle_type == "fast_car":
                    # Fast car - normal size but higher speed
                    vehicle.LENGTH = 4.5
                    vehicle.WIDTH = 2.0
                    vehicle.fixed_speed = speed  # Already fast from config
                elif vehicle_type == "truck":
                    vehicle.LENGTH = 7.5  # Longer vehicle
                    vehicle.WIDTH = 2.5
                    vehicle.fixed_speed = min(speed, 25)  # Trucks are slower
                elif vehicle_type == "bus":
                    vehicle.LENGTH = 10.0  # Very long vehicle
                    vehicle.WIDTH = 2.8   # Wider
                    vehicle.fixed_speed = min(speed, 18)  # Buses are slower
                
                self.road.vehicles.append(vehicle)

# Register the custom environment
register(
    id='custom-highway-v0',
    entry_point=CustomHighwayEnv,
)

pygame.init()

def main():
    # Select model
    model_files = list_models()
    if not model_files:
        print("No trained models found in 'models/' directory.")
        return

    print("\nAvailable models:")
    for i, f in enumerate(model_files):
        print(f"[{i}] {f}")
    # model_index = 1 # uncomment to harcode model, comment next line
    model_index = int(input("Select a model to run by number: ")) 
    model_path = os.path.join("models", model_files[model_index])
    output_file = get_output_filename()

    print(f"\nLoading model from {model_path}...")
    model = DQN.load(model_path)
    model.set_random_seed(BASE_SEED)

    num_experiments = 5
    num_episodes = 100

    results = {"WITH SUPERVISOR": [], "WITHOUT SUPERVISOR": []}

    for mode in results.keys():
        all_collisions = []
        all_violations = []
        all_avoided_violations = []
        all_violations_dict = []
        all_avoided_violations_dict = []

        for experiment in range(num_experiments):
            experiment_seed = BASE_SEED * (10 ** len(str(abs(num_experiments)))) + experiment
            print(f"\nExperiment {experiment + 1}/{num_experiments} ({mode}).")
            num_collision = 0
            num_violations = 0
            num_avoided_violations = 0
            env = gymnasium.make("custom-highway-v0", render_mode="rgb_array")
            supervisor = Supervisor(env.unwrapped, CustomHighwayEnv.default_config(), verbose=False) 
            ep_violations_dict = {str(norm): 0 for norm in supervisor.norms}
            ep_avoided_violations_dict = {str(norm): 0 for norm in supervisor.norms}

            for episode in range(num_episodes):
                print(f"\nExperiment {experiment +1}/{num_experiments} ({mode}). Episode {episode + 1}/{num_episodes}, Collision: {num_collision}, Unavoided Violatons: {str(ep_violations_dict)}, Number of Violations: {num_violations}, Avoided Violations: {str(ep_avoided_violations_dict)}, Number of Avoided Violations: {num_avoided_violations}")
                done = truncated = False
                episode_seed = experiment_seed * (10 ** len(str(abs(num_episodes)))) + episode
                obs, _ = env.reset(seed=episode_seed) # <- seeded
                supervisor.reset_norms()
                while not (done or truncated):
                    action, _  = model.predict(obs, deterministic=True)
                    violations, violations_dict = supervisor.count_action_norm_violations(action)

                    if mode == "WITH SUPERVISOR":
                        # Select new action and compute number of avoided violations
                        new_action, new_violations, new_violations_dict = supervisor.decide_action(model, obs)
                        avoided                 = violations - new_violations
                        avoided_violations_dict = {norm: violations_dict.get(norm, 0) - new_violations_dict.get(norm, 0) for norm in set(violations_dict) | set(new_violations_dict)}
                        action                  = new_action
                        violations              = new_violations
                        violations_dict         = new_violations_dict
                    else:
                        avoided = 0

                    num_violations         += violations
                    num_avoided_violations += avoided
                    count_by_presence(ep_violations_dict, violations_dict)
                    count_by_presence(ep_avoided_violations_dict, avoided_violations_dict)

                    obs, reward, done, truncated, info = env.step(action)

                    if done or truncated:
                        if info["crashed"]:
                            num_collision += 1

                # env.render()  # Uncomment if you want to render the environment

            all_collisions.append(num_collision)
            all_violations.append(num_violations)
            all_avoided_violations.append(num_avoided_violations)
            all_violations_dict.append(ep_violations_dict)
            all_avoided_violations_dict.append(ep_avoided_violations_dict)
            print(f"Experiment {experiment + 1}/{num_experiments} finished. Collision count: {num_collision}, Unavoided Violatons: {str(all_violations_dict)}, Total Unavoided: {num_violations}, Avoided Violations: {str(all_avoided_violations_dict)}, Total Avoided Violations: {num_avoided_violations}")

        env.close()
        results[mode] = [all_collisions, all_violations, all_avoided_violations, all_violations_dict, all_avoided_violations_dict]
        print(f"Results for {mode}: Collisions: {all_collisions}, Unavoided Violatons: {str(all_violations_dict)}, Total Unavoided: {num_violations}, Avoided Violations: {str(all_avoided_violations_dict)}, Total Avoided Violations: {num_avoided_violations}")

    if not os.path.exists(output_file):
        open(output_file, 'w').close()

    # get count of string 'Training run #' in file
    count = 1
    with open(output_file, "r") as f2:
        lines = f2.readlines()
        count += sum(1 for line in lines if "Training run #" in line) 

    # Write results to a file
    with open(output_file, "a") as f:

        f.write(f"######## Training run #{count} ############\n")
        f.write(f"Model: {model_files[model_index]}:\n")
        f.write(f"Experiments: {num_experiments}\n")
        f.write(f"Episodes: {num_episodes}\n\n")

        for mode, [collisions, violations, avoided_violations, violations_dict, avoided_violations_dict] in results.items():
            f.write(f"{mode}\n")
            
            f.write(f"Collisions: {sum(collisions)}\n")
            f.write(f"Average collisions: {np.mean(collisions):.2f} ({np.std(collisions):.2f})\n")
            f.write(f"Total unavoided violations: {sum(violations)}\n")
            f.write(f"Average unavoided violatoins by type: {average_by_presence(violations_dict)}\n")
            f.write(f"Average total unavoided violations: {np.mean(violations):.2f} ({np.std(violations):.2f}) \n\n")
            f.write(f"Total avoided violations: {sum(avoided_violations)}\n")
            f.write(f"Average avoided violatoins by type: {average_by_presence(avoided_violations_dict)}\n")
            f.write(f"Average total avoided violations: {np.mean(avoided_violations):.2f} ({np.std(avoided_violations):.2f}) \n\n")

    print(f"Results written to {output_file}.txt")
if __name__ == "__main__":
    main()