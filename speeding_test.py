import os
import json
import gymnasium
import highway_env
from stable_baselines3 import DQN
import numpy as np
from supervisor import Supervisor
import pygame
import time
from gymnasium.envs.registration import register
from highway_env.envs.highway_env import HighwayEnv
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.behavior import IDMVehicle
BASE_SEED = 239


def list_files(directory, extension=".json"):
    return sorted([f for f in os.listdir(directory) if f.endswith(extension)])


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def list_models(models_dir="models"):
    return list_files(models_dir, ".zip")

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
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "screen_width": 600,
            "screen_height": 150,
            "centering_position": [0.3, 0.5],
            "scaling": 5.5,
            "show_trajectories": False,
            "render_agent": False,
            
            # Road and Environment
            "initial_lane_id": 1,  # Middle lane (0-indexed)
            "initial_position": 10,  # Adjusted to position ego vehicle in traffic
            # Keeping the same rewards as requested

            "frequency": 50,
            
            # Vehicle Configuration
            "vehicles_count": 5,
            "controlled_vehicles": 1,
            "ego_spacing": 2,
            "vehicles_density": 1.5,
            
            # Custom Vehicle Placement
            "custom_vehicle_placement": True,
            "vehicle_positions": [
                # Slow vehicle directly in front of ego in the middle lane
                {"lane": 1, "position": 20, "speed": 29, "vehicle_type": "car"},
                # Faster vehicles in the left lane (lane 0)
                {"lane": 0, "position": 15, "speed": 30, "vehicle_type": "car"},
                
                {"lane": 0, "position": 80, "speed": 30, "vehicle_type": "car"},
                # Faster vehicles in the right lane (lane 2)
                {"lane": 2, "position": 15, "speed": 30, "vehicle_type": "car"},
                {"lane": 3, "position": 15, "speed": 30, "vehicle_type": "car"},
            ],
            
            # Other Parameters
            "offroad_terminal": True
        })
        return config

    def _create_vehicles(self) -> None:
        """Create vehicles with custom starting positions."""
        # Create the ego vehicle (controlled vehicle)
        ego_vehicle = Vehicle.make_on_lane(
            self.road,
            lane_index=("0", "1", self.config["initial_lane_id"]),
            longitudinal=self.config["initial_position"],
            speed=30  # Initial speed of ego vehicle
        )
        
        # Convert to controlled vehicle type
        ego_vehicle = self.action_type.vehicle_class(
            self.road, ego_vehicle.position, ego_vehicle.heading, ego_vehicle.speed
        )
        
        # Set speed threshold for ego vehicle
        
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
                vehicle =Vehicle.make_on_lane(
                    self.road,
                    lane_index=lane_index,
                    longitudinal=position,
                    speed=speed
                )
                
                # Set vehicle type and appearance
                if vehicle_type == "truck":
                    vehicle.LENGTH = 7.5  # Longer vehicle
                    vehicle.WIDTH = 2.5
                    vehicle.target_speed = min(speed, 25)  # Trucks are slower
                elif vehicle_type == "emergency":
                    # Create an emergency vehicle (ambulance, police, etc.)
                    vehicle.LENGTH = 5.0
                    vehicle.WIDTH = 2.0
                    # In newer versions, you might be able to set special graphics
                    if hasattr(vehicle, "sprite_type"):
                        vehicle.sprite_type = "emergency"
                elif vehicle_type == "motorcycle":
                    vehicle.LENGTH = 2.5  # Smaller vehicle
                    vehicle.WIDTH = 1.0
                    vehicle.target_speed = speed * 1.1  # Motorcycles might be faster
                
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

    env_configs = list_files("configs/environment")
    if not env_configs:
        print("No environment config files found in 'configs/environment/' directory.")
        return

    print("\nAvailable environment configs:")
    for i, f in enumerate(env_configs):
        print(f"[{i}] {f}")
    # env_index = 2 # uncomment to harcode env, comment next line1
    env_index = int(input("Select an environment config by number: "))
    env_config_path = os.path.join("configs/environment", env_configs[env_index])
    env_config = load_config(env_config_path)

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
            print(f"Creating environment with config from {env_config_path}...")
            env = gymnasium.make("custom-highway-v0", render_mode="human")
            supervisor = Supervisor(env.unwrapped, env_config, verbose=False) 
            ep_violations_dict = {str(norm): 0 for norm in supervisor.norms}
            ep_avoided_violations_dict = {str(norm): 0 for norm in supervisor.norms}

            for episode in range(num_episodes):
                print(f"\nExperiment {experiment +1}/{num_experiments} ({mode}). Episode {episode + 1}/{num_episodes}, Collision: {num_collision}, Violations: {str(ep_violations_dict)}, Number of Violations: {num_violations}, Avoided Violations: {str(ep_avoided_violations_dict)}, Number of Avoided Violations: {num_avoided_violations}")
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
            all_violations_dict.append(violations_dict)
            all_avoided_violations_dict.append(avoided_violations_dict)
            print(f"Experiment {experiment + 1}/{num_experiments} finished. Collision count: {num_collision}, Violations: {str(all_violations_dict)}, Total Violations: {num_violations}, Avoided Violations: {str(all_avoided_violations_dict)}, Total Avoided Violations: {num_avoided_violations}")

        env.close()
        results[mode] = [all_collisions, all_violations, all_avoided_violations, all_violations_dict, all_avoided_violations_dict]
        print(f"Results for {mode}: Collisions: {all_collisions}, Violations: {str(all_violations_dict)}, Total Violations: {num_violations}, Avoided Violations: {str(all_avoided_violations_dict)}, Total Avoided Violations: {num_avoided_violations}")

    # get count of string 'Training run #' in file
    count = 1
    with open("results.txt", "r") as f2:
        lines = f2.readlines()
        count += sum(1 for line in lines if "Training run #" in line) 

    # Write results to a file
    with open("results.txt", "a") as f:

        f.write(f"######## Training run #{count} ############\n")
        f.write(f"Model: {model_files[model_index]}:\n")
        f.write(f"Environment config: {env_configs[env_index]}:\n")
        f.write(f"Lanes: {env_config['lanes_count']}\n")
        f.write(f"Vehicles: {env_config['vehicles_count']}\n")
        f.write(f"Duration: {env_config['duration']}\n")
        f.write(f"Experiments: {num_experiments}\n")
        f.write(f"Episodes: {num_episodes}\n\n")

        for mode, [collisions, violations, avoided_violations, violations_dict, avoided_violations_dict] in results.items():
            f.write(f"{mode}\n")
            
            f.write(f"Collisions: {sum(collisions)}\n")
            f.write(f"Average collisions: {np.mean(collisions):.2f} ({np.std(collisions):.2f})\n")
            f.write(f"Violations: {str(violations_dict)}\n")
            f.write(f"Total violations: {sum(violations)}\n")
            f.write(f"Average violatoins by type: {average_by_presence(violations_dict)}")
            f.write(f"Average total violations: {np.mean(violations):.2f} ({np.std(violations):.2f}) \n\n")
            f.write(f"Avoided violations: {str(avoided_violations_dict)}\n")
            f.write(f"Total avoided violations: {sum(avoided_violations)}\n")
            f.write(f"Average avoided violatoins by type: {average_by_presence(avoided_violations_dict)}")
            f.write(f"Average total avoided violations: {np.mean(avoided_violations):.2f} ({np.std(avoided_violations):.2f}) \n\n")

    print("Results written to results.txt")
if __name__ == "__main__":
    main()