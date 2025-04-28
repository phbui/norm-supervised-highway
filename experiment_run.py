import os
import json
import gymnasium
import highway_env
from stable_baselines3 import DQN
import numpy as np
from collections import defaultdict
from supervisor import Supervisor
BASE_SEED = 239

def list_files(directory, extension=".json"):
    return sorted([f for f in os.listdir(directory) if f.endswith(extension)])

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def list_models(models_dir="models"):
    return list_files(models_dir, ".zip")

def get_output_filename(default="results.txt"):
    fname = input(f"Enter output filename [{default}]: ").strip()
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
            print(f"Creating environment with config from {env_config_path}...")
            env = gymnasium.make("highway-fast-v0", render_mode="rgb_array", config=env_config)
            supervisor = Supervisor(env.unwrapped, env_config, verbose=False) 
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

    # get count of string 'Training run #' in file
    count = 1
    with open("results.txt", "r") as f2:
        lines = f2.readlines()
        count += sum(1 for line in lines if "Training run #" in line) 

    # Write results to a file
    with open(output_file, "a") as f:

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
            f.write(f"Total unavoided violations: {sum(violations)}\n")
            f.write(f"Average unavoided violatoins by type: {average_by_presence(violations_dict)}\n")
            f.write(f"Average total unavoided violations: {np.mean(violations):.2f} ({np.std(violations):.2f}) \n\n")
            f.write(f"Total avoided violations: {sum(avoided_violations)}\n")
            f.write(f"Average avoided violatoins by type: {average_by_presence(avoided_violations_dict)}\n")
            f.write(f"Average total avoided violations: {np.mean(avoided_violations):.2f} ({np.std(avoided_violations):.2f}) \n\n")

    print("Results written to results.txt")
if __name__ == "__main__":
    main()
