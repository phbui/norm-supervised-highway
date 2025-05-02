from collections import defaultdict
import gymnasium
import json
import numpy as np
import os

from stable_baselines3 import DQN

from supervisor import Supervisor
import metrics
import norms

BASE_SEED = 239

def list_files(directory, extension=".json"):
    return sorted([f for f in os.listdir(directory) if f.endswith(extension)])

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

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
        all_tets = []
        all_safety_scores = []

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
            ep_tets = []
            ep_safety_scores = []

            for episode in range(num_episodes):
                done = truncated = False
                episode_seed = experiment_seed * (10 ** len(str(abs(num_episodes)))) + episode
                obs, _ = env.reset(seed=episode_seed) # <- seeded
                supervisor.reset_norms()
                # Used for safety score calculations
                tailgating_norm = norms.TailgatingNorm(
                    env.unwrapped.road,
                    env.unwrapped.action_type,
                    env_config["simulation_frequency"]
                )
                local_num_violations = 0
                local_num_avoided = 0
                local_violations_dict = {str(norm): 0 for norm in supervisor.norms}
                local_avoided_violations_dict = {str(norm): 0 for norm in supervisor.norms}
                local_ttc_history = []
                local_safety_scores = []
                tstep = 0
                while not (done or truncated):
                    tstep += 1
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

                    local_num_violations         += violations
                    local_num_avoided += avoided
                    count_by_presence(local_violations_dict, violations_dict)
                    count_by_presence(local_avoided_violations_dict, avoided_violations_dict)

                    ttcs = metrics.calculate_neighbour_ttcs(env.unwrapped.vehicle, env.unwrapped.road)
                    local_ttc_history.append(ttcs[0])

                    distance = tailgating_norm.evaluate_criteria(env.unwrapped.vehicle)
                    safe_distance = metrics.calculate_safe_distance(env.unwrapped.vehicle.speed, env.unwrapped.action_type, env_config["simulation_frequency"])
                    safety_score = metrics.calculate_safety_score(distance, safe_distance)
                    local_safety_scores.append(safety_score)

                    obs, reward, done, truncated, info = env.step(action)

                    if done or truncated:
                        if info["crashed"]:
                            num_collision += 1
                        
                        # Normalize the counts by the number of time steps
                        num_violations += local_num_violations / tstep
                        num_avoided_violations    += local_num_avoided / tstep
                        local_violations_dict         = {norm: count / tstep for norm, count in local_violations_dict.items()}
                        local_avoided_violations_dict = {norm: count / tstep for norm, count in local_avoided_violations_dict.items()}

                ep_violations_dict         = count_by_presence(ep_violations_dict, local_violations_dict)
                ep_avoided_violations_dict = count_by_presence(ep_avoided_violations_dict, local_avoided_violations_dict)
                ep_tets.append(metrics.calculate_tet(local_ttc_history, env_config["simulation_frequency"]))
                ep_safety_scores.append(np.nanmean(local_safety_scores))

                print(f"\nExperiment {experiment +1}/{num_experiments} ({mode}). Episode {episode + 1}/{num_episodes}, Collision: {num_collision}, Unavoided Violatons: {str(ep_violations_dict)}, Number of Violations: {num_violations}, Avoided Violations: {str(ep_avoided_violations_dict)}, Number of Avoided Violations: {num_avoided_violations}, TET: {ep_tets[-1]:.2f} seconds, Safety Score: {ep_safety_scores[-1]:.2f}")

                # env.render()  # Uncomment if you want to render the environment

            all_collisions.append(num_collision)
            all_violations.append(num_violations)
            all_avoided_violations.append(num_avoided_violations)
            all_violations_dict.append(ep_violations_dict)
            all_avoided_violations_dict.append(ep_avoided_violations_dict)
            all_tets.append(np.nanmean(ep_tets))
            all_safety_scores.append(np.nanmean(ep_safety_scores))
            print(f"Experiment {experiment + 1}/{num_experiments} finished. Collision count: {num_collision}, Unavoided Violatons: {str(all_violations_dict)}, Total Unavoided: {num_violations}, Avoided Violations: {str(all_avoided_violations_dict)}, Total Avoided Violations: {num_avoided_violations}, Average TET: {np.mean(ep_tets):.4f} seconds, Average Safety Score: {np.nanmean(ep_safety_scores):.4f}")

        env.close()
        results[mode] = [all_collisions, all_violations, all_avoided_violations, all_violations_dict, all_avoided_violations_dict, all_tets, all_safety_scores]
        print(f"Results for {mode}: Collisions: {all_collisions}, Unavoided Violatons: {str(all_violations_dict)}, Total Unavoided: {num_violations}, Avoided Violations: {str(all_avoided_violations_dict)}, Total Avoided Violations: {num_avoided_violations}, Average TET: {np.mean(all_tets):.4f} seconds, Average Safety Score: {np.mean(all_safety_scores):.4f}")

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
        f.write(f"Environment config: {env_configs[env_index]}:\n")
        f.write(f"Lanes: {env_config['lanes_count']}\n")
        f.write(f"Vehicles: {env_config['vehicles_count']}\n")
        f.write(f"Duration: {env_config['duration']}\n")
        f.write(f"Experiments: {num_experiments}\n")
        f.write(f"Episodes: {num_episodes}\n\n")

        for mode, [collisions, violations, avoided_violations, violations_dict, avoided_violations_dict, tet, safety_score] in results.items():
            f.write(f"{mode}\n")
            
            f.write(f"Collisions: {sum(collisions)}\n")
            f.write(f"Average collisions: {np.mean(collisions):.2f} ({np.std(collisions):.2f})\n")
            f.write(f"Total unavoided violations: {sum(violations)}\n")
            f.write(f"Average unavoided violatoins by type: {average_by_presence(violations_dict)}\n")
            f.write(f"Average total unavoided violations: {np.mean(violations):.2f} ({np.std(violations):.2f}) \n\n")
            f.write(f"Total avoided violations: {sum(avoided_violations)}\n")
            f.write(f"Average avoided violatoins by type: {average_by_presence(avoided_violations_dict)}\n")
            f.write(f"Average total avoided violations: {np.mean(avoided_violations):.2f} ({np.std(avoided_violations):.2f}) \n\n")
            f.write(f"Average TET: {np.mean(tet):.4f} ({np.std(tet):.4f}) seconds\n")
            f.write(f"Average safety score: {np.nanmean(safety_score):.4f} ({np.nanstd(safety_score):.4f})\n\n")

    print(f"Results written to {output_file}.txt")
if __name__ == "__main__":
    main()