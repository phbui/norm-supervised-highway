from collections import defaultdict
import sys
from time import sleep
import gymnasium
import json
import numpy as np
import os
import torch

from stable_baselines3 import DQN

from norm_supervisor.supervisor import Supervisor, PolicyAugmentMode, PolicyAugmentMethod
import norm_supervisor.norms.norms as norms
import norm_supervisor.metrics as metrics

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

def process_args(args: list[str]) -> dict[str, any]:
    """Process command line arguments to determine the supervisor method and parameters.
    
    :param args: List of command line arguments.
    :return: Dictionary with method, fixed_beta, and kl_budget.
    """
    method     = PolicyAugmentMethod.ADAPTIVE.value  # Default method
    fixed_beta = 0.01        # Default fixed beta value
    kl_budget  = 0.001       # Default KL budget

    if len(args) == 1:
        method = "baseline"

    if len(args) > 1:
        method = args[1].lower()
        print(f"Using method: {method}")

    if len(args) > 2:
        cmd_value = args[2]
        try:
            value = float(cmd_value)
            if method == PolicyAugmentMethod.FIXED.value:
                fixed_beta = value
                print(f"Using fixed beta value: {fixed_beta}")
            elif method == PolicyAugmentMethod.ADAPTIVE.value:
                kl_budget = value
                print(f"Using KL budget: {kl_budget}")
            else:
                print(f"Unknown method: {method}. Using default values.")
        except ValueError:
            print(f"Invalid value argument: {cmd_value}. Using default value.")

    return {
        "method": method,
        "fixed_beta": fixed_beta,
        "kl_budget": kl_budget
    }

def main(env_name = "highway-fast-v0"):
    """Main function to run the experiment with the selected model and environment configuration.
    
    Example usage:
        python experiment_run.py                  # Run baseline methods
        python experiment_run.py adaptive 0.001   # Run supervisor with adaptive beta
        python experiment_run.py fixed 0.01       # Run supervisor with fixed beta
    """
    # Process command line arguments
    args = process_args(sys.argv)

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

    # Check if CUDA is available and set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"\nLoading model from {model_path}...")
    model = DQN.load(model_path, device=device)
    model.set_random_seed(BASE_SEED)

    num_experiments = 5
    num_episodes = 100

    if args['method'] == "baseline":
        results = {
            "WITH SUPERVISOR NAIVE AUGMENT": [],
            #"WITH SUPERVISOR FILTER ONLY": [],
            #"WITHOUT SUPERVISOR": []
        }
    else:
        results = {
            "WITH SUPERVISOR": []
        }

    for mode in results.keys():
        all_collisions = []
        all_close_calls = []
        all_violations = []
        all_avoided_violations = []
        all_violations_dict = []
        all_avoided_violations_dict = []
        all_violations_weight = []
        all_violations_weight_dict = []
        all_violations_weight_difference = []
        all_violations_weight_difference_dict = []
        all_tets = []
        all_safety_scores = []
        all_speeds = []

        for experiment in range(num_experiments):
            experiment_seed = BASE_SEED * (10 ** len(str(abs(num_experiments)))) + experiment
            print(f"\nExperiment {experiment + 1}/{num_experiments} ({mode}).")
            num_collision = 0
            num_close_calls = 0
            num_violations = 0
            num_violations_weight = 0
            num_avoided_violations = 0
            num_violations_weight_difference = 0
            print(f"Creating environment with config from {env_config_path}...")
            env = gymnasium.make(env_name, render_mode="rgb_array", config=env_config)
            if mode == "WITH SUPERVISOR NAIVE AUGMENT":
                supervisor_mode = PolicyAugmentMode.NAIVE_AUGMENT.value
            elif mode == "WITH SUPERVISOR FILTER ONLY":
                supervisor_mode = PolicyAugmentMode.FILTER_ONLY.value
            else:
                supervisor_mode = PolicyAugmentMode.DEFAULT.value
            verbose_supervisor = False # set for verbose output
            supervisor = Supervisor(
                env.unwrapped,
                env_config,
                mode=supervisor_mode,
                method=args['method'] if args['method'] != "baseline" else "adaptive", # Never used
                fixed_beta=args['fixed_beta'],
                kl_budget=args['kl_budget'],
                verbose=verbose_supervisor if mode.startswith("WITH SUPERVISOR") else False
            ) 
            ep_violations_dict = {str(norm): 0 for norm in supervisor.norms}
            ep_avoided_violations_dict = {str(norm): 0 for norm in supervisor.norms}
            ep_violations_weight_dict = {str(norm): 0 for norm in supervisor.norms}
            ep_violations_weight_difference_dict = {str(norm): 0 for norm in supervisor.norms}
            ep_tets = []
            ep_safety_scores = []
            ep_speeds = []

            for episode in range(num_episodes):
                done = truncated = False
                episode_seed = experiment_seed * (10 ** len(str(abs(num_episodes)))) + episode
                obs, _ = env.reset(seed=episode_seed) # <- seeded
                supervisor.reset_norms()
                # Used for safety score calculations
                tailgating_norm = norms.TailgatingNorm(
                    weight=5,
                    action_type=env.unwrapped.action_type,
                    simulation_frequency=env_config["simulation_frequency"]
                )
                local_num_violations = 0
                local_num_avoided = 0
                local_violations_weight_difference = 0
                local_violations_dict = {str(norm): 0 for norm in supervisor.norms}
                local_avoided_violations_dict = {str(norm): 0 for norm in supervisor.norms}
                local_violations_weight = 0
                local_violations_weight_dict = {str(norm): 0 for norm in supervisor.norms}
                local_violations_weight_difference_dict = {str(norm): 0 for norm in supervisor.norms}
                local_ttc_history = []
                local_safety_scores = []
                local_speed_history = []
                tstep = 0
                while not (done or truncated):
                    tstep += 1
                    action, _  = model.predict(obs, deterministic=True)
                    action = action.item()
                    violations_dict = supervisor.count_norm_violations(action)
                    violations = sum(violations_dict.values())
                    violations_weight_dict = supervisor.weight_norm_violations(violations_dict)
                    violations_weight = sum(violations_weight_dict.values())

                    if mode.startswith("WITH SUPERVISOR"):
                        # Select new action and compute number of avoided violations
                        new_action = supervisor.decide_action(model, obs)
                        new_violations_dict = supervisor.count_norm_violations(new_action)
                        new_violations = sum(new_violations_dict.values())
                        new_violations_weight_dict = supervisor.weight_norm_violations(new_violations_dict)
                        new_violations_weight = sum(new_violations_weight_dict.values())
                        avoided                 = violations - new_violations
                        avoided_violations_dict = {norm: violations_dict.get(norm, 0) - new_violations_dict.get(norm, 0) for norm in set(violations_dict) | set(new_violations_dict)}
                        weight_difference       = violations_weight - new_violations_weight
                        weight_difference_dict  = {norm: violations_weight_dict.get(norm, 0) - new_violations_weight_dict.get(norm, 0) for norm in set(violations_weight_dict) | set(new_violations_weight_dict)}
                        action                  = new_action
                        violations              = new_violations
                        violations_dict         = new_violations_dict
                        violations_weight       = new_violations_weight
                        violations_weight_dict  = new_violations_weight_dict
                    else:
                        avoided = 0
                        weight_difference = 0
                        avoided_violations_dict = {norm: 0 for norm in violations_dict}
                        weight_difference_dict = {norm: 0 for norm in violations_dict}

                    local_num_violations                += violations
                    local_violations_weight             += violations_weight
                    local_num_avoided                   += avoided
                    local_violations_weight_difference  += weight_difference
                    count_by_presence(local_violations_dict, violations_dict)
                    count_by_presence(local_avoided_violations_dict, avoided_violations_dict)
                    count_by_presence(local_violations_weight_dict, violations_weight_dict)
                    count_by_presence(local_violations_weight_difference_dict, weight_difference_dict)

                    ttcs = metrics.calculate_neighbour_ttcs(env.unwrapped.vehicle)
                    local_ttc_history.append(ttcs[0])

                    # Count number of time steps spent within the collision threshold
                    if any(ttc < Supervisor.COLLISION_THRESHOLD for ttc in ttcs):
                        num_close_calls += 1

                    distance = tailgating_norm.evaluate_criterion(env.unwrapped.vehicle)
                    safe_distance = metrics.calculate_safe_distance(env.unwrapped.vehicle.speed, env.unwrapped.action_type, env_config["simulation_frequency"])
                    safety_score = metrics.calculate_safety_score(distance, safe_distance)
                    local_safety_scores.append(safety_score)

                    local_speed_history.append(env.unwrapped.vehicle.speed)

                    obs, reward, done, truncated, info = env.step(action)

                    if done or truncated:
                        if info["crashed"]:

                            num_collision += 1
                        
                        # Normalize the counts by the number of time steps
                        num_violations                          += local_num_violations / tstep
                        num_violations_weight                   += local_violations_weight / tstep
                        num_avoided_violations                  += local_num_avoided / tstep
                        num_violations_weight_difference        += local_violations_weight_difference / tstep
                        local_violations_dict                   = {norm: count / tstep for norm, count in local_violations_dict.items()}
                        local_avoided_violations_dict           = {norm: count / tstep for norm, count in local_avoided_violations_dict.items()}
                        local_violations_weight_dict            = {norm: count / tstep for norm, count in local_violations_weight_dict.items()}
                        local_violations_weight_difference_dict = {norm: count / tstep for norm, count in local_violations_weight_difference_dict.items()}

                
                ep_violations_dict                   = count_by_presence(ep_violations_dict, local_violations_dict)
                ep_avoided_violations_dict           = count_by_presence(ep_avoided_violations_dict, local_avoided_violations_dict)
                ep_violations_weight_dict            = count_by_presence(ep_violations_weight_dict, local_violations_weight_dict)
                ep_violations_weight_difference_dict = count_by_presence(ep_violations_weight_difference_dict, local_violations_weight_difference_dict)
                ep_tets.append(metrics.calculate_tet(local_ttc_history, env_config["simulation_frequency"], Supervisor.BRAKING_THRESHOLD))
                ep_safety_scores.append(np.nanmean(local_safety_scores))
                ep_speeds.append(np.nanmean(local_speed_history))

                print(f"\nExperiment {experiment +1}/{num_experiments} ({mode}). " +
                      f"Episode {episode + 1}/{num_episodes}, " +
                      f"Collision: {num_collision}, " +
                      f"Close Calls: {num_close_calls}, " +
                      f"Unavoided Violatons: {str(ep_violations_dict)}, " +
                      f"Number of Violations: {num_violations}, " +
                      f"Avoided Violations: {str(ep_avoided_violations_dict)}, " +
                      f"Number of Avoided Violations: {num_avoided_violations}, " +
                      f"Violations Weight: {num_violations_weight}, " +
                      f"Violations Weight Difference: {num_violations_weight_difference}, " +
                      f"Violations Weight across Norms: {str(ep_violations_weight_dict)}, " +
                      f"Violations Weight Differences across Norms: {str(ep_violations_weight_difference_dict)}, " +
                      f"TET: {ep_tets[-1]:.2f} seconds, " +
                      f"Safety Score: {ep_safety_scores[-1]:.2f}, "
                      f"Speed: {ep_speeds[-1]:.2f} m/s")

                #env.render()  # Uncomment if you want to render the environment
                #sleep(1)

            all_collisions.append(num_collision)
            all_close_calls.append(num_close_calls)
            all_violations.append(num_violations)
            all_avoided_violations.append(num_avoided_violations)
            all_violations_dict.append(ep_violations_dict)
            all_avoided_violations_dict.append(ep_avoided_violations_dict)
            all_violations_weight.append(num_violations_weight)
            all_violations_weight_dict.append(ep_violations_weight_dict)
            all_violations_weight_difference.append(num_violations_weight_difference)
            all_violations_weight_difference_dict.append(ep_violations_weight_difference_dict)
            all_tets.append(np.nanmean(ep_tets))
            all_safety_scores.append(np.nanmean(ep_safety_scores))
            all_speeds.append(np.nanmean(ep_speeds))
            print(f"Experiment {experiment + 1}/{num_experiments} finished. " +
                  f"Collision count: {num_collision}, " +
                  f"Close Calls: {num_close_calls}, " +
                  f"Unavoided Violatons: {str(all_violations_dict)}, " +
                  f"Total Unavoided: {num_violations}, " +
                  f"Avoided Violations:{str(all_avoided_violations_dict)}, " +
                  f"Total Avoided Violations: {num_avoided_violations}, " +
                  f"Total Violations Weight: {num_violations_weight}, " +
                  f"Violations Weight Difference: {num_violations_weight_difference}, " +
                  f"Violations Weight across Norms: {str(all_violations_weight_dict)}, " +
                  f"Violations Weight Differences across Norms: {str(all_violations_weight_difference_dict)}, " +
                  f"Average TET: {np.mean(ep_tets):.4f} seconds, " +
                  f"Average Safety Score: {np.nanmean(ep_safety_scores):.4f}, "
                  f"Average Speed: {np.nanmean(ep_speeds):.4f} m/s")

        env.close()
        results[mode] = [
            all_collisions, 
            all_close_calls,
            all_violations, 
            all_avoided_violations, 
            all_violations_dict, 
            all_avoided_violations_dict, 
            all_violations_weight,
            all_violations_weight_dict,
            all_violations_weight_difference,
            all_violations_weight_difference_dict,
            all_tets, 
            all_safety_scores,
            all_speeds]
        
        print(f"Results for {mode}: Collisions: {all_collisions}, " +
              f"Close Calls: {all_close_calls}, " +
              f"Unavoided Violatons: {str(all_violations_dict)}, " +
              f"Total Unavoided: {num_violations}, " +
              f"Avoided Violations: {str(all_avoided_violations_dict)}, " +
              f"Total Avoided Violations: {num_avoided_violations}, " +
              f"Total Violations Weight: {num_violations_weight}, " +
              f"Violations Weight Difference: {num_violations_weight_difference}, " +
              f"Violations Weight across Actions: {str(all_violations_weight_dict)}, " +
              f"Violations Weight Differences across Actions: {str(all_violations_weight_difference_dict)}, " +
              f"Average TET: {np.mean(all_tets):.4f} seconds, " +
              f"Average Safety Score: {np.mean(all_safety_scores):.4f}, "
              f"Average Speed: {np.mean(all_speeds):.4f} m/s")

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
        f.write(f"Episodes: {num_episodes}\n")
        if "WITH SUPERVISOR" in results:
            f.write(f"Method: {args['method']}\n")
            f.write(f"Fixed Beta: {args['fixed_beta'] if args['method'] == PolicyAugmentMethod.FIXED.value else "N/A"}\n")
            f.write(f"KL Budget: {args['kl_budget'] if args['method'] == PolicyAugmentMethod.ADAPTIVE.value else "N/A"}\n\n")
        else:
            f.write("\n")

        for mode, [collisions,
                   close_calls,
                   violations, 
                   avoided_violations, 
                   violations_dict, 
                   avoided_violations_dict, 
                   violations_weight,
                   violations_weight_dict,
                   violations_weight_difference,
                   violations_weight_difference_dict,
                   tet, 
                   safety_score,
                   speed] in results.items():
            f.write(f"{mode}\n")
            f.write(f"Collisions: {sum(collisions)}\n")
            f.write(f"Average collisions: {np.mean(collisions):.2f} ({np.std(collisions):.2f})\n")
            f.write(f"Close calls ({Supervisor.COLLISION_THRESHOLD}): {sum(close_calls)}\n")
            f.write(f"Average close calls: {np.mean(close_calls):.2f} ({np.std(close_calls):.2f})\n")
            f.write(f"Total unavoided violations: {sum(violations)}\n")
            f.write(f"Average unavoided violations by type: {average_by_presence(violations_dict)}\n")
            f.write(f"Average total unavoided violations: {np.mean(violations):.2f} ({np.std(violations):.2f}) \n\n")
            f.write(f"Total avoided violations: {sum(avoided_violations)}\n")
            f.write(f"Average avoided violations by type: {average_by_presence(avoided_violations_dict)}\n")
            f.write(f"Average total avoided violations: {np.mean(avoided_violations):.2f} ({np.std(avoided_violations):.2f}) \n\n")
            f.write(f"Total violations weight: {sum(violations_weight)}\n")
            f.write(f"Average total violations weight: {np.mean(violations_weight):.2f} ({np.std(violations_weight):.2f}) \n")
            f.write(f"Average total violations weight by type: {average_by_presence(violations_weight_dict)}\n\n")
            f.write(f"Total violations weight difference: {sum(violations_weight_difference)}\n")
            f.write(f"Average total violations weight difference: {np.mean(violations_weight_difference):.2f} ({np.std(violations_weight_difference):.2f}) \n")
            f.write(f"Average total violations weight difference by type: {average_by_presence(violations_weight_difference_dict)}\n\n")        
            f.write(f"Average TET ({Supervisor.BRAKING_THRESHOLD}): {np.mean(tet):.4f} ({np.std(tet):.4f}) seconds\n")
            f.write(f"Average safety score: {np.nanmean(safety_score):.4f} ({np.nanstd(safety_score):.4f})\n")
            f.write(f"Average speed: {np.nanmean(speed):.4f} ({np.nanstd(speed):.4f}) m/s\n\n")

    print(f"Results written to {output_file}")

if __name__ == "__main__":
    main("highway-fast-v0")
    # main("merge-v0")

