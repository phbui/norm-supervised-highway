import os
import json
import gymnasium
import highway_env
from stable_baselines3 import DQN
import numpy as np
from supervisor import Supervisor

def list_files(directory, extension=".json"):
    return sorted([f for f in os.listdir(directory) if f.endswith(extension)])


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def list_models(models_dir="models"):
    return list_files(models_dir, ".zip")


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

    num_experiments = 5
    num_episodes = 100

    results = {"WITH SUPERVISOR": [], "WITHOUT SUPERVISOR": []}

    for mode in results.keys():
        all_collisions = []
        all_violations = []
        all_avoided_violations = []

        for experiment in range(num_experiments):
            print(f"\nExperiment {experiment + 1}/{num_experiments} ({mode}).")
            num_collision = 0
            num_violations = 0
            num_avoided_violations = 0
            print(f"Creating environment with config from {env_config_path}...")
            env = gymnasium.make("highway-fast-v0", render_mode="rgb_array", config=env_config)
            supervisor = Supervisor(env_config=env_config, verbose=False) 

            for episode in range(num_episodes):
                print(f"\nExperiment {experiment +1}/{num_experiments} ({mode}). Episode {episode + 1}/{num_episodes}, Collision: {num_collision}, Violations: {num_violations}, Avoided Violations: {num_avoided_violations}")
                done = truncated = False
                obs, info = env.reset()
                while not (done or truncated):
                    action, _states = model.predict(obs, deterministic=True)

                    if supervisor:
                        action, violations = supervisor.decide_action(action, obs, info) 
                    else:
                        """
                        get violations but dont override action.
                        note that violations tend to be higher for supervised agent because slow down fall back tends to persist a violation rather than 
                        speeding through it  
                        """
                        _, violations = supervisor.decide_action(action, obs, info) # 
                       
                    num_violations += violations # count violations
                    obs, reward, done, truncated, info = env.step(action)

                    # TODO: calclulate norm violations using state after step and safety metrics. 
                    avoided_violations = supervisor.detect_avoided_violations(violations, action, obs, info)
                    num_avoided_violations += avoided_violations

                    if done or truncated:
                        if info["crashed"]:
                            num_collision += 1

                # env.render()  # Uncomment if you want to render the environment

            all_collisions.append(num_collision)
            all_violations.append(num_violations)
            all_avoided_violations.append(num_avoided_violations)
            print(f"Experiment {experiment + 1}/{num_experiments} finished. Collision count: {num_collision}. Violations: {num_violations}, Avoided Violations: {num_avoided_violations}")

        env.close()
        results[mode] = [all_collisions, all_violations, all_avoided_violations]
        print(f"Results for {mode}: Collisions: {all_collisions}, Violations: {all_violations}, Avoided Violations: {all_avoided_violations}")

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

        for mode, [collisions, violations, avoided_violations] in results.items():
            f.write(f"{mode}\n")
            
            f.write(f"Collisions: {sum(collisions)}\n")
            f.write(f"Average collisions: {np.mean(collisions):.2f} ({np.std(collisions):.2f})\n")
            f.write(f"Violations: {sum(violations)}\n")
            f.write(f"Average violations: {np.mean(violations):.2f} ({np.std(violations):.2f}) \n\n")
            f.write(f"Avoided violations: {sum(avoided_violations)}\n")
            f.write(f"Average avoided violations: {np.mean(avoided_violations):.2f} ({np.std(avoided_violations):.2f}) \n\n")

    print("Results written to results.txt")
if __name__ == "__main__":
    main()
