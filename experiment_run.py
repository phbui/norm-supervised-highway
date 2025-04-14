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
    model_index = 0
    model_path = os.path.join("models", model_files[model_index])

    env_configs = list_files("configs/environment")
    if not env_configs:
        print("No environment config files found in 'configs/environment/' directory.")
        return

    print("\nAvailable environment configs:")
    for i, f in enumerate(env_configs):
        print(f"[{i}] {f}")
    env_index = 1
    env_config_path = os.path.join("configs/environment", env_configs[env_index])
    env_config = load_config(env_config_path)

    print(f"\nLoading model from {model_path}...")
    model = DQN.load(model_path)

    num_experiments = 10
    num_episodes = 100

    results = {"WITH SUPERVISOR": [], "WITHOUT SUPERVISOR": []}

    for mode in results.keys():
        all_collisions = []

        for experiment in range(num_experiments):
            print(f"\nExperiment {experiment + 1}/{num_experiments} ({mode}).")
            num_collision = 0
            print(f"Creating environment with config from {env_config_path}...")
            env = gymnasium.make("highway-fast-v0", render_mode="rgb_array", config=env_config)
            supervisor = Supervisor(verbose=False) if mode == "WITH SUPERVISOR" else None

            for episode in range(num_episodes):
                print(f"\nEpisode {episode + 1}/{num_episodes}, Collision count: {num_collision}")
                done = truncated = False
                obs, info = env.reset()
                while not (done or truncated):
                    action, _states = model.predict(obs, deterministic=True)

                    if supervisor:
                        action = supervisor.decide_action(action, obs, info)
                    obs, reward, done, truncated, info = env.step(action)

                    if done or truncated:
                        if info["crashed"]:
                            num_collision += 1

                # env.render()  # Uncomment if you want to render the environment

            all_collisions.append(num_collision)
            print(f"Experiment {experiment + 1}/{num_experiments} finished. Collision count: {num_collision}")

        env.close()
        results[mode] = all_collisions

    # Write results to a file
    with open("results.txt", "w") as f:
        f.write(f"Averaged over {num_experiments} experiments\n")
        f.write(f"{num_episodes} episodes\n\n")

        for mode, collisions in results.items():
            f.write(f"{mode}\n")
            f.write(f"Collisions: {collisions}\n")
            f.write(f"Average collisions: {np.mean(collisions):.2f}\n")
            f.write(f"SVD: {np.std(collisions):.2f}\n\n")

    print("Results written to results.txt")
if __name__ == "__main__":
    main()
