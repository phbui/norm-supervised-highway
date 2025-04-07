import os
import json
import gymnasium
import highway_env
from stable_baselines3 import DQN


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
    model_index = int(input("Select a model to run by number: "))
    model_path = os.path.join("models", model_files[model_index])

    env_configs = list_files("configs/environment")
    if not env_configs:
        print("No environment config files found in 'configs/environment/' directory.")
        return

    print("\nAvailable environment configs:")
    for i, f in enumerate(env_configs):
        print(f"[{i}] {f}")
    env_index = int(input("Select an environment config by number: "))
    env_config_path = os.path.join("configs/environment", env_configs[env_index])
    env_config = load_config(env_config_path)

    print(f"\nLoading model from {model_path}...")
    model = DQN.load(model_path)

    print(f"Creating environment with config from {env_config_path}...")
    env = gymnasium.make("highway-fast-v0", render_mode="human", config=env_config)

    while True:
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            env.render()

if __name__ == "__main__":
    main()
