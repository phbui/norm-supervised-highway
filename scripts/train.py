import os
import json
import gymnasium
import highway_env
import torch
from stable_baselines3 import DQN


def list_configs(config_dir):
    files = [f for f in os.listdir(config_dir) if f.endswith(".json")]
    return sorted(files)


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def choose_config(config_dir, config_type):
    configs = list_configs(config_dir)
    if not configs:
        print(f"No {config_type} config files found in '{config_dir}' directory.")
        exit(1)

    print(f"\nAvailable {config_type} configs:")
    for i, f in enumerate(configs):
        print(f"[{i}] {f}")

    index = int(input(f"Select a {config_type} config file by number: "))
    config_path = os.path.join(config_dir, configs[index])
    return load_config(config_path)


def main():
    # Check if CUDA is available and set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    training_config = choose_config("configs/training", "training")
    env_config = choose_config("configs/environment", "environment")

    model_name = input("\nEnter a name for the saved model: ")
    model_path = os.path.join("models", model_name)
    os.makedirs("models", exist_ok=True)

    env = gymnasium.make("highway-fast-v0", render_mode="human", config=env_config)

    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=training_config.get("net_arch", [256, 256])),
        learning_rate=training_config.get("learning_rate", 5e-4),
        buffer_size=training_config.get("buffer_size", 15000),
        learning_starts=training_config.get("learning_starts", 200),
        batch_size=training_config.get("batch_size", 32),
        gamma=training_config.get("gamma", 0.8),
        train_freq=training_config.get("train_freq", 1),
        gradient_steps=training_config.get("gradient_steps", 1),
        target_update_interval=training_config.get("target_update_interval", 50),
        verbose=1,
        tensorboard_log=training_config.get("tensorboard_log", "models/tensorboard_logs/"),
        device=device
    )

    print("\nStarting training...")
    model.learn(total_timesteps=int(training_config.get("total_timesteps", 20000)))
    model.save(model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    main()
