import os
import json
import gymnasium
import highway_env
from stable_baselines3 import DQN

def list_configs(config_dir="configs"):
    files = [f for f in os.listdir(config_dir) if f.endswith(".json")]
    return sorted(files)

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def main():
    config_files = list_configs()
    if not config_files:
        print("No config files found in 'configs/' directory.")
        return

    print("Available training configs:")
    for i, f in enumerate(config_files):
        print(f"[{i}] {f}")

    index = int(input("Select a config file by number: "))
    config_path = os.path.join("configs", config_files[index])
    config = load_config(config_path)

    model_name = input("Enter a name for the saved model: ")
    model_path = os.path.join("models", model_name)
    os.makedirs("models", exist_ok=True)

    env = gymnasium.make("highway-fast-v0", render_mode="human")

    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=config.get("net_arch", [256, 256])),
        learning_rate=config.get("learning_rate", 5e-4),
        buffer_size=config.get("buffer_size", 15000),
        learning_starts=config.get("learning_starts", 200),
        batch_size=config.get("batch_size", 32),
        gamma=config.get("gamma", 0.8),
        train_freq=config.get("train_freq", 1),
        gradient_steps=config.get("gradient_steps", 1),
        target_update_interval=config.get("target_update_interval", 50),
        verbose=1,
        tensorboard_log=config.get("tensorboard_log", "highway_dqn/")
    )

    print("Starting training...")
    model.learn(total_timesteps=int(config.get("total_timesteps", 20000)))
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
