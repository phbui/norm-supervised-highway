import os
import gymnasium
import highway_env
from stable_baselines3 import DQN

def list_models(models_dir="models"):
    files = [f for f in os.listdir(models_dir) if f.endswith(".zip")]
    return sorted(files)

def main():
    model_files = list_models()
    if not model_files:
        print("No trained models found in 'models/' directory.")
        return

    print("Available models:")
    for i, f in enumerate(model_files):
        print(f"[{i}] {f}")

    index = int(input("Select a model to run by number: "))
    model_path = os.path.join("models", model_files[index])

    print(f"Loading model from {model_path}...")
    model = DQN.load(model_path)

    env = gymnasium.make("highway-fast-v0", render_mode="human")

    while True:
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            env.render()

if __name__ == "__main__":
    main()
