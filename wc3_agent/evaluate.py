"""
Run a trained WC3 PPO agent.

Usage:
    python evaluate.py --model models/wc3_ppo_final
    python evaluate.py              # uses latest checkpoint automatically
"""
import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO
from env.wc3_env import WC3Env
from config import MODELS_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    if args.model:
        model_path = args.model
    else:
        zips = sorted(MODELS_DIR.glob("wc3_ppo_*.zip"))
        if not zips:
            print("No saved model found. Train first with: python train.py")
            return
        model_path = str(zips[-1])

    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    print("Waiting 3 seconds — switch to WC3 window...")
    time.sleep(3)

    env = WC3Env(render_mode="human")

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            steps += 1
            done = terminated or truncated
            env.render()

        result = info.get("result", "timeout")
        print(f"Episode {ep+1}: steps={steps}, reward={total_reward:.2f}, result={result}")

    env.close()


if __name__ == "__main__":
    main()
