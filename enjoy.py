"""Watch the trained agent fly. Usage: python enjoy.py [episodes]"""
import sys
import gymnasium as gym
from stable_baselines3 import PPO

EPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5

model = PPO.load("ppo_lander.zip")
env = gym.make("LunarLander-v3", render_mode="human")
for e in range(EPS):
    obs, _ = env.reset()
    done, total = False, 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)
        total += float(reward)
        done = bool(term or trunc)
    print(f"flight {e + 1}: {total:.0f} {'LANDING' if total >= 200 else ''}")
env.close()
