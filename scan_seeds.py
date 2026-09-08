"""Fast seed scan: find an episode that lands clean (ret > 150). No rendering."""
import gymnasium as gym
from stable_baselines3 import PPO

model = PPO.load("/Users/inunity/lunar-rl/ppo_lander.zip")
for seed in range(30):
    env = gym.make("LunarLander-v3")
    obs, _ = env.reset(seed=seed)
    done, ret, steps = False, 0.0, 0
    while not done and steps < 1000:
        act, _ = model.predict(obs, deterministic=True)
        obs, rew, term, trunc, _ = env.step(act)
        done = term or trunc; ret += rew; steps += 1
    flag = "  <-- USE" if ret > 150 else ""
    print(f"seed {seed}: return {ret:.0f} in {steps} steps{flag}", flush=True)
    env.close()
