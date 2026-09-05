"""LunarLander PPO - Theia-reel style: tiny policy net, return curve, landing video."""
import csv
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

ENV_ID = "LunarLander-v3"
TIMESTEPS = 1_000_000
OUT = "/Users/inunity/lunar-rl"


class CurveLog(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.ep = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep:
                self.ep += 1
                self.rows.append((self.ep, round(ep["r"], 1), round(ep["l"], 1)))
                if self.ep % 200 == 0:
                    print(f"ep {self.ep}: return {ep['r']:.0f}", flush=True)
        return True


def main():
    env = gym.make(ENV_ID)
    # ~229 params total: the reel's 212-weight spirit (8->16->4 + value head)
    model = PPO("MlpPolicy", env, verbose=0, seed=7,
                policy_kwargs=dict(net_arch=[16]))
    n = sum(p.numel() for p in model.policy.parameters())
    print(f"policy params: {n}", flush=True)
    cb = CurveLog()
    model.learn(total_timesteps=TIMESTEPS, callback=cb)
    model.save(f"{OUT}/ppo_lander.zip")
    with open(f"{OUT}/curve.csv", "w", newline="") as f:
        csv.writer(f).writerows([("episode", "return", "length")] + cb.rows)
    print(f"trained {cb.ep} episodes", flush=True)

    # evaluate
    ev = gym.make(ENV_ID)
    rets = []
    for _ in range(20):
        o, _ = ev.reset()
        done, R = False, 0.0
        while not done:
            o, r, term, trunc, _ = ev.step(model.predict(o, deterministic=True)[0])
            R += float(r)
            done = bool(term or trunc)
        rets.append(R)
    print(f"eval mean {np.mean(rets):.0f} solved(>=200): {sum(x >= 200 for x in rets)}/20", flush=True)

    # record best-of-3 landing video
    import imageio.v2 as imageio
    best = None
    for _ in range(3):
        rec = gym.make(ENV_ID, render_mode="rgb_array")
        o, _ = rec.reset()
        frames, done, R = [], False, 0.0
        while not done:
            frames.append(rec.render())
            o, r, term, trunc, _ = rec.step(model.predict(o, deterministic=True)[0])
            R += float(r)
            done = bool(term or trunc)
        rec.close()
        if best is None or R > best[0]:
            best = (R, frames)
    imageio.mimsave(f"{OUT}/landing.mp4", best[1], fps=50)
    print(f"video saved, return {best[0]:.0f}", flush=True)


if __name__ == "__main__":
    main()
