"""Live 'brain' viz for the PPO LunarLander rover — Theia-reel style.

Left: the lander flying. Right: what's happening BELOW the learning,
per step — obs vector -> hidden-16 activations -> action probs + value,
plus the live decision readout (action, reward, return).

Output: brain.mp4 + brain_still.png in OUT.
Run: uv run --python 3.12 --with 'gymnasium[box2d]' --with stable-baselines3 \
       --with matplotlib --with imageio --with imageio-ffmpeg python3 viz_live.py
"""
import numpy as np
import torch as th
import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import imageio.v2 as imageio

OUT = "/Users/inunity/lunar-rl"
ACTS = ["DO NOTHING", "FIRE LEFT", "FIRE MAIN", "FIRE RIGHT"]
OBS_NAMES = ["x", "y", "vx", "vy", "angle", "angVel", "legL", "legR"]
NEON = ["#22d3ee", "#e879f9", "#a3e635", "#fbbf24"]


def brain_step(model, obs):
    """Manual forward pass: hidden activations, action probs, value."""
    with th.no_grad():
        o = th.as_tensor(obs, dtype=th.float32).unsqueeze(0).to(model.device)
        feats = model.policy.features_extractor(o)
        latent_pi, latent_v = model.policy.mlp_extractor(feats)
        logits = model.policy.action_net(latent_pi)
        probs = th.softmax(logits, dim=1).cpu().numpy()[0]
        value = float(model.policy.value_net(latent_v).cpu().numpy()[0][0])
        hidden = latent_pi.cpu().numpy()[0]
    return hidden, probs, value


def draw_frame(fig, ax_game, ax_obs, ax_hid, ax_pol, frame, obs, hidden, probs, value, step, act, rew, ret):
    ax_game.imshow(frame); ax_game.axis("off")
    ax_game.set_title(f"STEP {step}  |  DECISION: {ACTS[act]}  |  value {value:+.1f}  |  reward {rew:+.2f}  |  return {ret:.0f}",
                     color="white", fontsize=11, fontweight="bold", loc="left")

    ax_obs.barh(OBS_NAMES, obs, color="#22d3ee"); ax_obs.set_title("OBS (what it sees)", color="#22d3ee", fontsize=10)
    ax_pol.barh(ACTS, probs, color=[NEON[i] if i == act else "#334155" for i in range(4)])
    ax_pol.set_xlim(0, 1); ax_pol.set_title("POLICY (what it wants)", color="#e879f9", fontsize=10)
    ax_hid.bar(range(16), hidden, color="#a3e635"); ax_hid.set_title("HIDDEN-16 (what it thinks)", color="#a3e635", fontsize=10)
    ax_hid.set_xticks(range(16)); ax_hid.set_xticklabels([f"n{i}" for i in range(16)], fontsize=7, color="gray")
    for ax in (ax_obs, ax_hid, ax_pol):
        ax.set_facecolor("#0b1020"); ax.tick_params(colors="gray", labelsize=8)
        for s in ax.spines.values():
            s.set_color("#1e293b")


def main():
    model = PPO.load(f"{OUT}/ppo_lander.zip")
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, _ = env.reset(seed=13)

    fig = plt.figure(figsize=(14, 7), facecolor="#05070f")
    gs = GridSpec(2, 2, width_ratios=[1.15, 1], hspace=0.35, wspace=0.25)
    ax_game = fig.add_subplot(gs[:, 0]); ax_obs = fig.add_subplot(gs[0, 1])
    ax_hid = fig.add_subplot(gs[1, 1])
    # policy panel overlays bottom of obs? -> dedicated: split right col into 3 via inset
    ax_pol = fig.add_axes([0.56, 0.06, 0.40, 0.22])
    ax_hid.set_position([0.56, 0.36, 0.40, 0.22])
    ax_obs.set_position([0.56, 0.66, 0.40, 0.26])

    writer = imageio.get_writer(f"{OUT}/brain.mp4", fps=30, codec="libx264", quality=8)
    step, ret, done, saved_still = 0, 0.0, False, False
    while not done and step < 1200:
        hidden, probs, value = brain_step(model, obs)
        act, _ = model.predict(obs, deterministic=True)
        act = int(act)
        obs, rew, term, trunc, _ = env.step(act)
        done = term or trunc; ret += rew; step += 1
        draw_frame(fig, ax_game, ax_obs, ax_hid, ax_pol, env.render(), obs, hidden, probs, value, step, act, rew, ret)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        writer.append_data(buf)
        if not saved_still and step == 120:
            fig.savefig(f"{OUT}/brain_still.png", dpi=90, facecolor=fig.get_facecolor())
            saved_still = True
        for ax in (ax_game, ax_obs, ax_hid, ax_pol):
            ax.clear()
    writer.close()
    print(f"done: {step} steps, return {ret:.0f} -> brain.mp4 + brain_still.png")


if __name__ == "__main__":
    main()
