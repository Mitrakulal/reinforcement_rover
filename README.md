# reinforcement_rover — LunarLander PPO (Theia-reel style)

Teach a tiny neural net to land a moon lander. Inspired by the
[@gettheia_extension Lunar Lander reel](https://www.instagram.com/gettheia_extension/):
8-number state in, 4 thruster actions out, ~10k episodes of hovering before it settles.

## The layers (bottom-up)

| Layer | What | In this repo |
|---|---|---|
| Env | `LunarLander-v3` (Gymnasium + Box2D physics) | 8 obs: x, y, vx, vy, angle, angular vel, 2 leg contacts. 4 actions: main, left RCS, right RCS, coast |
| Policy net | PPO actor-critic, hidden layer of 16 | **373 params total** — the reel's 212-weight spirit |
| Trainer | stable-baselines3 PPO, 1M timesteps | `train.py` — logs return curve, saves model, evals, records landing video |
| Reward | stock LunarLander shaping | +100 soft landing, +10/leg, -100 crash, -0.3/main-engine frame |

## Run it

```bash
uv venv --python 3.12 .venv && uv pip install -r requirements.txt
.venv/bin/python train.py   # ~20-30 min on CPU, fully automatic
```

Outputs: `ppo_lander.zip` (model) · `curve.csv` + `curve.png` (return curve) · `landing.mp4` (best-of-3 eval flight).

## v1 results (honest)

1M steps / 1986 episodes: fast early climb (219 by ep 1200), final 50-ep average **70**,
eval **3/20 solved**, best recorded flight **192**. The tiny net learns to fly but doesn't
reliably settle — same lesson as the reel: shaping that teaches flying fights landing.
v2 plan: bigger net (64x64), 3-5M steps, tuned entropy.

## Files

- `train.py` — trainer + evaluator + video recorder
- `enjoy.py` — watch the trained agent fly live: `python enjoy.py 5`
- `requirements.txt` — pinned stack
- `curve.png` — v1 learning curve
- `landing.mp4` — v1 best flight (7.4s)
- `ppo_lander.zip` — v1 weights
