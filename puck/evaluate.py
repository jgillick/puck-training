"""Evaluate a trained PPO policy in the Puck balance environment."""
import argparse, pathlib
from stable_baselines3 import PPO
from puck.env import DEFAULT_MODEL_PATH, PROJECT_ROOT, PuckBalanceEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        nargs="?",
        type=pathlib.Path,
        default=PROJECT_ROOT / "runs" / "ppo" / "puck_ppo.zip",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true", help="render the environment, stay open after finishing")
    parser.add_argument("--nohold", action="store_true", default=False, help="when rendering, close the window after finishing")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", type=pathlib.Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--steps", type=int, default=500)
    # allow realtime multiplier (like 1x, 2x, etc. default means non-realtime, fastest speed)
    parser.add_argument("--realtime", type=float, nargs="?", const=1.0, default=None, help="realtime multiplier")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    env = PuckBalanceEnv(
        model_path=args.model_path,
        render_mode="human" if args.render else None,
        max_episode_steps=args.steps,
    )
    env.render_realtime = args.realtime
    env.render_nohold = args.nohold
    try:
        model = PPO.load(args.model, env=env, device="auto")
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False
            info: dict[str, object] = {}
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
            outcome = "fell" if terminated else "time limit"
            print(
                f"Episode {episode + 1}: reward={total_reward:.2f}, "
                f"steps={steps}, outcome={outcome}, "
                f"final_tilt={info['tilt_angle']:.3f} rad"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
