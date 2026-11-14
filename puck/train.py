"""Train a PPO policy for the Puck balance environment."""
import argparse, pathlib
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from puck.env import DEFAULT_MODEL_PATH, PROJECT_ROOT, PuckBalanceEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-path", type=pathlib.Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--output-dir", type=pathlib.Path, default=PROJECT_ROOT / "runs" / "ppo"
    )
    parser.add_argument("--max-episode-steps", type=int, default=500)
    args = parser.parse_args()

    if args.timesteps <= 0:
        parser.error("--timesteps must be positive")
    if args.n_steps <= 1:
        parser.error("--n-steps must be greater than 1")
    if args.batch_size <= 1:
        parser.error("--batch-size must be greater than 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = Monitor(
        PuckBalanceEnv(
            model_path=args.model_path,
            max_episode_steps=args.max_episode_steps,
        ),
        filename=str(args.output_dir / "monitor.csv"),
    )
    try:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=args.seed,
            n_steps=args.n_steps,
            batch_size=min(args.batch_size, args.n_steps),
        )
        model.learn(total_timesteps=args.timesteps)
        model_path = args.output_dir / "puck_ppo"
        model.save(model_path)
        print(f"Saved policy to {model_path}.zip")
    finally:
        env.close()


if __name__ == "__main__":
    main()
