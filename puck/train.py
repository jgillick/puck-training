"""Train a PPO policy for the Puck balance environment."""
import argparse, pathlib
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from puck.env import DEFAULT_MODEL_PATH, PROJECT_ROOT, PuckBalanceEnv


class ResetDifficultyCurriculum(BaseCallback):
    """Increase reset perturbations throughout training."""

    def __init__(self, total_timesteps: int) -> None:
        super().__init__()
        self.total_timesteps = total_timesteps
        self._last_percent = -1

    def _on_step(self) -> bool:
        percent = min(100, 100 * self.num_timesteps // self.total_timesteps)
        if percent != self._last_percent:
            self.training_env.env_method("set_difficulty", percent / 100.0)
            self._last_percent = percent
        return True


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
    parser.add_argument(
        "--target-height",
        type=float,
        default=0.3,
        help="normalized leg stance height target (0.05 to 1.0)",
    )
    parser.add_argument("--target-pitch", type=float, default=0.0, help="body pitch target in radians")
    parser.add_argument("--target-roll", type=float, default=0.0, help="body roll target in radians")
    parser.add_argument(
        "--target-body-height",
        type=float,
        default=None,
        help=(
            "override the automatically computed body COM height in meters; "
            "use for tilted targets that cannot auto-calibrate two-wheel contact"
        ),
    )
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
            target_stance_height=args.target_height,
            target_pitch=args.target_pitch,
            target_roll=args.target_roll,
            target_body_height=args.target_body_height,
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
        model.learn(
            total_timesteps=args.timesteps,
            callback=ResetDifficultyCurriculum(args.timesteps),
        )
        model_path = args.output_dir / "puck_ppo"
        model.save(model_path)
        print(f"Saved policy to {model_path}.zip")
    finally:
        env.close()


if __name__ == "__main__":
    main()
