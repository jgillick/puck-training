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
    parser.add_argument(
        "--target-height",
        type=float,
        default=0.3,
        help="normalized leg stance height target (0.05 to 1.0)",
    )
    parser.add_argument("--target-pitch", type=float, default=0.0, help="body pitch target in radians")
    parser.add_argument("--target-roll", type=float, default=0.0, help="body roll target in radians")
    parser.add_argument(
        "--target-forward-speed",
        type=float,
        default=0.0,
        help="body +Y forward speed target in meters per second",
    )
    parser.add_argument(
        "--target-sideways-speed",
        type=float,
        default=0.0,
        help="body +X sideways speed target in meters per second",
    )
    parser.add_argument(
        "--target-yaw-rate",
        type=float,
        default=0.0,
        help="body +Z yaw rate target in radians per second",
    )
    parser.add_argument(
        "--target-body-height",
        type=float,
        default=None,
        help=(
            "override the automatically computed body COM height in meters; "
            "use for tilted targets that cannot auto-calibrate two-wheel contact"
        ),
    )
    parser.add_argument(
        "--difficulty",
        type=float,
        default=0.5,
        help="reset perturbation strength from 0 (easy) to 1 (hard)",
    )
    # allow realtime multiplier (like 1x, 2x, etc. default means non-realtime, fastest speed)
    parser.add_argument("--realtime", type=float, nargs="?", const=1.0, default=None, help="realtime multiplier")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    env = PuckBalanceEnv(
        model_path=args.model_path,
        render_mode="human" if args.render else None,
        max_episode_steps=args.steps,
        target_stance_height=args.target_height,
        target_pitch=args.target_pitch,
        target_roll=args.target_roll,
        target_forward_speed=args.target_forward_speed,
        target_sideways_speed=args.target_sideways_speed,
        target_yaw_rate=args.target_yaw_rate,
        target_body_height=args.target_body_height,
    )
    env.set_difficulty(args.difficulty)
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
                f"final_tilt={info['tilt_angle']:.3f} rad, "
                f"pitch={info['pitch']:.3f} rad, roll={info['roll']:.3f} rad, "
                f"body_height={info['body_height']:.3f} m, "
                f"forward={info['forward_speed']:.3f} m/s, "
                f"sideways={info['sideways_speed']:.3f} m/s, "
                f"yaw_rate={info['yaw_rate']:.3f} rad/s"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
