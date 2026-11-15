"""Evaluate a trained Puck standing policy."""
import argparse, pathlib, time

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")

    print("importing torch...", end=" ", flush=True)
    start = time.perf_counter()
    import torch
    print(f"{time.perf_counter() - start:.2f}s")
    print("importing genesis...", end=" ", flush=True)
    start = time.perf_counter()
    import genesis as gs
    print(f"{time.perf_counter() - start:.2f}s")
    from genesis_forge.wrappers import RslRlWrapper
    from rsl_rl.runners import OnPolicyRunner
    from puck.env import PuckStandEnv

    if args.device == "cpu":
        torch.set_default_device("cpu")
        backend = gs.cpu
    else:
        backend = gs.gpu
    gs.init(backend=backend, logging_level="warning")

    base_env = PuckStandEnv(headless=not args.render)
    env = RslRlWrapper(base_env)
    try:
        env.build()
        observations, _ = env.reset()
        from .train import training_config
        runner = OnPolicyRunner(
            env,
            training_config(),
            str(args.checkpoint.parent),
            device=gs.device,
        )
        runner.load(str(args.checkpoint))
        policy = runner.get_inference_policy(device=gs.device)

        for episode in range(1, args.episodes + 1):
            total_reward = 0.0
            for _ in range(base_env.max_episode_length_steps + 1):
                with torch.inference_mode():
                    actions = policy(observations)
                    observations, reward, dones, extras = env.step(actions)
                total_reward += float(reward[0].item())
                if bool(dones[0].item()):
                    outcome = (
                        "fell"
                        if bool(extras["terminations"][0].item())
                        else "timeout"
                    )
                    print(
                        f"Episode {episode}: reward={total_reward:.2f}, "
                        f"outcome={outcome}"
                    )
                    break
            else:
                raise RuntimeError(
                    "Evaluation episode did not terminate within its time limit"
                )
    finally:
        env.close()


if __name__ == "__main__":
    main()
