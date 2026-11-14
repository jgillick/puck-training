"""Train the Puck standing policy with Genesis Forge and RSL-RL."""

import argparse
import copy
import pathlib

import genesis as gs
import torch
from genesis_forge.wrappers import RslRlWrapper
from rsl_rl.runners import OnPolicyRunner

from puck.env import PROJECT_ROOT, PuckStandEnv


def training_config(seed: int = 0) -> dict:
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 0.3,
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "elu",
            "obs_normalization": True,
        },
        "seed": seed,
        "num_steps_per_env": 24,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument(
        "--log-dir",
        type=pathlib.Path,
        default=PROJECT_ROOT / "runs" / "forge" / "puck-stand",
    )
    args = parser.parse_args()
    if args.num_envs <= 0 or args.iterations <= 0:
        parser.error("--num-envs and --iterations must be positive")

    if args.device == "cpu":
        torch.set_default_device("cpu")
        backend = gs.cpu
    else:
        backend = gs.gpu
    gs.init(backend=backend, seed=args.seed, logging_level="warning")

    args.log_dir.mkdir(parents=True, exist_ok=True)
    env = RslRlWrapper(PuckStandEnv(num_envs=args.num_envs, headless=True))
    try:
        env.build()
        env.reset()
        runner = OnPolicyRunner(
            env,
            copy.deepcopy(training_config(args.seed)),
            str(args.log_dir),
            device=gs.device,
        )
        runner.learn(
            num_learning_iterations=args.iterations,
            init_at_random_ep_len=False,
        )
        print(f"Training checkpoints and logs saved in {args.log_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
