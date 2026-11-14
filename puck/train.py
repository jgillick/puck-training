"""Train the Puck standing policy with Genesis Forge and RSL-RL."""

import argparse
import copy
import pathlib

import genesis as gs
import torch
from genesis_forge.wrappers import RslRlWrapper
from rsl_rl.runners import OnPolicyRunner

from puck.env import PROJECT_ROOT, PuckStandEnv

POLICY_VERSION = 1


class PuckRunner(OnPolicyRunner):
    """Tag saved checkpoints so incompatible action semantics cannot be resumed."""

    def save(self, path: str, infos: dict | None = None) -> None:
        checkpoint_info = dict(infos or {})
        checkpoint_info["puck_policy_version"] = POLICY_VERSION
        super().save(path, checkpoint_info)


def is_current_policy_checkpoint(path: pathlib.Path) -> bool:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint_info = checkpoint.get("infos")
    return (
        isinstance(checkpoint_info, dict)
        and checkpoint_info.get("puck_policy_version") == POLICY_VERSION
    )


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
        "--resume",
        type=pathlib.Path,
        help="Resume training from an RSL-RL checkpoint.",
    )
    parser.add_argument(
        "--log-dir",
        type=pathlib.Path,
        default=PROJECT_ROOT / "runs" / "forge" / "puck-torque",
    )
    args = parser.parse_args()
    if args.num_envs <= 0 or args.iterations <= 0:
        parser.error("--num-envs and --iterations must be positive")
    if args.resume is not None and not args.resume.is_file():
        parser.error(f"Checkpoint does not exist: {args.resume}")
    if args.resume is not None and not is_current_policy_checkpoint(args.resume):
        parser.error(
            "Checkpoint is not from the torque-only policy; "
            "train a new checkpoint before resuming"
        )

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
        runner = PuckRunner(
            env,
            copy.deepcopy(training_config(args.seed)),
            str(args.log_dir),
            device=gs.device,
        )
        if args.resume is not None:
            runner.load(str(args.resume))
            runner.current_learning_iteration += 1
            print(f"Resuming training from {args.resume}")
        runner.learn(
            num_learning_iterations=args.iterations,
            init_at_random_ep_len=False,
        )
        print(f"Training checkpoints and logs saved in {args.log_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
