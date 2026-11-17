"""Evaluate a trained Puck standing policy."""
import argparse, pathlib, time, torch
import genesis as gs
from genesis.utils import geom

ARROW_COLOR = (0.1, 0.9, 0.2, 0.9)
ARROW_LENGTH = 0.35   # meters
ARROW_RADIUS = 0.01
ARROW_LIFT = 0.20     # draw above the body so it doesn't clip the mesh
AXIS_VECTORS = {
    "x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
}


def draw_heading_arrow(base_env, local_axis, env_idx: int = 0) -> None:
    """Draw an arrow from the robot along its body-frame forward axis."""
    base_env.scene.clear_debug_objects()
    pos = base_env.robot.get_pos()[env_idx]
    quat = base_env.robot.get_quat()[env_idx]
    axis = torch.tensor(local_axis, dtype=torch.float32, device=gs.device)
    world_dir = geom.transform_by_quat(axis, quat)
    origin = (pos + torch.tensor([0.0, 0.0, ARROW_LIFT], device=gs.device))
    base_env.scene.draw_debug_arrow(
        origin.cpu().numpy(),
        (world_dir * ARROW_LENGTH).cpu().numpy(),
        radius=ARROW_RADIUS,
        color=ARROW_COLOR,
    )


def describe_outcome(extras, base_env) -> str:
    if bool(extras["terminations"][0].item()):
        logs = extras.get(base_env.extras_logging_key, {})
        reasons = [
            name
            for name in ("ground_contact", "bad_orientation")
            if float(logs.get(f"Terminations / {name}", 0.0)) > 0.0
        ]
        return f"fell ({', '.join(reasons) or 'cause unknown'})"
    return "timeout"

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--forward-axis", choices=("x", "-x", "y", "-y"), default="y")
    parser.add_argument( "--linger", type=float, default=2.0, help="Seconds to keep simulating after termination (0 = move on immediately)")
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
    hold = {"active": False}
    _real_reset = base_env.reset

    def reset_unless_holding(env_ids=None):
        if hold["active"]:
            return None, base_env.extras  # skip the physics reset while lingering
        return _real_reset(env_ids)
    setattr(base_env, "reset", reset_unless_holding) #base_env.reset = reset_unless_holding

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
            forward_vec = AXIS_VECTORS[args.forward_axis]
            linger_steps = round(args.linger / base_env.dt)
            max_steps = base_env.max_episode_length_steps

            for episode in range(1, args.episodes + 1):
                total_reward = 0.0
                steps = 0
                outcome = None
                linger_left = 0
                hold["active"] = False

                while True:
                    if args.render:
                        draw_heading_arrow(base_env, forward_vec)

                    with torch.inference_mode():
                        actions = policy(observations)
                        observations, reward, dones, extras = env.step(actions)

                    cmd = env.unwrapped.velocity_command.command
                    v_body = env.unwrapped.entity_manager.get_linear_velocity()
                    print(f"cmd_x={cmd[0, 0]:+.2f} y={cmd[0, 1]:+.2f}  actual_body_x={v_body[0, 0]:+.2f} y={v_body[0, 1]:+.2f}")

                    total_reward += float(reward[0].item())
                    steps += 1

                    if outcome is None and bool(dones[0].item()):
                        outcome = describe_outcome(extras, base_env)
                        if linger_steps <= 0:
                            break
                        hold["active"] = True
                        linger_left = linger_steps
                        print(
                            f"Episode {episode}: {outcome} at step {steps}; "
                            f"lingering {args.linger:.1f}s"
                        )
                    elif outcome is not None:
                        linger_left -= 1
                        if linger_left <= 0:
                            break
                    elif max_steps is not None and steps > max_steps + 1:
                        raise RuntimeError(
                            "Evaluation episode did not terminate within its time limit"
                        )

                if hold["active"]:
                    hold["active"] = False
                    observations, _ = env.reset()  # real reset before the next episode

                print(
                    f"Episode {episode}: reward={total_reward:.2f}, "
                    f"outcome={outcome}"
                )
    finally:
        env.close()


if __name__ == "__main__":
    main()
