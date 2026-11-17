"""Drive a trained Puck policy with the arrow keys.

Up/Down: forward/back    Left/Right: turn
Run from the project root:  uv run python runsim.py runs/<date>/model_X.pt
"""
import argparse, pathlib, torch
import genesis as gs
from pynput import keyboard
from genesis_forge.wrappers import RslRlWrapper
from rsl_rl.runners import OnPolicyRunner
from puck.env import PuckStandEnv
from puck.train import training_config

Key = keyboard.Key
pressed: set = set()   # written by the listener thread, read by the control loop

def on_press(key):
    print(f"Key pressed: {key}")
    pressed.add(key)

def on_release(key):
    pressed.discard(key)

def make_controller(vel):
    """Build the command tensor from held keys, each step."""
    rng = vel.range
    print(f"make_controller rng={rng}")

    def controller(_step):
        print(f"controller pressed={pressed}")
        cmd = torch.zeros_like(vel.command)
        fwd = (rng["lin_vel_x"][1] if Key.up in pressed else 0.0) + \
              (rng["lin_vel_x"][0] if Key.down in pressed else 0.0)
        turn = (rng["ang_vel_z"][1] if Key.left in pressed else 0.0) + \
               (rng["ang_vel_z"][0] if Key.right in pressed else 0.0)
        cmd[:, vel.get_command_idx("lin_vel_x")] = fwd
        cmd[:, vel.get_command_idx("ang_vel_z")] = turn
        return cmd

    return controller

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")

    if args.device == "cpu":
        torch.set_default_device("cpu")
    gs.init(backend=gs.cpu if args.device == "cpu" else gs.gpu, logging_level="warning")

    base_env = PuckStandEnv(headless=False)
    env = RslRlWrapper(base_env)
    env.build()

    base_env.velocity_command.use_external_controller(
        make_controller(base_env.velocity_command)
    )

    obs, _ = env.reset()
    runner = OnPolicyRunner(env, training_config(), str(args.checkpoint.parent), device=gs.device)
    runner.load(str(args.checkpoint))
    policy = runner.get_inference_policy(device=gs.device)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    print("Arrow keys to drive. Ctrl+C to quit.")
    try:
        with torch.inference_mode():
            while True:
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        env.close()

if __name__ == "__main__":
    main()
