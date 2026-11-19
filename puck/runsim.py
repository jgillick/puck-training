import argparse, pathlib, time

class KeyboardController:
    """Optional WASD / arrow-key velocity controller (requires pynput)."""

    def __init__(self, command_manager):
        from pynput import keyboard

        self.manager = command_manager
        self.keys = set()
        self.names = list(command_manager.range.keys())
        self.listener = keyboard.Listener(
            on_press=self._press, on_release=self._release
        )
        self.listener.start()

    def _press(self, key):
        try:
            self.keys.add(key.char.lower())
        except AttributeError:
            self.keys.add(key)

    def _release(self, key):
        try:
            self.keys.discard(key.char.lower())
        except AttributeError:
            self.keys.discard(key)

    def __call__(self, _step):
        from pynput.keyboard import Key

        k = self.keys
        pressed = lambda *keys: any(key in k for key in keys)

        # Normalized commands, mapped into the environment's configured ranges.
        values = {
            "lin_vel_x": float(pressed("w", Key.up)) - float(pressed("s", Key.down)),
            "lin_vel_y": float(pressed("e")) - float(pressed("q")),
            "ang_vel_z": float(pressed("a", Key.left)) - float(pressed("d", Key.right)),
        }

        command = torch.zeros(
            (1, len(self.names)), device=gs.device
        )
        for i, name in enumerate(self.names):
            lo, hi = self.manager.range[name]
            value = values.get(name, 0.0)
            command[0, i] = value * (hi if value > 0 else abs(lo))

        return command

    def stop(self):
        self.listener.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--input", choices=("gamepad", "keyboard"), default="gamepad")
    args = parser.parse_args()

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
    from puck.train import training_config

    if args.device == "cpu":
        torch.set_default_device("cpu")
        backend = gs.cpu
    else:
        backend = gs.gpu

    gs.init(backend=backend, logging_level="warning")

    base_env = PuckStandEnv(headless=False)
    env = RslRlWrapper(base_env)
    controller = None

    try:
        env.build()

        # Interactive velocity commands
        if args.input == "gamepad":
            from genesis_forge.gamepads import Gamepad

            print("Connecting gamepad...")
            controller = Gamepad()
            base_env.velocity_command.use_gamepad(
                controller,
                lin_vel_x_axis=1,
                lin_vel_y_axis=0,
                ang_vel_z_axis=2,
            )
        else:
            print("Keyboard controls: WASD / arrows; Q/E for sideways motion")
            controller = KeyboardController(base_env.velocity_command)
            base_env.velocity_command.use_external_controller(controller)

        # Load trained policy
        print(f"Loading {args.checkpoint}...")
        runner = OnPolicyRunner(
            env,
            training_config(),
            str(args.checkpoint.parent),
            device=gs.device,
        )
        runner.load(str(args.checkpoint))
        policy = runner.get_inference_policy(device=gs.device)

        obs, _ = env.reset()
        print("Puck is live! Ctrl-C to quit.")

        with torch.inference_mode():
            while True:
                actions = policy(obs)
                obs, _rews, _dones, _infos = env.step(actions)

    except KeyboardInterrupt:
        pass
    except gs.GenesisException as e:
        if str(e) != "Viewer closed.":
            raise
    finally:
        if controller is not None:
            controller.stop()
        env.close()


if __name__ == "__main__":
    main()
