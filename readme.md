# Puck RL training system 🏋🏼

### your _Personal Upright Control Kompanion_

Puck is an 8DOF bipedal wheeled robot project. This is my attempt to train an RL policy to stand it up and balance!

See also Puck's [MakerWorld design page](https://t413.com/go/puck-mw?ref=puckrl), [Firmware Flash](https://t413.com/p/projects/puck/?ref=puckrl) web tool, my [Discord](https://t413.com/go/discord?ref=puckrl).

![puck gif horizontal](https://t413.com/p/projects/puck/puck-horiz-small.gif)

## Setup

- Get `uv`: `brew install uv` or `pip install uv`
- `uv sync` to install dependencies
- `uv run python -m puck.train --num-envs 64 --iterations 500` trains a policy
- Resume for 500 more iterations with `uv run python -m puck.train --num-envs 64 --iterations 500 --resume runs/forge/model_100.pt`
- Evaluate and render a checkpoint with `uv run python -m puck.evaluate runs/[yyyy-mm-dd]/model_100.pt --render`

## Training environment

The Genesis Forge environment imports `robot.xml` and builds a mixed flat/rough terrain in Genesis. It starts at the standing pose: right flexor/hip/knee at `0`, `pi/4`, `pi/2`, and mirrored left joints. Body height is measured from the MJCF base-link origin (`body` at local `0, 0, 0`), not the visual mesh center; the terrain-relative height reward is enabled with variable-height training.

The policy has eight outputs: six limb position offsets around that standing pose, then normalized direct wheel torques. Limb position control uses the robot's actuator limits and fixed initial gains; wheels remain torque controlled. The curriculum starts with a small forward-only velocity command (0.05–0.1 m/s) and zero pitch/roll targets, a fixed 0.3 m height command, no observation noise, and flat-terrain resets. It then adds observation noise (level 1), forward/reverse and yaw-rate commands (level 2), rough terrain (level 3), variable height targets (level 4), lateral velocity targets (level 5), and variable pitch/roll targets (level 6). Terrain selection follows the current curriculum level on every reset. Pitch and roll tracking rewards are active from the start to keep the headbody level; when angle targets vary, the generic upright penalty is disabled so it does not oppose them.

Progress is checked after every 1,200 environment steps and averaged over five episode-reset samples. Episodes must average at least 60% of the maximum length. During forward-only stages, forward tracking must exceed 0.8; once directional commands are active, each introduced velocity target and both attitude targets must track above 0.5. Variable-height training also requires mean squared height error below 0.01. At level 4, `leg_pose_outside_band` is removed; `leg_symmetry` and `hip_flexor_alignment` remain until level 7. Observations contain five steps of history and are clipped to `[-1, 1]`; returned rewards are clipped to the same range. Sustained non-wheel ground contact or a bad orientation ends the episode.

Training records a full-episode video every 25 iterations under `runs/[yyyy-mm-dd]/videos`, starting at an episode boundary. View the live TensorBoard logs in another terminal:

```sh
uv run tensorboard --logdir runs/[yyyy-mm-dd]
```

Checkpoints can be resumed with `--resume`; the iteration count is the number of additional iterations to run. Checkpoints from the previous action/observation layout are incompatible; start a fresh run before resuming or evaluating.

## About the robot

Puck is an 8DOF bipedal wheeled robot.

- 8 DOF, degrees of freedom, meaning 8 QDD actuators. Each leg has
   * Hip Flexor: Robstride EL05. This swings the leg out/in, letting the legs swing sidewise and/or the body/head have a roll axis.
   * Hip Actuator: Robstride EL05. This swings the leg forward/backward, also letting the body/head have a pitch axis.
   * Knee Actuator: Xiaomi Cybergear. This bends the leg at the knee.
   * Wheel Actuator: Xiaomi Cybergear. This spins the wheel / foot.
- The body is also the head. I'll use these interchangeably.
   * Big googly eyes
   * Super fancy hatch system
   * E-Stop on top
   * Nice recessed snap-in spot for the main controller: M5 Atom S3R
   * Holds the battery, a 6S drone battery with BMS
- MJCF Model is exported by `onshape-to-robot` then modified
   * IMU definition added based on controller position
   * setup actuators to be specific to the el05 + cybergear drives
   * add wheel friction, actuator sensors, and timestep
- Actuation specifics
   * Actuators are all driven in MIT mode, with torque, velocity, kp, kd, and position all being set in every packet. MIT mode works because if kp and kd are 0 then position target has zero effect. Otherwise vel/torq are like FF values. With `kp+kd=0` then vel and/or torque targets can work like standard torque or velocity modes.
   * Actuators are all zeroed with the robot _crouching_, its initial position in the MJCF as well.
   * Position targets are in radians. _Right-hand_ rotation is _positive_.
   * Each actuator replies back with position, velocity, and torque state after every command (and/or when requested). These are intended to be fed into the RL model.
- Simulation model specifics
   * The robot starts out _crouching_ in its initial position. Wheels very close to the ground, to where it hits the ground super quickly.
   * The initial robot-off, off-actuator, position it will come to a rest on will most likely have contact with the body, wheels, and hamstring (really the knee actuator bumpers which in this simplified model is the hamstring).
   * The wheels + hamstrings + body on the ground is a good power-on state. A good initial goal is to stand up, balancing to height of 0.5 (0 is fully crouching, 1 is full leg extension)
   * the real robot can drive around crouching, with the wheels and knees on the ground sliding. the simulator can do this too. This isn't really a desired movement state for this model though, at least right now.
   * my cpp classical controls implementation uses variable kp, set kd, and position mode. this will be easier for a policy to do too, rather than coordinate a multi-leveled control system with actuator position feedback and torque output. but if it's better .. go for it.
- Control desires:
   * Height. Expressed as a value from [0 to 1], 0 is crouching, 1 is maximum leg extension.
   * Head/Body pitch & roll. Two independent values. Both expressed from [-1 to 1]
   * Forward velocity in m/s. Referenced to _head/body position in the world_. When a forward velocity target is step-functioned up the robot will have to tilt forward-- the wheels will drive _backwards_. This is good, and starts making the body translate forwards. Cybergears are [0.140mm diam * pi * 300rpm / 60] = 2.2 m/s max (~5mph)
   * HeadBody yaw. Referenced in rad/s
   * Side velocity in m/s. This is the difficult one. It requres the robot to _step_ with its wheels lifting off the ground in sequence.
   * Eventually: actions like jump, hop, and behavior modifiers.

Notes from jeremy, the creator of genesis-forge:

- use PositionActionManager for limb joints (not wheels)
   - make default position be with puck _standing_, wheels on the ground, in mid-height position.
   - turn noise *off* until everything can stand, then reintroduce
- turn video output on!
   - env gets wrapped in video wrapper, trigger video every x number of iterations
- use tensor board to view the logs. tails incoming logs
- `"hidden_dims": [128, 128],` should be `[512, 256, 128]`. or more.
- add VelocityCommandManager. now. start at slow vel target
   * three: fwd, yaw, side. only fwd gets slow target
- add general `self.height_command = CommandManager(self, range(0.1, 0.5))` value is _meters_
- figure out where body 0,0 is. probably is important for height_command
- ObservationManager should add 5 steps of history
- Keep rewards _and_ observations values between -1 and 1. important.
- cirriculum:
   * get the system to where it can train to be standing first. Staning pose initial position, joints driven in position control will help get there.
   * DO NOT start building a cirriculum until it can stand
   * For future reference:
   * cirriculum starts going when reward > threshold. eg forward velocity result
   * ramps up difficulity:
      - 1st step: run w/ height set to 0.5
      - 2nd step: run with variable height. use `self.height_command`
      - 2nd step is faster fwd.
      - then reverse
      - then rough terrain

