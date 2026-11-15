# Puck RL training system 🏋🏼

### your _Personal Upright Control Kompanion_

Puck is an 8DOF bipedal wheeled robot project. This is my attempt to train an RL policy to stand it up and balance!

![puck gif horizontal](https://t413.com/p/projects/puck/puck-horiz-small.gif)

## Setup

- Get `uv`: `brew install uv` or `pip install uv`
- `uv sync` to install dependencies
- `uv run python -m puck.train --num-envs 64 --iterations 500` trains a policy
- Resume for 500 more iterations with `uv run python -m puck.train --num-envs 64 --iterations 500 --resume runs/forge/model_100.pt`
- Evaluate and render a checkpoint with `uv run python -m puck.evaluate runs/[yyyy-mm-dd]/model_100.pt --render`

## Training environment

The Genesis Forge environment imports `robot.xml` and builds the ground plane in Genesis. It starts at a 0.5 m base-link height with the standing pose: right flexor/hip/knee at `0`, `pi/4`, `pi/2`, and mirrored left joints. The body height observation and reward use the MJCF base-link origin (`body` at local `0, 0, 0`), not the visual mesh center.

The policy has eight outputs: six limb position offsets around that standing pose, then normalized direct wheel torques. Limb position control uses the robot's actuator limits and fixed initial gains; wheels remain torque controlled. Commands provide a 0–0.1 m/s forward target, zero side/yaw targets, and a height target in meters from 0.1–0.5. Observations contain five steps of history, have no noise during initial training, and are clipped to `[-1, 1]`; returned rewards are clipped to the same range. Rewards favor commanded body height and forward speed, uprightness, mirrored leg motion, and wheel support. Sustained body contact after a short grace period ends the episode.

Training records a four-second video every 20 iterations under `runs/[yyyy-mm-dd]/videos`. View the live TensorBoard logs in another terminal:

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
