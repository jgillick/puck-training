# Puck RL training system 🏋🏼

### your _Personal Upright Control Kompanion_

Puck is an 8DOF bipedal wheeled robot project. This is my attempt to train an RL policy to stand it up and balance!

See also Puck's [MakerWorld design page](https://t413.com/go/puck-mw?ref=puckrl), [Firmware Flash](https://t413.com/p/projects/puck/?ref=puckrl) web tool, my [Discord](https://t413.com/go/discord?ref=puckrl).

![puck gif horizontal](https://t413.com/p/projects/puck/puck-horiz-small.gif)
![training gif](https://t413.com/p/projects/puck/pucktraining.gif)

## Basics

This repository is a _self-contained_ robot-model + training harness. It's goal is to, from scratch each time, train-up a small neural net to drive Puck. It's goal is to bring _determinism_ to the chaos of a new neural net. Think of this as a school's curriculum: it can start fresh with _any_ new student and bring them up to speed. Along the way the training process outputs video files to show the progress, and detailed charts of the progress let you gain insights.

## Running

- Get `uv`: `brew install uv` or `pip install uv`
- `uv sync` to install dependencies
- `uv run python -m puck.train --iterations 500` trains a policy (though 500 iterations isn't very much)
  * this outputs to runs/{date}/ by default.
  * Videos of training are saved to runs/{date}/videos/ every few iterations (configurable)
  * View charts of the training progress (live or after) with `uv run tensorboard --logdir runs/[yyyy-mm-dd]`
- Evaluate and render a checkpoint with `uv run python -m puck.evaluate runs/[yyyy-mm-dd]/model_100.pt --render`

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
   * The robot model starts out crouching on the ground in its initial zero-position. 
   * The env changes this to a standing position for better training.
- Control desires:
   * Height. Ideally expressed as a value from [0 to 1], 0 is crouching, 1 is maximum leg extension.
   * Head/Body pitch & roll. Two independent values. Both expressed from [-1 to 1]
   * Forward velocity in m/s. Referenced to _head/body position in the world_. When a forward velocity target is step-functioned up the robot will have to tilt forward-- the wheels will drive _backwards_. This is good, and starts making the body translate forwards. Cybergears are [0.140mm diam * pi * 300rpm / 60] = 2.2 m/s max (~5mph)
   * HeadBody yaw angular velocity. Referenced in rad/s
   * Side velocity in m/s. This is the difficult one. It requires the robot to _step_ with its wheels lifting off the ground in sequence.
   * Eventually: actions like jump, hop, and behavior modifiers.


## Notes about training

- The default robot position is made to be _standing_, not its zero position. This is essential for learning.
   * Joints use PositionActionManager, so they're commanded in position mode
   * Wheels are commanded in torque mode
- Model output size is driven by `hidden_dims` and is using the suggested `[512, 256, 128]`
- Drivability:
   * Position command ability is handled through VelocityCommandManager, one for fwd, yaw, and side
   * Height with `self.height_command = CommandManager(self, range(0.1, 0.5))` value is _meters_
   * Pitch & roll of the head-body
- ObservationManager keeps 5 steps of history
- Rewards _and_ observations values stay between -1 and 1
- Curriculum training:
   * Ramps up difficulty over time, once competency is shown
   * Can't stall. Learning happens with challenges. Iterating without improvement leads to bad habits.
   * Adds difficulty: noise, terrain, perturbations
   * Adds goals and goal difficulty for setpoints: height, pitch, roll, velocity tracking
