# Puck RL training system 🏋🏼

### your _Personal Upright Control Kompanion_

Puck is a 8DOF bipedal wheeled robot project. This is my attempt to train an RL policy to drive it!

![puck gif horizontal](https://t413.com/p/projects/puck/puck-horiz-small.gif)

## Setup

- Get `uv`: `brew install uv` or `pip install uv`
- `uv sync` to install dependencies
- `uv run python -m puck.train --timesteps 500000 --target-height 0.4` starts training a model!
   * see `--help` for all the options
- simulate results with `uv run mjpython -m puck.evaluate --render --episodes 1 --realtime 1 --steps 4000`

## Work in progress..

More to come.
