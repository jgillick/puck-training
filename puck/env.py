"""Genesis Forge environment for teaching Puck to stand and balance."""

import math, pathlib, typing
from collections.abc import Sequence
import torch
import genesis as gs
from genesis_forge import ManagedEnvironment
from genesis_forge.managers import (
    ActuatorManager,
    BaseActionManager,
    CommandManager,
    EntityManager,
    ObservationManager,
    PositionActionManager,
    RewardManager,
    TerminationManager,
    VelocityCommandManager,
)
from genesis_forge.mdp import observations, reset, rewards, terminations

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
ROBOT_MODEL_PATH = PROJECT_ROOT / "robot.xml"
CONTROL_DT = 1 / 50
INITIAL_ROOT_POSITION = (0.0, 0.0, 0.23) #(x,y,z) height (m) offset caused by STANDING_JOINT_POSITIONS
IDENTITY_QUATERNION = (1.0, 0.0, 0.0, 0.0)
INITIAL_LIMB_KP = 5.0
INITIAL_LIMB_KD = 1.0
LIMB_JOINT_NAMES = (
    "flexor_r",
    "hip_r",
    "knee_r",
    "flexor_l",
    "hip_l",
    "knee_l",
)
WHEEL_JOINT_NAMES = ("wheel_r", "wheel_l")
STANDING_JOINT_POSITIONS = {
    "flexor_r": 0.0,
    "hip_r": math.pi / 4,
    "knee_r": math.pi / 2,
    "flexor_l": 0.0,
    "hip_l": -math.pi / 4,
    "knee_l": -math.pi / 2,
    "wheel_r": 0.0,
    "wheel_l": 0.0,
}
LEG_JOINT_NAMES = set(STANDING_JOINT_POSITIONS.keys()) - set(WHEEL_JOINT_NAMES)

def leg_pose_outside_band(env) -> torch.Tensor:
    assert isinstance(env, PuckStandEnv)
    positions = env.actuator_manager.get_dofs_position()
    leg_names = list(LEG_JOINT_NAMES)
    indices = [env.actuator_manager.dofs_names.index(n) for n in leg_names]
    targets = positions.new_tensor([STANDING_JOINT_POSITIONS[n] for n in leg_names])
    tolerance = 0.15  # radians; tune to allow balancing motion
    excess = torch.relu((positions[:, indices] - targets).abs() - tolerance)
    return (excess / tolerance).square().mean(dim=1)

class PuckRewardManager(RewardManager):
    """Keep the scalar reward returned to the trainer in the normalized range."""

    def step(self) -> torch.Tensor:
        return super().step().clamp_(-1.0, 1.0)


class PuckWheelTorqueActionManager(BaseActionManager):
    """Map two normalized actions to bounded wheel torques."""

    def build(self) -> None:
        super().build()
        self._force_lower, self._force_upper = self.env.robot.get_dofs_force_range(
            self.dofs_idx
        )

    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        actions = actions.clamp(-1.0, 1.0)
        return torch.where(
            actions >= 0.0,
            actions * self._force_upper,
            actions * -self._force_lower,
        )

    def send_actions_to_simulation(self, actions: torch.Tensor) -> torch.Tensor:
        self.env.robot.control_dofs_force(
            actions,
            self.dofs_idx,
        )
        return actions


class PuckStandEnv(ManagedEnvironment):
    """Train Puck to stand and balance with position-controlled limbs."""

    def __init__(
        self,
        num_envs: int = 1,
        headless: bool = True,
        max_episode_length_sec: int = 10,
    ) -> None:
        super().__init__(
            num_envs=num_envs,
            dt=CONTROL_DT,
            max_episode_length_sec=max_episode_length_sec,
        )
        self.scene = gs.Scene(
            show_viewer=not headless,
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            rigid_options=gs.options.RigidOptions(
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
        )
        self.terrain = self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file=str(ROBOT_MODEL_PATH),
                pos=INITIAL_ROOT_POSITION,
                quat=IDENTITY_QUATERNION,
            )
        )
        self.camera = self.scene.add_camera(
            pos=(1.2, -1.4, 0.9),
            lookat=(0.0, 0.0, 0.35),
            res=(640, 480),
            fov=40,
            spp=1,
            env_idx=0,
        )
        self.camera.follow_entity(
            self.robot,
            fixed_axis=(None, None, 0.9),  # keep the camera at a steady height
            smoothing=0.9,
        )

    def config(self) -> None:
        """Configure standing-pose actions, commands, rewards, and observations."""
        self.entity_manager = EntityManager(
            self,
            entity=self.robot,
            on_reset={
                "pose": {
                    "fn": reset.position(
                        position=INITIAL_ROOT_POSITION,
                        quat=IDENTITY_QUATERNION,
                    )
                }
            },
        )
        self.actuator_manager = ActuatorManager(
            self,
            joint_names=".*",
            default_pos=STANDING_JOINT_POSITIONS,
            kp=INITIAL_LIMB_KP,
            kv=INITIAL_LIMB_KD,
        )
        self.action_manager = PositionActionManager(
            self,
            actuator_manager=self.actuator_manager,
            actuator_joints=list(LIMB_JOINT_NAMES),
            scale=0.5,
            use_default_offset=True,
        )
        self.wheel_action_manager = PuckWheelTorqueActionManager(
            self,
            actuator_manager=self.actuator_manager,
            actuator_joints=list(WHEEL_JOINT_NAMES),
        )
        self.velocity_command = VelocityCommandManager(
            self,
            range={
                "lin_vel_x": (0.0, 0.1),
                "lin_vel_y": (0.0, 0.0),
                "ang_vel_z": (-0.1, 0.1),
            },
        )
        self.height_command = CommandManager(self, range=(0.1, 0.5))

        def ground_contact_failure(env) -> torch.Tensor:
            contacts = self.robot.get_contacts(with_entity=self.terrain, is_padded=True)
            wheel_ids = torch.tensor(
                [
                    self.robot.get_link("wheel").idx,
                    self.robot.get_link("wheel_2").idx,
                ],
                device=gs.device,
            )
            wheel_contacts = (
                contacts["link_b"].unsqueeze(-1) == wheel_ids
            ).any(dim=-1)
            non_wheel_contacts = contacts["valid_mask"] & ~wheel_contacts
            return non_wheel_contacts.any(dim=1)

        def hip_flexor_alignment(env) -> torch.Tensor:
            joint_positions = self.actuator_manager.get_dofs_position()
            flexor_indices = [
                self.actuator_manager.dofs_names.index(name)
                for name in ("flexor_r", "flexor_l")
            ]
            max_flexor_angle = joint_positions[:, flexor_indices].abs().amax(dim=1)
            tolerance = math.pi / 30
            excess = torch.relu(max_flexor_angle - tolerance)
            return torch.exp(-4.0 * (excess / tolerance).square())

        def yaw_rate_error(env) -> torch.Tensor:
            commanded_yaw_rate = self.velocity_command.command[:, 2]
            actual_yaw_rate = self.entity_manager.get_angular_velocity()[:, 2]
            return (actual_yaw_rate - commanded_yaw_rate).square()

        def hip_knee_symmetry_reward(env) -> torch.Tensor:
            '''Reward symmetrical hip & knee (values are inverted between l / r)'''
            positions = self.actuator_manager.get_dofs_position()
            dof_indices = {
                name: index
                for index, name in enumerate(self.actuator_manager.dofs_names)
            }
            mirrored_joints = (
                ("hip_l", "hip_r", -1.0),
                ("knee_l", "knee_r", -1.0),
            )
            errors = torch.stack(
                [
                    positions[:, dof_indices[left]]
                    - sign * positions[:, dof_indices[right]]
                    for left, right, sign in mirrored_joints
                ],
                dim=1,
            )
            return torch.exp(-errors.square().mean(dim=1))


        PuckRewardManager(
            self,
            cfg={
                "forward_velocity": {
                    "weight": 1.0,
                    "fn": rewards.command_tracking_lin_vel(
                        vel_cmd_manager=self.velocity_command,
                        entity_manager=self.entity_manager,
                    ),
                },
                "upright": {
                    "weight": -5.0,
                    "fn": rewards.flat_orientation_l2(
                        entity_manager=self.entity_manager
                    ),
                },
                "yaw_velocity": {
                    "weight": 1.0,
                    "fn": rewards.command_tracking_ang_vel(
                        vel_cmd_manager=self.velocity_command,
                        entity_manager=self.entity_manager,
                    ),
                },
                "yaw_rate_error": {
                    "weight": -0.5,
                    "fn": yaw_rate_error,
                },
                "leg_symmetry": {
                    "weight": 1.0,
                    "fn": hip_knee_symmetry_reward,
                },
                "hip_flexor_alignment": {
                    "weight": 1.0,
                    "fn": hip_flexor_alignment,
                },
                "leg_pose_outside_band": {
                    "weight": -1.0,
                    "fn": leg_pose_outside_band,
                },
                "body_angular_velocity": {
                    "weight": -0.02,
                    "fn": rewards.ang_vel_xy_l2(entity_manager=self.entity_manager),
                },
                "body_linear_velocity": {
                    "weight": -0.05,
                    "fn": rewards.lin_vel_xy_l2(entity_manager=self.entity_manager),
                },
                "action_rate": {
                    "weight": -0.01,
                    "fn": rewards.action_rate_l2(
                        action_manager=self.action_manager
                    ),
                },
            },
        )
        TerminationManager(
            self,
            term_cfg={
                "timeout": {"fn": terminations.timeout(), "time_out": True},
                "ground_contact": {"fn": ground_contact_failure},
                "bad_orientation": {
                    "fn": terminations.bad_orientation(
                        limit_angle=75,
                        entity_manager=self.entity_manager,
                    )
                },
            },
        )
        self.observation_manager = ObservationManager(
            self,
            cfg={
                "projected_gravity": {
                    "fn": lambda env: self.entity_manager.get_projected_gravity(),
                    "clip": (-1.0, 1.0),
                },
                "body_angular_velocity": {
                    "fn": lambda env: self.entity_manager.get_angular_velocity(),
                    "scale": 0.2,
                    "clip": (-1.0, 1.0),
                    "units": "rad/s",
                },
                "body_linear_velocity": {
                    "fn": lambda env: self.entity_manager.get_linear_velocity(),
                    "scale": 0.5,
                    "clip": (-1.0, 1.0),
                    "units": "m/s",
                },
                "joint_position": {
                    "fn": lambda env: (
                        self.actuator_manager.get_dofs_position()
                    ),
                    "scale": 1.0 / math.pi,
                    "clip": (-1.0, 1.0),
                    "units": "rad",
                },
                "joint_velocity": {
                    "fn": lambda env: self.actuator_manager.get_dofs_velocity(),
                    "scale": 0.01,
                    "clip": (-1.0, 1.0),
                    "units": "rad/s",
                },
                "base_height": {
                    "fn": lambda env: (
                        self.entity_manager.base_pos[:, 2:3] - 0.3
                    )
                    / 0.2,
                    "clip": (-1.0, 1.0),
                    "units": "normalized",
                },
                "height_command": {
                    "fn": lambda env: (
                        (self.height_command.command - 0.3) / 0.2
                    ),
                    "clip": (-1.0, 1.0),
                    "units": "normalized",
                },
                "velocity_command": {
                    "fn": self.velocity_command.observation,
                    "scale": 10.0,
                    "clip": (-1.0, 1.0),
                },
                "actions": {
                    "fn": observations.current_actions(),
                    "clip": (-1.0, 1.0),
                },
            },
            history_len=5,
            noise=0.0,
        )

    def build(self) -> None:
        super().build()
        if self.actuator_manager.num_dofs != 8:
            raise ValueError(
                "Expected 8 actuated joints, "
                f"found {self.actuator_manager.num_dofs}: "
                f"{self.actuator_manager.dofs_names}"
            )

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None):
        observations_batch, info = super().reset(env_ids)

        if env_ids is None:
            reset_ids = self.all_envs_idx
        elif isinstance(env_ids, torch.Tensor):
            reset_ids = env_ids.to(device=gs.device, dtype=torch.long)
        else:
            reset_ids = torch.as_tensor(env_ids, device=gs.device, dtype=torch.long)

        default_positions = self.actuator_manager.default_dofs_pos
        if default_positions.ndim == 1:
            default_positions = default_positions.unsqueeze(0).expand(
                len(reset_ids), -1
            )
        else:
            default_positions = default_positions[reset_ids]
        self.robot.set_dofs_position(
            default_positions,
            dofs_idx_local=self.actuator_manager.dofs_idx,
            envs_idx=reset_ids,
        )
        self.robot.zero_all_dofs_velocity(reset_ids)
        self.entity_manager.step()

        if env_ids is None:
            observations_batch = self.get_observations()
        return observations_batch, info

    @staticmethod
    def _as_batch(value: torch.Tensor) -> torch.Tensor:
        return value.unsqueeze(0) if value.ndim == 1 else value
