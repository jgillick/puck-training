"""Genesis Forge environment for teaching Puck to stand and balance."""

import math
import pathlib
from collections.abc import Sequence

import genesis as gs
import torch
from genesis_forge import ManagedEnvironment
from genesis_forge.managers import (
    ActuatorManager,
    BaseActionManager,
    EntityManager,
    ObservationManager,
    RewardManager,
    TerminationManager,
)
from genesis_forge.mdp import observations, reset, rewards, terminations

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
ROBOT_MODEL_PATH = PROJECT_ROOT / "robot.xml"
CONTROL_DT = 1 / 50
REST_SETTLE_STEPS = 250
INITIAL_ROOT_POSITION = (0.0, 0.0, 0.2)
IDENTITY_QUATERNION = (1.0, 0.0, 0.0, 0.0)
STAND_HEIGHT = 0.5

STAND_POSE = {
    "flexor_l": 0.0,
    "flexor_r": 0.0,
    "hip_l": -math.pi * STAND_HEIGHT / 2,
    "hip_r": math.pi * STAND_HEIGHT / 2,
    "knee_l": -math.pi * STAND_HEIGHT,
    "knee_r": math.pi * STAND_HEIGHT,
    "wheel_l": 0.0,
    "wheel_r": 0.0,
}


class PuckActionManager(BaseActionManager):
    """Position-control the legs and apply policy torques to the wheels."""

    wheel_torque_limit = 12.0

    def __init__(
        self, env: ManagedEnvironment, actuator_manager: ActuatorManager
    ) -> None:
        super().__init__(
            env,
            actuator_manager=actuator_manager,
            actuator_joints=".*",
        )

    def build(self) -> None:
        super().build()
        names = tuple(self._dofs)
        self._position_action_indices = torch.tensor(
            [i for i, name in enumerate(names) if not name.startswith("wheel_")],
            device=gs.device,
        )
        self._wheel_action_indices = torch.tensor(
            [i for i, name in enumerate(names) if name.startswith("wheel_")],
            device=gs.device,
        )
        self.position_dofs_idx = [
            self.dofs_idx[i] for i in self._position_action_indices.tolist()
        ]
        self.wheel_dofs_idx = [
            self.dofs_idx[i] for i in self._wheel_action_indices.tolist()
        ]
        scale_by_joint = {
            "flexor": 0.3,
            "hip": 0.6,
            "knee": 1.0,
        }
        self._position_scale = torch.tensor(
            [
                scale_by_joint[name.split("_", maxsplit=1)[0]]
                for name in names
                if not name.startswith("wheel_")
            ],
            device=gs.device,
        )
        lower, upper = self.actuator_manager.get_dofs_limits(self.dofs_idx)
        self._position_lower = lower[self._position_action_indices]
        self._position_upper = upper[self._position_action_indices]
        self._position_default = self.actuator_manager.default_dofs_pos[
            :, self._position_action_indices
        ].clamp(
            min=self._position_lower,
            max=self._position_upper,
        )

    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        targets = torch.zeros_like(actions)
        positions = (
            self._position_default
            + actions[:, self._position_action_indices] * self._position_scale
        )
        targets[:, self._position_action_indices] = torch.maximum(
            torch.minimum(positions, self._position_upper), self._position_lower
        )
        targets[:, self._wheel_action_indices] = (
            actions[:, self._wheel_action_indices].clamp(-1.0, 1.0)
            * self.wheel_torque_limit
        )
        return targets

    def send_actions_to_simulation(self, actions: torch.Tensor) -> None:
        self.actuator_manager.control_dofs_position(
            actions[:, self._position_action_indices], self.position_dofs_idx
        )
        self.env.robot.control_dofs_force(
            actions[:, self._wheel_action_indices], self.wheel_dofs_idx
        )


class PuckStandEnv(ManagedEnvironment):
    """Train a position-controlled policy to raise Puck from its resting crouch."""

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
        self._rest_position: torch.Tensor | None = None
        self._rest_quaternion: torch.Tensor | None = None
        self._rest_dof_positions: torch.Tensor | None = None

    def config(self) -> None:
        """Configure the reset, position control, reward, and observation managers."""
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
            default_pos=STAND_POSE,
            kp={
                "flexor_.*": 10.0,
                "hip_.*": 18.0,
                "knee_.*": 18.0,
                "wheel_.*": 0.0,
            },
            kv={
                "flexor_.*": 0.5,
                "hip_.*": 0.8,
                "knee_.*": 0.8,
                "wheel_.*": 0.0,
            },
            max_force={
                "flexor_.*": 6.0,
                "hip_.*": 6.0,
                "knee_.*": 12.0,
                "wheel_.*": 12.0,
            },
        )
        self.action_manager = PuckActionManager(self, self.actuator_manager)

        def stand_pose_reward(env: ManagedEnvironment) -> torch.Tensor:
            position_error = self.actuator_manager.get_dofs_position()[
                :, self.action_manager._position_action_indices
            ] - self.action_manager._position_default
            return torch.exp(-4.0 * position_error.square().mean(dim=1))

        def wheel_support_reward(env: ManagedEnvironment) -> torch.Tensor:
            contacts = self.robot.get_contacts(with_entity=self.terrain, is_padded=True)
            link_ids = contacts["link_b"]
            valid = contacts["valid_mask"]
            body_contact = (
                (link_ids == self.robot.get_link("body").idx) & valid
            ).any(dim=1)
            wheel_ids = torch.tensor(
                [
                    self.robot.get_link("wheel").idx,
                    self.robot.get_link("wheel_2").idx,
                ],
                device=gs.device,
            )
            wheel_contact = (
                (link_ids.unsqueeze(-1) == wheel_ids) & valid.unsqueeze(-1)
            ).any(dim=1).any(dim=1)
            return wheel_contact.float() - body_contact.float()

        RewardManager(
            self,
            cfg={
                "stand_pose": {"weight": 4.0, "fn": stand_pose_reward},
                "wheel_support": {"weight": 2.0, "fn": wheel_support_reward},
                "upright": {
                    "weight": -5.0,
                    "fn": rewards.flat_orientation_l2(
                        entity_manager=self.entity_manager
                    ),
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
                "bad_orientation": {
                    "fn": terminations.bad_orientation(
                        limit_angle=75,
                        entity_manager=self.entity_manager,
                    )
                },
            },
        )
        ObservationManager(
            self,
            cfg={
                "projected_gravity": {
                    "fn": lambda env: self.entity_manager.get_projected_gravity()
                },
                "body_angular_velocity": {
                    "fn": lambda env: self.entity_manager.get_angular_velocity(),
                    "scale": 0.2,
                    "clip": (-50.0, 50.0),
                    "units": "rad/s",
                },
                "body_linear_velocity": {
                    "fn": lambda env: self.entity_manager.get_linear_velocity(),
                    "scale": 0.5,
                    "clip": (-20.0, 20.0),
                    "units": "m/s",
                },
                "joint_position_error": {
                    "fn": lambda env: (
                        self.actuator_manager.get_dofs_position()
                        - self.actuator_manager.default_dofs_pos
                    ),
                    "clip": (-10.0, 10.0),
                    "units": "rad",
                },
                "joint_velocity": {
                    "fn": lambda env: self.actuator_manager.get_dofs_velocity(),
                    "scale": 0.05,
                    "clip": (-200.0, 200.0),
                    "units": "rad/s",
                },
                "base_height": {
                    "fn": lambda env: self.entity_manager.base_pos[:, 2:3],
                    "units": "m",
                },
                "actions": {"fn": observations.current_actions()},
            },
        )

    def build(self) -> None:
        super().build()
        if self._rest_position is not None:
            return

        # Replay the passive fall once so every episode starts from the same
        # multi-contact, actuator-off crouch instead of an arbitrary drop height.
        for _ in range(REST_SETTLE_STEPS):
            self.scene.step()

        self._rest_position = self._as_batch(self.robot.get_pos()).clone()
        self._rest_quaternion = self._as_batch(self.robot.get_quat()).clone()
        self._rest_dof_positions = self.actuator_manager.get_dofs_position().clone()
        self.entity_manager.step()

        if self.actuator_manager.num_dofs != len(STAND_POSE):
            raise ValueError(
                f"Expected {len(STAND_POSE)} actuated joints, "
                f"found {self.actuator_manager.num_dofs}: "
                f"{self.actuator_manager.dofs_names}"
            )

    def reset(
        self, env_ids: torch.Tensor | Sequence[int] | None = None
    ) -> tuple[torch.Tensor | None, dict]:
        observations_batch, info = super().reset(env_ids)
        if self._rest_position is None:
            return observations_batch, info

        if env_ids is None:
            reset_ids = self.all_envs_idx
        elif isinstance(env_ids, torch.Tensor):
            reset_ids = env_ids.to(device=gs.device, dtype=torch.long)
        else:
            reset_ids = torch.as_tensor(env_ids, device=gs.device, dtype=torch.long)

        self.robot.set_pos(
            self._rest_position[reset_ids],
            envs_idx=reset_ids,
            zero_velocity=True,
        )
        self.robot.set_quat(
            self._rest_quaternion[reset_ids],
            envs_idx=reset_ids,
            zero_velocity=True,
        )
        self.robot.set_dofs_position(
            self._rest_dof_positions[reset_ids],
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
