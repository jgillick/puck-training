"""Genesis Forge environment for teaching Puck to stand and balance."""

import math, pathlib, enum, dataclasses, typing
from collections.abc import Sequence
import numpy as np
import torch
import genesis as gs
from genesis_forge import ManagedEnvironment
from genesis.utils import geom as gs_util_geom
from genesis_forge.managers import (
    ActuatorManager,
    BaseActionManager,
    CommandManager,
    EntityManager,
    ObservationManager,
    PositionActionManager,
    RewardManager,
    TerrainManager,
    TerminationManager,
    VelocityCommandManager,
)
from genesis_forge.mdp import observations, rewards, terminations

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
ROBOT_MODEL_PATH = PROJECT_ROOT / "robot.xml"
CONTROL_DT = 1 / 50
INITIAL_ROOT_POSITION = (0.0, 0.0, 0.23) #(x,y,z) height (m) offset caused by STANDING_JOINT_POSITIONS
IDENTITY_QUATERNION = (1.0, 0.0, 0.0, 0.0)
INITIAL_LIMB_KP = 5.0
INITIAL_LIMB_KD = 1.0
WHEEL_JOINT_NAMES = ("wheel_r", "wheel_l")
STANDING_JOINT_POSITIONS = {
    "flexor_r": 0.0,
    "hip_r": math.pi / 4,
    "hip_l": -math.pi / 4,
    "knee_r": math.pi / 2,
    "knee_l": -math.pi / 2,
    "flexor_l": 0.0,
}
CURRICULUM_CHECK_EVERY_STEPS = 600
CURRICULUM_AVG_SAMPLES = 5
CURRICULUM_MIN_EPISODE_LENGTH_RATIO = 0.6
MIN_TRACKING_THRESHOLD = 0.8 #threshold for tracking performance to advance curriculum
HEIGHT_COMMAND_MINMAX = (0.1, 0.5)
HEIGHT_MID = INITIAL_ROOT_POSITION[2]
INITIAL_FORWARD_VELOCITY = 0.3
MAX_FORWARD_VELOCITY = 2.0 #m/s
MAX_SIDE_VELOCITY = 1.0 #m/s
MAX_YAW_VELOCITY = 5.0 #rad/s
TRAINING_VELOCITY_INCREMENTS = 0.2
VELOCITY_TRACKING_SENSITIVITY = 0.2 #m/s
HEIGHT_TRACKING_SENSITIVITY = 0.1 #m
YAW_TRACKING_SENSITIVITY = 0.2 # rad/s
# Staging:
class Stage(enum.IntEnum):
    INITIAL = 0
    ADD_NOISE = 1
    FWD_CMD = 2
    FWDBACK_CMD = 3
    YAW_COMMAND = 4
    SIDE_VELOCITY = 7
    VARIABLE_HEIGHT = 10
    VARIABLE_ATTITUDE = 12
    ROUGH_TERRAIN = 16
    Max = 17

MAX_CURRICULUM_LEVEL = Stage.Max + math.ceil((MAX_FORWARD_VELOCITY - INITIAL_FORWARD_VELOCITY) / TRAINING_VELOCITY_INCREMENTS)
ATTITUDE_TARGET_LIMIT = 0.15
ATTITUDE_TRACKING_SCALE = 0.15
CURRICULUM_SPAWN_USABLE_RATIO = 0.2

@dataclasses.dataclass
class CurriculumMetrics:
    forward: float
    side: float
    yaw: float
    pitch: float
    roll: float
    height: float
    episode_length: float

    def tracks_baseline(self, threshold: float) -> list[str]:
        failing = []
        for field in dataclasses.fields(self):
            if field.name == 'episode_length': continue
            if getattr(self, field.name) <= threshold: failing.append(field.name)
        return failing

def ground_contact_failure(env) -> torch.Tensor:
    contacts = env.robot.get_contacts(with_entity=env.terrain, is_padded=True)
    wheel_ids = torch.tensor(
        [
            env.robot.get_link("wheel").idx,
            env.robot.get_link("wheel_2").idx,
        ],
        device=gs.device,
    )
    wheel_contacts = (
        (contacts["link_a"].unsqueeze(-1) == wheel_ids).any(dim=-1)
        | (contacts["link_b"].unsqueeze(-1) == wheel_ids).any(dim=-1)
    )
    non_wheel_contacts = contacts["valid_mask"] & ~wheel_contacts
    return non_wheel_contacts.any(dim=1)


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
            # max_episode_random_scaling=0.3,
        )
        self.curriculum_level = Stage.INITIAL.value
        self.curriculum_samples: list[CurriculumMetrics] = []
        self.next_curriculum_check_step = CURRICULUM_CHECK_EVERY_STEPS
        self._last_reset_mean_ep_length = 0.0
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
        terrain_pixels = (np.indices((192, 384)) // 8).sum(axis=0) % 2
        terrain_texture = np.empty((192, 384, 3), dtype=np.uint8)
        terrain_texture[terrain_pixels == 0] = (62, 74, 80)
        terrain_texture[terrain_pixels == 1] = (84, 96, 100)
        self.terrain = self.scene.add_entity(
            surface=gs.surfaces.Default(
                diffuse_texture=gs.textures.ImageTexture(
                    image_array=terrain_texture
                )
            ),
            morph=gs.morphs.Terrain(
                n_subterrains=(1, 2),
                subterrain_size=(48, 24),
                subterrain_types=[["flat_terrain", "random_uniform_terrain"]],
                subterrain_parameters={
                    "random_uniform_terrain": {
                        "min_height": 0.0,
                        "max_height": 0.08,
                        "step": 0.04,
                        "downsampled_scale": 0.25,
                    },
                },
            ),
        )
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file=str(ROBOT_MODEL_PATH),
                pos=INITIAL_ROOT_POSITION,
                quat=IDENTITY_QUATERNION,
            )
        )
        robot_entity = getattr(self.robot, "entity", self.robot)
        body_link = robot_entity.get_link("body")
        self.imu = self.scene.add_sensor(
            gs.sensors.IMU(
                entity_idx=robot_entity.idx,
                link_idx_local=body_link.idx_local,
                pos_offset=(0.141812, -0.0107074, 0.0723044),
                acc_noise=(0.02, 0.02, 0.02),
                gyro_noise=(0.005, 0.005, 0.005),
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
        self.periodic_outfile : pathlib.Path | None = None

    def config(self) -> None:
        """Configure standing-pose actions, commands, rewards, and observations."""
        self.terrain_manager = TerrainManager(self, terrain=self.terrain)

        def reset_pose(env, entity, envs_idx) -> None:
            pos = self.terrain_manager.generate_random_env_pos(
                envs_idx=envs_idx,
                usable_ratio=(CURRICULUM_SPAWN_USABLE_RATIO if self.curriculum_level >= Stage.ADD_NOISE else 0.0),
                subterrain=self.curriculum_terrain,
                height_offset=INITIAL_ROOT_POSITION[2],
            )
            entity.set_pos(pos, envs_idx=envs_idx, zero_velocity=True)
            quat = torch.tensor(
                IDENTITY_QUATERNION, device=gs.device, dtype=gs.tc_float
            ).repeat(len(envs_idx), 1)
            entity.set_quat(quat, envs_idx=envs_idx, zero_velocity=True)

        cmd_scale = torch.tensor(
            [1/MAX_SIDE_VELOCITY, 1/MAX_FORWARD_VELOCITY, 1/MAX_YAW_VELOCITY],
            device=gs.device
        )

        self.entity_manager = EntityManager(
            self,
            entity=self.robot,
            on_reset={
                "pose": {
                    "fn": reset_pose
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
            actuator_joints=list(STANDING_JOINT_POSITIONS.keys()),
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
                "lin_vel_x": (0.0, 0.0),
                "lin_vel_y": (-INITIAL_FORWARD_VELOCITY, -INITIAL_FORWARD_VELOCITY), #start training with initial small forward velocity
                "ang_vel_z": (0.0, 0.0),
            },
        )
        self.height_command = CommandManager(self, range=HEIGHT_COMMAND_MINMAX) #fixed target at initial height
        self.attitude_command = CommandManager(
            self, range={"pitch": (0.0, 0.0), "roll": (0.0, 0.0)}
        )

        def square_linear_error(error: torch.Tensor, scale: float) -> torch.Tensor:
            '''exponential reward with additial linear penalty to provide gradient pressure for large errors'''
            return torch.exp(-(error / scale).square()) - 0.1 * (error.abs() / scale)

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

        def forward_velocity_tracking(env) -> torch.Tensor:
            target = self.velocity_command.command[:, 1]
            actual = self.entity_manager.get_linear_velocity()[:, 1]
            return square_linear_error(-(actual - target), VELOCITY_TRACKING_SENSITIVITY)

        def side_velocity_tracking(env) -> torch.Tensor:
            target = self.velocity_command.command[:, 0]
            actual = self.entity_manager.get_linear_velocity()[:, 0]
            return square_linear_error(actual - target, VELOCITY_TRACKING_SENSITIVITY)

        def yaw_velocity_tracking(env) -> torch.Tensor:
            target = self.velocity_command.command[:, 2]
            actual = self.entity_manager.get_angular_velocity()[:, 2]
            return square_linear_error(actual - target, YAW_TRACKING_SENSITIVITY)

        def attitude_tracking(axis: str):
            axis_index = {"roll": 0, "pitch": 1}[axis]

            def reward(env) -> torch.Tensor:
                body_rpy = gs_util_geom.quat_to_xyz(
                    self.entity_manager.base_quat, rpy=True
                )
                target = self.attitude_command.get_command(axis).reshape(-1)
                error = body_rpy[:, axis_index] - target
                assert isinstance(error, torch.Tensor)
                error = torch.atan2(torch.sin(error), torch.cos(error))
                return torch.exp(-(error / ATTITUDE_TRACKING_SCALE).square())

            return reward

        def height_tracking(env) -> torch.Tensor:
            target = self.height_command.command.reshape(-1)
            actual = self.entity_manager.base_pos[:, 2] - self.terrain_manager.get_terrain_height(self.entity_manager.base_pos[:, 0], self.entity_manager.base_pos[:, 1])
            return square_linear_error(-(actual - target), HEIGHT_TRACKING_SENSITIVITY)

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


        self.reward_manager = RewardManager(
            self,
            cfg={
                "upright": {
                    "weight": -5.0,
                    "fn": rewards.flat_orientation_l2(
                        entity_manager=self.entity_manager
                    ),
                },
                "yaw_velocity":         { "weight": 1.0, "fn": yaw_velocity_tracking },
                "forward_velocity":     { "weight": 2.0, "fn": forward_velocity_tracking },
                "side_velocity":        { "weight": 1.0, "fn": side_velocity_tracking },
                "pitch_tracking":       { "weight": 1.0, "fn": attitude_tracking("pitch") },
                "roll_tracking":        { "weight": 1.0, "fn": attitude_tracking("roll") },
                "leg_symmetry":         { "weight": 1.0, "fn": hip_knee_symmetry_reward },
                "hip_flexor_alignment": { "weight": 1.5, "fn": hip_flexor_alignment },
                "height_tracking":      { "weight": 3.0, "fn": height_tracking },
                "body_angular_velocity": {
                    "weight": -0.02,
                    "fn": rewards.ang_vel_xy_l2(entity_manager=self.entity_manager),
                },
                # penalize high-frequency actions to encourage smooth control
                "action_rate_limbs":    {"weight": -0.02, "fn": rewards.action_rate_l2(action_manager=self.action_manager)},
                "action_rate_wheels":   {"weight": -0.05,  "fn": rewards.action_rate_l2(action_manager=self.wheel_action_manager)},
                "limb_dof_vel":         {"weight": -1e-3, "fn": rewards.dof_velocity_l2(action_manager=self.action_manager)},
                "action_acceleration": {
                    "weight": -2e-04,
                    "fn": rewards.action_acceleration_l2(),
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
        def imu_observation(env):
            data = self.imu.read()
            return torch.cat((
                data.ang_vel * 0.2,
                data.lin_acc / 9.81,
            ), dim=-1)

        HEIGHT_COMMAND_MID = (HEIGHT_COMMAND_MINMAX[1] + HEIGHT_COMMAND_MINMAX[0]) / 2
        HEIGHT_RANGE_SCALE = (HEIGHT_COMMAND_MINMAX[1] - HEIGHT_COMMAND_MINMAX[0]) / 2

        self.observation_manager = ObservationManager(
            self,
            cfg={
                "imu": {
                    "fn": imu_observation,
                    "clip": (-2.0, 2.0),
                },
                # use action_manager for joint only observations, not wheel joints. then add wheel velocity obs separately
                "joint_position": {"fn": lambda env: self.action_manager.get_dofs_position(), "scale": 1/math.pi, "clip": (-1, 1)},
                "joint_velocity": {"fn": lambda env: self.action_manager.get_dofs_velocity(), "scale": 0.01, "clip": (-1, 1)},
                "wheel_velocity": {"fn": lambda env: self.wheel_action_manager.get_dofs_velocity(), "scale": 0.03, "clip": (-1, 1)},
                # TODO: all motor drive torques!?
                "height_command": {
                    "fn": lambda env: (self.height_command.command - HEIGHT_COMMAND_MID) / HEIGHT_RANGE_SCALE,
                    "clip": (-1.0, 1.0),
                    "units": "normalized",
                },
                "attitude_command": {
                    "fn": self.attitude_command.observation,
                    "scale": 1.0 / ATTITUDE_TARGET_LIMIT,
                    "clip": (-1.0, 1.0),
                    "units": "normalized",
                },
                "velocity_command": {
                    "fn": lambda env: self.velocity_command.observation(env) * cmd_scale,
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
        self._apply_curriculum_stage()

    @property
    def curriculum_terrain(self) -> str:
        return "random_uniform_terrain" if self.curriculum_level >= Stage.ROUGH_TERRAIN else "flat_terrain"

    def _apply_curriculum_stage(self) -> None:
        directional = self.curriculum_level >= Stage.FWD_CMD
        yaw_enabled = self.curriculum_level >= Stage.YAW_COMMAND
        sideways = self.curriculum_level >= Stage.SIDE_VELOCITY

        forward_velocity_limit = min(MAX_FORWARD_VELOCITY, INITIAL_FORWARD_VELOCITY + max(0, self.curriculum_level - Stage.FWD_CMD) * TRAINING_VELOCITY_INCREMENTS)
        reverse_velocity_limit = min(INITIAL_FORWARD_VELOCITY, max(0, self.curriculum_level - Stage.FWDBACK_CMD + 1) * TRAINING_VELOCITY_INCREMENTS)
        side_velocity_limit = min(
            MAX_SIDE_VELOCITY,
            max(0, self.curriculum_level - Stage.SIDE_VELOCITY + 1) * TRAINING_VELOCITY_INCREMENTS
        )
        yaw_velocity_limit = min(
            MAX_YAW_VELOCITY,
            max(0, self.curriculum_level - Stage.YAW_COMMAND + 1) * TRAINING_VELOCITY_INCREMENTS
        )
        self.velocity_command.range = {
            "lin_vel_x": (-side_velocity_limit, side_velocity_limit) if sideways else (0.0, 0.0),
            "lin_vel_y": (-forward_velocity_limit, reverse_velocity_limit) if directional else (-INITIAL_FORWARD_VELOCITY, -INITIAL_FORWARD_VELOCITY),
            "ang_vel_z": (-yaw_velocity_limit, yaw_velocity_limit) if yaw_enabled else (0.0, 0.0),
        }
        self.velocity_command.resample_time_sec = 2.0
        self.velocity_command.resample_command(self.all_envs_idx)

        variable_attitude = self.curriculum_level >= Stage.VARIABLE_ATTITUDE
        attitude_range = (
            (-ATTITUDE_TARGET_LIMIT, ATTITUDE_TARGET_LIMIT)
            if variable_attitude
            else (0.0, 0.0)
        )
        self.attitude_command.range = {
            "pitch": attitude_range,
            "roll": attitude_range,
        }
        self.attitude_command.resample_time_sec = (
            2.0 if variable_attitude else None
        )
        self.attitude_command.resample_command(self.all_envs_idx)

        variable_height = self.curriculum_level >= Stage.VARIABLE_HEIGHT
        self.height_command.range = HEIGHT_COMMAND_MINMAX if variable_height else (HEIGHT_MID, HEIGHT_MID)
        self.height_command.resample_time_sec = 2.0 if variable_height else None
        self.height_command.resample_command(self.all_envs_idx)
        if self.curriculum_level >= Stage.YAW_COMMAND and (sym := self.reward_manager["leg_symmetry"]).weight > 0.0:
            self.log(f"relaxed legs enabled")
            sym.weight = 0.0
            self.reward_manager["hip_flexor_alignment"].weight  = 0.0

    def _add_observation_noise(
        self, value: torch.Tensor, standard_deviation: float
    ) -> torch.Tensor:
        if self.curriculum_level < 1:
            return value
        return value + torch.randn_like(value) * standard_deviation

    def _curriculum_metrics(self) -> CurriculumMetrics:
        return CurriculumMetrics(
            forward = self.reward_manager.last_episode_mean_reward("forward_velocity", before_weight=True),
            side =    self.reward_manager.last_episode_mean_reward("side_velocity", before_weight=True),
            yaw =     self.reward_manager.last_episode_mean_reward("yaw_velocity", before_weight=True),
            pitch =   self.reward_manager.last_episode_mean_reward("pitch_tracking", before_weight=True),
            roll =    self.reward_manager.last_episode_mean_reward("roll_tracking", before_weight=True),
            height =  self.reward_manager.last_episode_mean_reward("height_tracking", before_weight=True),
            episode_length=self._last_reset_mean_ep_length,
        )

    def _update_curriculum(self) -> None:
        if self.curriculum_level >= MAX_CURRICULUM_LEVEL:
            return
        if self.step_count < self.next_curriculum_check_step:
            return

        self.curriculum_samples.append(self._curriculum_metrics())
        if len(self.curriculum_samples) < CURRICULUM_AVG_SAMPLES:
            return

        average_values = torch.tensor(
            [dataclasses.astuple(sample) for sample in self.curriculum_samples]
        ).mean(dim=0).tolist()
        averages: CurriculumMetrics = CurriculumMetrics(*average_values)

        enough_survival = averages.episode_length >= (CURRICULUM_MIN_EPISODE_LENGTH_RATIO * (self.max_episode_length_steps or 0))
        track_fails = averages.tracks_baseline(MIN_TRACKING_THRESHOLD)

        if not track_fails and enough_survival:
            self.curriculum_level += 1
            self._apply_curriculum_stage()
            self.log(f"Curriculum BUMP avgs: {averages}")
        else:
            self.log(f"Curriculum hold: track-{'ok' if not track_fails else ','.join(track_fails)} len {averages.episode_length:0.1f} avgs: {averages}")
        self.next_curriculum_check_step = (self.step_count + CURRICULUM_CHECK_EVERY_STEPS)
        self.curriculum_samples.clear()

    def step(self, actions: torch.Tensor):
        observations_batch, reward, terminated, truncated, extras = super().step(actions)
        episode = extras.setdefault("episode", {})
        episode["Curriculum/level"] = self.curriculum_level

        def rval(range, key: str | None = None, usemin = True):
            if isinstance(range, dict) and key is not None: range = range[key]
            if isinstance(range, (tuple, list)) and len(range) == 2: return float(range[0]) if usemin else range[1]
            print(f"unexpected range type {type(range)} value {range} key {key}")
            return 0.0

        # report command max/min values if enabled
        if self.curriculum_level >= Stage.FWD_CMD:
            episode["Curriculum/velcmd max"] = rval(self.velocity_command.range, 'lin_vel_y')
        if self.curriculum_level >= Stage.YAW_COMMAND:
            episode["Curriculum/yaw max"] = rval(self.velocity_command.range, "ang_vel_z")
        if self.curriculum_level >= Stage.SIDE_VELOCITY:
            episode["Curriculum/side max"] = rval(self.velocity_command.range, "lin_vel_x")
        if self.curriculum_level >= Stage.VARIABLE_HEIGHT:
            episode["Curriculum/height max"] = rval(tuple(self.height_command.range))
        if self.curriculum_level >= Stage.VARIABLE_ATTITUDE:
            episode["Curriculum/pitch max"] = rval(self.attitude_command.range, 'pitch')
            episode["Curriculum/roll max"] = rval(self.attitude_command.range, 'roll')

        actual_height = (self.entity_manager.base_pos[:, 2] - self.terrain_manager.get_terrain_height(self.entity_manager.base_pos[:, 0], self.entity_manager.base_pos[:, 1]))
        episode["Diagnostics/height"] = actual_height.mean().item()
        episode["Diagnostics/height_target"] = self.height_command.command.mean().item()
        episode["Diagnostics/height_error"] = (actual_height - self.height_command.command.reshape(-1)).mean().item()
        return observations_batch, reward, terminated, truncated, extras

    def build(self) -> None:
        super().build()
        if self.actuator_manager.num_dofs != 8:
            raise ValueError(
                "Expected 8 actuated joints, "
                f"found {self.actuator_manager.num_dofs}: "
                f"{self.actuator_manager.dofs_names}"
            )

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None):
        if env_ids is not None:
            reset_ids_for_metrics = (
                env_ids
                if isinstance(env_ids, torch.Tensor)
                else torch.as_tensor(env_ids, device=gs.device, dtype=torch.long)
            )
            if len(reset_ids_for_metrics) > 0:
                self._last_reset_mean_ep_length = (
                    self.episode_length[reset_ids_for_metrics].float().mean().item()
                )
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

        if env_ids is not None and len(reset_ids) > 0:
            self._update_curriculum()
        if env_ids is None:
            observations_batch = self.get_observations()
        return observations_batch, info

    @staticmethod
    def _as_batch(value: torch.Tensor) -> torch.Tensor:
        return value.unsqueeze(0) if value.ndim == 1 else value

    def stagename(self) -> str:
        try:
            return Stage(min(self.curriculum_level, Stage.Max)).name + f"({self.curriculum_level})"
        except ValueError:
            return f"Stage({self.curriculum_level})"
    def logbase(self, * args, **kwargs) -> None:
        if self.periodic_outfile:
            with open(self.periodic_outfile, 'a') as f:
                print(*args, **kwargs, file=f)
        print(*args, **kwargs)
    def log(self, * args, **kwargs) -> None:
        # report learning iteration
        self.logbase(f"training@[c{self.stagename()}|{self.curriculum_level} s{self.step_count}] ", *args, **kwargs)

