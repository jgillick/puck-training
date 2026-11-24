import math, pathlib, enum, dataclasses, torch, collections.abc, numpy, typing, random
import genesis as gs
from genesis_forge import ManagedEnvironment
from genesis.utils import geom as gs_util_geom
import genesis_forge.managers as gf_mgrs, genesis_forge.mdp as gf_mdp

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
ROBOT_MODEL_PATH = PROJECT_ROOT / "robot.xml"
CONTROL_DT = 1 / 50
INITIAL_ROOT_POSITION = (0.0, 0.0, 0.23) #(x,y,z) height (m) offset caused by STANDING_JOINT_POSITIONS
IDENTITY_QUATERNION = (1.0, 0.0, 0.0, 0.0)
LIMB_KP, LIMB_KD = 15.0, 1.0
WHEEL_JOINT_NAMES = ("wheel_r", "wheel_l")
STANDING_JOINT_POSITIONS = {
    "flexor_r": 0.0,
    "hip_r": math.pi / 4,
    "hip_l": -math.pi / 4,
    "knee_r": math.pi / 2,
    "knee_l": -math.pi / 2,
    "flexor_l": 0.0,
}
CURRICULUM_CHECK_EVERY_STEPS = 400
CURRICULUM_MIN_EPISODE_LENGTH_RATIO = 0.6
MIN_TRACKING_THRESHOLD = 0.8 #threshold for tracking performance to advance curriculum
HEIGHT_COMMAND_MINMAX = (0.1, 0.5)
HEIGHT_MID = INITIAL_ROOT_POSITION[2]
INITIAL_FORWARD_VELOCITY = 0.3
MAX_FORWARD_VELOCITY = 2.0 #m/s
MAX_SIDE_VELOCITY = 0.4 #m/s
MAX_YAW_VELOCITY = 3.0 #rad/s
MAX_VEL_ACCEL = ( 1.0, 1.0, 0.4 )  # m/s^2, m/s^2, rad/s^2
MAX_HEIGHT_ACCEL = 0.1 #m/s^2
MAX_ATTITUDE_ACCEL = 0.1 #rad/s^2
VEL_CMD_SCALES = (1 / 2.0, 1 / 10.0, 1 / 10.0) #allow headroom in scaling
VELOCITY_TRACKING_SENSITIVITY = 0.2 #m/s
YAW_TRACKING_SENSITIVITY = 0.2 # rad/s
# Staging:
class Stage(enum.IntEnum):
    INITIAL = 0
    PITCH = enum.auto()
    ROLL = enum.auto()
    HEIGHT = enum.auto()
    FORWARD = enum.auto()
    FWDBACK = enum.auto()
    YAW = enum.auto()
    NOISE = enum.auto()
    TERRAIN = enum.auto()
    SIDE = enum.auto()
    Max = SIDE

ATTITUDE_TARGET_LIMIT = 45 * math.pi / 180 #degrees
CURRICULUM_SPAWN_USABLE_RATIO = 0.2
TERRAIN_SUB_LEVELS = (
    "flat_terrain",
    "random_uniform_terrain",
    "wave_terrain",
    "discrete_obstacles_terrain",
    "stepping_stones_terrain",
)
CURRICULUM_MAX_SAMPLES : typing.Final[int] = 40


@dataclasses.dataclass
class CurriculumMetrics:
    forward: float
    side: float
    yaw: float
    pitch: float
    roll: float
    height: float
    episode_length: float

    def tracks_baseline(self, thresholds: CurriculumMetrics) -> list[str]:
        failing = []
        for field in dataclasses.fields(self):
            if getattr(self, field.name) <= getattr(thresholds, field.name):
                failing.append(field.name)
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


class PuckWheelTorqueActionManager(gf_mgrs.BaseActionManager):
    """Map two normalized actions to bounded wheel torques."""
    def build(self) -> None:
        super().build()
        self._force_lower, self._force_upper = self.env.robot.get_dofs_force_range(self.dofs_idx)
    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        actions = actions.clamp(-1.0, 1.0)
        return torch.where(
            actions >= 0.0,
            actions * self._force_upper,
            actions * -self._force_lower,
        )
    def send_actions_to_simulation(self, actions: torch.Tensor) -> torch.Tensor:
        self.env.robot.control_dofs_force(actions, self.dofs_idx)
        return actions


class PuckStandEnv(ManagedEnvironment):
    """Train Puck to stand and balance with position-controlled limbs."""

    def __init__(self, num_envs: int, headless: bool = True, max_episode_length_sec: int = 10) -> None:
        super().__init__(
            num_envs=num_envs,
            dt=CONTROL_DT,
            max_episode_length_sec=max_episode_length_sec,
            # max_episode_random_scaling=0.3, #TODO, but throws error when enabled
        )
        self.curriculum_level : float = Stage.INITIAL.value #remainder is sub-level
        self.sub_level_step = 0.0 #default. each curriculum stage can change this, 0 or 1 disables
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
        terrain_pixels = (numpy.indices((192, 384)) // 8).sum(axis=0) % 2
        terrain_texture = numpy.empty((192, 384, 3), dtype=numpy.uint8)
        terrain_texture[terrain_pixels == 0] = (62, 74, 80)
        terrain_texture[terrain_pixels == 1] = (84, 96, 100)
        self.terrain = self.scene.add_entity(
            surface=gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_array=terrain_texture)),
            morph=gs.morphs.Terrain(
                n_subterrains=(1, len(TERRAIN_SUB_LEVELS)),
                subterrain_size=(48, 24),
                subterrain_types=[list(TERRAIN_SUB_LEVELS)],
                subterrain_parameters={
                    "random_uniform_terrain": {
                        "min_height": 0.0,
                        "max_height": 0.025,
                        "step": 0.01,
                        "downsampled_scale": 0.25,
                    },
                    "wave_terrain": {
                        "num_waves": 2.0,
                        "amplitude": 0.025,
                    },
                    "discrete_obstacles_terrain": {
                        "max_height": 0.04,
                        "min_size": 0.3,
                        "max_size": 1.0,
                        "num_rects": 30,
                    },
                    "stepping_stones_terrain": {
                        "stone_size": 0.8,
                        "stone_distance": 0.2,
                        "max_height": 0.04,
                        "platform_size": 2.0,
                    },
                },
            ),
        )
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file=str(ROBOT_MODEL_PATH), pos=INITIAL_ROOT_POSITION, quat=IDENTITY_QUATERNION)
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

    def get_stage(self) -> tuple[Stage, float]:
        stage : int = math.floor(self.curriculum_level)
        sub_level : float = round(self.curriculum_level - stage, 2)
        return Stage(stage), sub_level

    def config(self) -> None:
        """Configure standing-pose actions, commands, rewards, and observations."""
        self.terrain_manager = gf_mgrs.TerrainManager(self, terrain=self.terrain)

        def reset_pose(env, entity, envs_idx) -> None:
            pos = self.terrain_manager.generate_random_env_pos(
                envs_idx=envs_idx,
                usable_ratio=(CURRICULUM_SPAWN_USABLE_RATIO if self.curriculum_level >= Stage.NOISE else 0.0),
                subterrain=self.curriculum_terrain,
                height_offset=INITIAL_ROOT_POSITION[2],
            )
            entity.set_pos(pos, envs_idx=envs_idx, zero_velocity=True)
            quat = torch.tensor(
                IDENTITY_QUATERNION, device=gs.device, dtype=gs.tc_float
            ).repeat(len(envs_idx), 1)
            entity.set_quat(quat, envs_idx=envs_idx, zero_velocity=True)

        cmd_scale = torch.tensor(
            VEL_CMD_SCALES,
            device=gs.device
        )

        self.entity_manager = gf_mgrs.EntityManager(
            self,
            entity=self.robot,
            on_reset={
                "pose": { "fn": reset_pose } #type: ignore
            },
        )
        self.actuator_manager = gf_mgrs.ActuatorManager(self, joint_names=".*", default_pos=STANDING_JOINT_POSITIONS, kp=LIMB_KP, kv=1.0, damping=LIMB_KD)
        self.action_manager = gf_mgrs.PositionActionManager(
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
        self.velocity_command = gf_mgrs.VelocityCommandManager(
            self,
            stopped_probability=0.1,
            range={
                "lin_vel_x": (0.0, 0.0),
                "lin_vel_y": (-INITIAL_FORWARD_VELOCITY, -INITIAL_FORWARD_VELOCITY), #start training with initial small forward velocity
                "ang_vel_z": (0.0, 0.0),
            },
            resample_time_sec=5.0,
            debug_visualizer=True,
        )
        self._cmd_slew_max = torch.tensor(MAX_VEL_ACCEL, device=gs.device, dtype=gs.tc_float)

        self.height_command = gf_mgrs.CommandManager(self, range=HEIGHT_COMMAND_MINMAX) #fixed target at initial height
        self.attitude_command = gf_mgrs.CommandManager(
            self, range={"pitch": (0.0, 0.0), "roll": (0.0, 0.0)}
        )
        self.cmd_smooth = self.velocity_command.command.clone()
        self.height_smooth = self.height_command.command[:, 0].clone()
        self.pitch_smooth = self.attitude_command.command[:, 0].clone()
        self.roll_smooth = self.attitude_command.command[:, 1].clone()
        self.tracking_target : torch.Tensor | None = None
        self.tracking_result : torch.Tensor | None = None

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
            tolerance = math.pi / 15 #12deg
            excess = torch.relu(max_flexor_angle - tolerance)
            return torch.exp(-4.0 * (excess / tolerance).square())

        def forward_velocity_tracking(env) -> torch.Tensor:
            target = self.cmd_smooth[:, 1]
            actual = self.entity_manager.get_linear_velocity()[:, 1]
            if (stg := self.get_stage()[0]) == Stage.FORWARD or stg == Stage.FWDBACK:
                self.tracking_target = target
                self.tracking_result = actual
            return square_linear_error(-(actual - target), VELOCITY_TRACKING_SENSITIVITY)

        def side_velocity_tracking(env) -> torch.Tensor:
            target = self.cmd_smooth[:, 0]
            actual = self.entity_manager.get_linear_velocity()[:, 0]
            if self.get_stage()[0] == Stage.SIDE:
                self.tracking_target = target
                self.tracking_result = actual
            return square_linear_error(actual - target, VELOCITY_TRACKING_SENSITIVITY)

        def yaw_velocity_tracking(env) -> torch.Tensor:
            target = self.cmd_smooth[:, 2]
            actual = self.entity_manager.get_angular_velocity()[:, 2]
            if self.get_stage()[0] == Stage.YAW:
                self.tracking_target = target
                self.tracking_result = actual
            return square_linear_error(actual - target, YAW_TRACKING_SENSITIVITY)

        def pitch_tracking(env) -> torch.Tensor:
            body_rpy = gs_util_geom.quat_to_xyz(self.entity_manager.base_quat, rpy=True)
            actual = body_rpy[:, 1]
            return square_linear_error(actual - self.pitch_smooth, 0.2)
            if self.get_stage()[0] == Stage.PITCH:
                self.tracking_target = target
                self.tracking_result = actual
        def roll_tracking(env) -> torch.Tensor:
            body_rpy = gs_util_geom.quat_to_xyz(self.entity_manager.base_quat, rpy=True)
            actual = body_rpy[:, 0]
            return square_linear_error(actual - self.roll_smooth, 0.2)
            if self.get_stage()[0] == Stage.ROLL:
                self.tracking_target = target
                self.tracking_result = actual

        def height_tracking(env) -> torch.Tensor:
            actual = self.entity_manager.base_pos[:, 2] - self.terrain_manager.get_terrain_height(self.entity_manager.base_pos[:, 0], self.entity_manager.base_pos[:, 1])
            return torch.exp(-(actual - self.height_smooth).abs() / 0.05) # Laplace: slope doesn't vanish near 0
            if self.get_stage()[0] == Stage.HEIGHT:
                self.tracking_target = target
                self.tracking_result = actual

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
                [ positions[:, dof_indices[left]] - sign * positions[:, dof_indices[right]] for left, right, sign in mirrored_joints ],
                dim=1,
            )
            return torch.exp(-errors.square().mean(dim=1))


        self.reward_manager = gf_mgrs.RewardManager(
            self,
            cfg={
                "upright": {
                    "weight": -5.0,
                    "fn": gf_mdp.rewards.flat_orientation_l2(
                        entity_manager=self.entity_manager
                    ),
                },
                "yaw_velocity":         { "weight": 1.0, "fn": yaw_velocity_tracking },
                "forward_velocity":     { "weight": 2.0, "fn": forward_velocity_tracking },
                "side_velocity":        { "weight": 1.0, "fn": side_velocity_tracking },
                "pitch_tracking":       { "weight": 1.0, "fn": pitch_tracking },
                "roll_tracking":        { "weight": 1.0, "fn": roll_tracking },
                "leg_symmetry":         { "weight": 1.0, "fn": hip_knee_symmetry_reward },
                "hip_flexor_alignment": { "weight": 1.5, "fn": hip_flexor_alignment },
                "height_tracking":      { "weight": 3.0, "fn": height_tracking },
                # penalize high-frequency actions to encourage smooth control
                "action_rate_limbs":    {"weight": -0.01, "fn": gf_mdp.rewards.action_rate_l2(action_manager=self.action_manager)},
                "action_rate_wheels":   {"weight": -0.01,  "fn": gf_mdp.rewards.action_rate_l2(action_manager=self.wheel_action_manager)},
            },
        )
        gf_mgrs.TerminationManager(
            self,
            term_cfg={
                "ground_contact": {"fn": ground_contact_failure},
                "timeout": {"fn": gf_mdp.terminations.timeout(), "time_out": True},
                "bad_orientation": {
                    "fn": gf_mdp.terminations.bad_orientation(
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

        torque_scales: torch.Tensor | None = None
        def joint_torque_observation(env):
            nonlocal torque_scales
            if torque_scales is None:
                force_lower, force_upper = self.robot.get_dofs_force_range(
                    self.actuator_manager.dofs_idx
                )
                torque_scales = 1.0 / torch.maximum(
                    force_lower.abs(), force_upper.abs()
                ).clamp_min(1e-6)
                print(f"Torque scales: {torque_scales} for dofs {self.actuator_manager.dofs_names}")

            return self.actuator_manager.get_dofs_force() * torque_scales

        self.observation_manager = gf_mgrs.ObservationManager(
            self,
            cfg={
                "imu": { "fn": imu_observation, "clip": (-2.0, 2.0) },
                "joint_position": {"fn": lambda env: self.action_manager.get_dofs_position(), "scale": 1/math.pi, "clip": (-1, 1)},
                "joint_velocity": {"fn": lambda env: self.action_manager.get_dofs_velocity(), "scale": 0.01, "clip": (-1, 1)},
                "wheel_velocity": {"fn": lambda env: self.wheel_action_manager.get_dofs_velocity(), "scale": 0.03, "clip": (-1, 1)},
                "joint_torque":   {"fn": joint_torque_observation, "clip": (-1.0, 1.0) },
                "height_command": {
                    "fn": lambda env: ((self.height_smooth - HEIGHT_COMMAND_MID) / HEIGHT_RANGE_SCALE).unsqueeze(-1),
                    "clip": (-1.0, 1.0),
                    "units": "normalized",
                },
                "pitch_command": {"fn": lambda env: (self.pitch_smooth / ATTITUDE_TARGET_LIMIT).unsqueeze(-1), "clip": (-1.0, 1.0), "units": "normalized"},
                "roll_command": {"fn": lambda env: (self.roll_smooth / ATTITUDE_TARGET_LIMIT).unsqueeze(-1), "clip": (-1.0, 1.0), "units": "normalized"},
                "velocity_command": { "fn": lambda env: self.cmd_smooth * cmd_scale },
                "actions": { "fn": gf_mdp.observations.current_actions(), "clip": (-1.0, 1.0) },
            },
            history_len=5,
            noise=0.0,
        )
        self._apply_curriculum_stage()

    @property
    def curriculum_terrain(self) -> str:
        stage, sub_level = self.get_stage()
        if stage < Stage.TERRAIN:
            return TERRAIN_SUB_LEVELS[0]
        elif stage > Stage.TERRAIN: #choose random level
            sub_level = random.random()
        terrain_idx = min(
            int(sub_level * len(TERRAIN_SUB_LEVELS) + 1e-6),
            len(TERRAIN_SUB_LEVELS) - 1,
        )
        return TERRAIN_SUB_LEVELS[terrain_idx]

    def _apply_curriculum_stage(self) -> None:
        # set each limit to the initial
        lim_vel_fwd = INITIAL_FORWARD_VELOCITY
        lim_vel_bck = 0
        lim_vel_yaw = 0
        lim_pitch = 0
        lim_roll = 0
        lim_height = (HEIGHT_MID, HEIGHT_MID)
        lim_vel_side = 0
        stage, sub_level = self.get_stage()

        # for each stage- map self.sub_level to expand the range
        if self.curriculum_level >= Stage.FORWARD:
            # map lim_vel_fwd between INITIAL_FORWARD_VELOCITY and MAX_FORWARD_VELOCITY
            lim_vel_fwd = INITIAL_FORWARD_VELOCITY + sub_level * (MAX_FORWARD_VELOCITY - INITIAL_FORWARD_VELOCITY) if (stage == Stage.FORWARD) else MAX_FORWARD_VELOCITY
        if self.curriculum_level >= Stage.FWDBACK:
            lim_vel_bck = sub_level * MAX_FORWARD_VELOCITY if (stage == Stage.FWDBACK) else MAX_FORWARD_VELOCITY
        if self.curriculum_level >= Stage.YAW:
            lim_vel_yaw = sub_level * MAX_YAW_VELOCITY if (stage == Stage.YAW) else MAX_YAW_VELOCITY
        if self.curriculum_level >= Stage.SIDE:
            lim_vel_side = sub_level * MAX_SIDE_VELOCITY if (stage == Stage.SIDE) else MAX_SIDE_VELOCITY
        if self.curriculum_level >= Stage.PITCH:
            lim_pitch = sub_level * ATTITUDE_TARGET_LIMIT if (stage == Stage.PITCH) else ATTITUDE_TARGET_LIMIT
        if self.curriculum_level >= Stage.ROLL:
            lim_roll = sub_level * ATTITUDE_TARGET_LIMIT if (stage == Stage.ROLL) else ATTITUDE_TARGET_LIMIT
        if self.curriculum_level >= Stage.HEIGHT:
            if (stage == Stage.HEIGHT):
                lim_height = (
                    HEIGHT_MID - sub_level * (HEIGHT_MID - HEIGHT_COMMAND_MINMAX[0]),
                    HEIGHT_MID + sub_level * (HEIGHT_COMMAND_MINMAX[1] - HEIGHT_MID)
                )
            else: lim_height = HEIGHT_COMMAND_MINMAX
        self.sub_level_step = 0.2 #default, 5 levels
        if stage == Stage.TERRAIN:
            self.sub_level_step = 1.0 / len(TERRAIN_SUB_LEVELS)
        elif stage == Stage.INITIAL or stage == Stage.NOISE:
            self.sub_level_step = 0.0 #one big step
        elif stage == Stage.HEIGHT:
            self.sub_level_step = 0.5 #two steps

        # apply limits
        self.velocity_command.range = {
            "lin_vel_x": (-lim_vel_side, lim_vel_side),
            "lin_vel_y": (-lim_vel_fwd, lim_vel_bck),
            "ang_vel_z": (-lim_vel_yaw, lim_vel_yaw),
        }
        self.attitude_command.range = {
            "pitch": (-lim_pitch, lim_pitch),
            "roll": (-lim_roll, lim_roll),
        }
        self.height_command.range = lim_height

    def _add_observation_noise(self, value: torch.Tensor, standard_deviation: float) -> torch.Tensor:
        if self.curriculum_level < Stage.NOISE:
            return value
        return value + torch.randn_like(value) * standard_deviation

    def _curriculum_metrics(self) -> CurriculumMetrics:
        return CurriculumMetrics(
            forward  = self.reward_manager.last_episode_mean_reward("forward_velocity", before_weight=True),
            side     = self.reward_manager.last_episode_mean_reward("side_velocity", before_weight=True),
            yaw      = self.reward_manager.last_episode_mean_reward("yaw_velocity", before_weight=True),
            pitch    = self.reward_manager.last_episode_mean_reward("pitch_tracking", before_weight=True),
            roll     = self.reward_manager.last_episode_mean_reward("roll_tracking", before_weight=True),
            height   = self.reward_manager.last_episode_mean_reward("height_tracking", before_weight=True),
            episode_length=self._last_reset_mean_ep_length,
        )

    def _update_curriculum(self) -> None:
        if self.curriculum_level > Stage.Max:
            return
        if self.step_count < self.next_curriculum_check_step:
            return
        if len(self.curriculum_samples) < 10:
            return

        stage, sub_level = self.get_stage()

        average_values = torch.tensor(
            [dataclasses.astuple(sample) for sample in self.curriculum_samples]
        ).mean(dim=0).tolist()
        averages: CurriculumMetrics = CurriculumMetrics(*average_values)

        min_len = CURRICULUM_MIN_EPISODE_LENGTH_RATIO * (self.max_episode_length_steps or 0)
        thresholds = CurriculumMetrics(*[MIN_TRACKING_THRESHOLD] * 6, episode_length=min_len)
        track_fails = averages.tracks_baseline(thresholds)

        if not track_fails:
            if self.sub_level_step > 0 and sub_level + self.sub_level_step < 1.0 - 1e-6:
                self.curriculum_level = round(self.curriculum_level + self.sub_level_step, 6)
                self.log(f"Sub-level BUMP avgs: {averages}")
            else:
                self.curriculum_level = float(stage + 1)
                self.log(f"Curriculum BUMP avgs: {averages}")
            self._apply_curriculum_stage()
        else:
            avgstr = ", ".join(f"{k}={v:0.2f}" for k, v in averages.__dict__.items())
            self.log(f"Curriculum hold: {'ok' if not track_fails else ','.join(track_fails)} avgs: {avgstr}")
        self.next_curriculum_check_step = (self.step_count + CURRICULUM_CHECK_EVERY_STEPS)
        self.curriculum_samples.clear()

    def step(self, actions: torch.Tensor):
        self.cmd_smooth    += (self.velocity_command.command        -    self.cmd_smooth).clamp(-self._cmd_slew_max * self.dt, self._cmd_slew_max * self.dt)
        self.height_smooth += (self.height_command.command[:, 0]    - self.height_smooth).clamp(-MAX_HEIGHT_ACCEL * self.dt, MAX_HEIGHT_ACCEL * self.dt)
        self.pitch_smooth  += (self.attitude_command.command[:, 0]  -  self.pitch_smooth).clamp(-MAX_ATTITUDE_ACCEL * self.dt, MAX_ATTITUDE_ACCEL * self.dt)
        self.roll_smooth   += (self.attitude_command.command[:, 1]  -   self.roll_smooth).clamp(-MAX_ATTITUDE_ACCEL * self.dt, MAX_ATTITUDE_ACCEL * self.dt)

        observations_batch, reward, terminated, truncated, extras = super().step(actions)
        episode = extras.setdefault("episode", {})
        episode["Curriculum/level"] = self.curriculum_level
        wheel_rpm = self.wheel_action_manager.get_dofs_velocity() * (60.0 / (2.0 * math.pi))
        episode["Diagnostics/wheel_rpm"] = wheel_rpm.abs().mean().item()
        episode["Diagnostics/wheel_rpm_max"] = wheel_rpm.abs().max().item()

        # report command max/min values if enabled
        if self.tracking_target is not None:
            episode["Curriculum/target range"]= self.tracking_target.max().item() - self.tracking_target.min().item()
            episode["Curriculum/target avg"] = self.tracking_target.mean().item()
        if self.tracking_result is not None:
            episode["Curriculum/result range"] = self.tracking_result.max().item() - self.tracking_result.min().item()
            episode["Curriculum/result avg"] = self.tracking_result.mean().item()

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

    def reset(self, env_ids: torch.Tensor | collections.abc.Sequence[int] | None = None):
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

        if env_ids is None: reset_ids = self.all_envs_idx
        elif isinstance(env_ids, torch.Tensor): reset_ids = env_ids.to(device=gs.device, dtype=torch.long)
        else: reset_ids = torch.as_tensor(env_ids, device=gs.device, dtype=torch.long)

        default_positions = self.actuator_manager.default_dofs_pos
        if default_positions.ndim == 1:
            default_positions = default_positions.unsqueeze(0).expand(len(reset_ids), -1)
        else: default_positions = default_positions[reset_ids]
        self.robot.set_dofs_position(default_positions, dofs_idx_local=self.actuator_manager.dofs_idx, envs_idx=reset_ids)
        self.robot.zero_all_dofs_velocity(reset_ids)
        self.entity_manager.step()
        self.cmd_smooth[reset_ids] = self.velocity_command.command[reset_ids]
        self.height_smooth[reset_ids] = self.height_command.command[reset_ids, 0]
        self.pitch_smooth[reset_ids] = self.attitude_command.command[reset_ids, 0]
        self.roll_smooth[reset_ids] = self.attitude_command.command[reset_ids, 1]

        if env_ids is not None and len(reset_ids) > 0:
            self.curriculum_samples.append(self._curriculum_metrics())
            if len(self.curriculum_samples) > CURRICULUM_MAX_SAMPLES:
                self.curriculum_samples = self.curriculum_samples[-CURRICULUM_MAX_SAMPLES:]
            self._update_curriculum()
        if env_ids is None: observations_batch = self.get_observations()
        return observations_batch, info

    @staticmethod
    def _as_batch(value: torch.Tensor) -> torch.Tensor:
        return value.unsqueeze(0) if value.ndim == 1 else value

    def stagename(self) -> str:
        stage, sub_level = self.get_stage()
        try:
            return f"{Stage(stage).name}+{sub_level}"
        except ValueError:
            return f"Stage({self.curriculum_level})"
    def logbase(self, * args, **kwargs) -> None:
        if self.periodic_outfile:
            with open(self.periodic_outfile, 'a') as f:
                print(*args, **kwargs, file=f)
        print(*args, **kwargs)
    def log(self, * args, **kwargs) -> None:
        # report learning iteration
        self.logbase(f"training@[c{self.stagename()}|{self.curriculum_level:.2f} s{self.step_count}] ", *args, **kwargs)

