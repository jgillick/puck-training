"""Gymnasium environment for balancing the Puck robot in MuJoCo."""
import pathlib
import time
import typing

import gymnasium as gym
import mujoco
from mujoco import (
    MjData,
    MjModel,
    mj_id2name,
    mj_forward,
    mj_resetData,
    mj_step,
    mjtGeom,
    mjtJoint,
    mjtObj,
    mju_quatIntegrate,
)
import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "scene.xml"


class PuckBalanceEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """Train leg-position offsets and wheel-torque actions to balance Puck."""

    metadata = {"render_modes": ["human"], "render_fps": 50}

    _LEG_JOINT_NAMES = (
        "flexor_l",
        "flexor_r",
        "hip_l",
        "hip_r",
        "knee_l",
        "knee_r",
    )
    _WHEEL_JOINT_NAMES = ("wheel_l", "wheel_r")
    _LEG_POSITION_GAIN = 5.0
    _LEG_VELOCITY_GAIN = 1.0
    _LEG_TARGET_ACTION_SCALE = 0.35

    def __init__(
        self,
        model_path: str | pathlib.Path = DEFAULT_MODEL_PATH,
        render_mode: str | None = None,
        control_timestep: float = 0.02,
        max_episode_steps: int = 500,
        fall_angle: float = 0.8,
        reset_noise_scale: float = 0.01,
        target_stance_height: float = 0.3,
        target_pitch: float = 0.0,
        target_roll: float = 0.0,
        target_body_height: float | None = None,
        max_reset_tilt: float = 0.12,
        max_reset_joint_noise: float = 0.04,
        max_reset_angular_velocity: float = 0.5,
        max_reset_linear_velocity: float = 0.15,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError(f"Unsupported render mode: {render_mode!r}")
        if not np.isfinite(control_timestep) or control_timestep <= 0:
            raise ValueError("control_timestep must be positive")
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if not 0 < fall_angle < np.pi:
            raise ValueError("fall_angle must be between 0 and pi radians")
        if not np.isfinite(reset_noise_scale) or reset_noise_scale < 0:
            raise ValueError("reset_noise_scale must be finite and not negative")
        if not 0.05 <= target_stance_height <= 1.0:
            raise ValueError("target_stance_height must be between 0.05 and 1.0")
        if not np.isfinite(target_pitch) or not np.isfinite(target_roll):
            raise ValueError("target_pitch and target_roll must be finite")
        if np.hypot(target_pitch, target_roll) >= fall_angle:
            raise ValueError("target pitch and roll must be within the fall angle")
        if target_body_height is not None and (
            not np.isfinite(target_body_height) or target_body_height <= 0
        ):
            raise ValueError("target_body_height must be positive and finite")
        reset_limits = (
            max_reset_tilt,
            max_reset_joint_noise,
            max_reset_angular_velocity,
            max_reset_linear_velocity,
        )
        if not np.all(np.isfinite(reset_limits)) or any(
            limit < 0 for limit in reset_limits
        ):
            raise ValueError("reset perturbation limits must be finite and nonnegative")

        self.render_mode = render_mode
        self.render_realtime: float | None = None
        self.render_nohold = False
        self.max_episode_steps = max_episode_steps
        self.fall_angle = fall_angle
        self.reset_noise_scale = reset_noise_scale
        self.target_stance_height = target_stance_height
        self.target_pitch = target_pitch
        self.target_roll = target_roll
        self.max_reset_tilt = max_reset_tilt
        self.max_reset_joint_noise = max_reset_joint_noise
        self.max_reset_angular_velocity = max_reset_angular_velocity
        self.max_reset_linear_velocity = max_reset_linear_velocity
        self.difficulty = 0.0
        self.model = MjModel.from_xml_path(str(pathlib.Path(model_path).resolve()))
        self.data = MjData(self.model)

        physics_timestep = float(self.model.opt.timestep)
        frame_skip = round(control_timestep / physics_timestep)
        if frame_skip < 1 or not np.isclose(
            frame_skip * physics_timestep, control_timestep, rtol=0, atol=1e-9
        ):
            raise ValueError(
                "control_timestep must be a positive integer multiple of "
                f"the XML physics timestep ({physics_timestep})"
            )
        self.frame_skip = frame_skip

        free_joint_ids = np.flatnonzero(self.model.jnt_type == mjtJoint.mjJNT_FREE)
        if len(free_joint_ids) != 1:
            raise ValueError(
                f"Expected exactly one free root joint, found {len(free_joint_ids)}"
            )
        self._root_joint_id = int(free_joint_ids[0])
        self._root_body_id = int(self.model.jnt_bodyid[self._root_joint_id])
        self._root_qpos_adr = int(self.model.jnt_qposadr[self._root_joint_id])
        self._root_dof_adr = int(self.model.jnt_dofadr[self._root_joint_id])

        self.joint_names = self._named_objects(mjtObj.mjOBJ_JOINT, self.model.njnt)
        self.actuator_names = self._named_objects(
            mjtObj.mjOBJ_ACTUATOR, self.model.nu
        )
        self.sensor_names = self._named_objects(mjtObj.mjOBJ_SENSOR, self.model.nsensor)
        self._joint_ids = {
            name: mujoco.mj_name2id(self.model, mjtObj.mjOBJ_JOINT, name)
            for name in self._LEG_JOINT_NAMES + self._WHEEL_JOINT_NAMES
        }
        missing_joints = [
            name for name, joint_id in self._joint_ids.items() if joint_id < 0
        ]
        if missing_joints:
            raise ValueError(f"Model is missing required joints: {missing_joints}")
        self._pose_qpos_addresses = np.asarray(
            [
                int(self.model.jnt_qposadr[self._joint_ids[name]])
                for name in self._LEG_JOINT_NAMES
            ],
            dtype=np.intp,
        )
        self._pose_dof_addresses = np.asarray(
            [
                int(self.model.jnt_dofadr[self._joint_ids[name]])
                for name in self._LEG_JOINT_NAMES
            ],
            dtype=np.intp,
        )
        self._wheel_dof_addresses = np.asarray(
            [int(self.model.jnt_dofadr[self._joint_ids[name]]) for name in self._WHEEL_JOINT_NAMES],
            dtype=np.intp,
        )
        self._joint_action_indices = np.asarray(
            [
                i
                for i, name in enumerate(self.actuator_names)
                if name not in self._WHEEL_JOINT_NAMES
            ],
            dtype=np.intp,
        )
        self._wheel_action_indices = np.asarray(
            [
                i
                for i, name in enumerate(self.actuator_names)
                if name in self._WHEEL_JOINT_NAMES
            ],
            dtype=np.intp,
        )
        known_actuators = set(self._LEG_JOINT_NAMES + self._WHEEL_JOINT_NAMES)
        missing_actuators = [
            name
            for name in self._LEG_JOINT_NAMES + self._WHEEL_JOINT_NAMES
            if name not in self.actuator_names
        ]
        if missing_actuators:
            raise ValueError(f"Model is missing required actuators: {missing_actuators}")
        unexpected_actuators = [
            name for name in self.actuator_names if name not in known_actuators
        ]
        if unexpected_actuators:
            raise ValueError(f"Model has unsupported actuators: {unexpected_actuators}")

        actuator_limited = self.model.actuator_ctrllimited.astype(bool)
        if not np.all(actuator_limited):
            unbounded = [
                self.actuator_names[i]
                for i in np.flatnonzero(~actuator_limited)
            ]
            raise ValueError(
                "Every actuator must define a bounded ctrlrange in the XML; "
                f"unbounded actuators: {unbounded}"
            )
        self._ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()

        self._target_joint_positions = self._make_target_joint_positions()
        self._target_quaternion = self._make_target_quaternion()
        self._floor_geom_ids = np.flatnonzero(
            self.model.geom_type == mjtGeom.mjGEOM_PLANE
        )
        if not self._floor_geom_ids.size:
            raise ValueError("Model must contain a plane geom for wheel-ground support")
        self._wheel_body_ids = {
            int(self.model.jnt_bodyid[self._joint_ids[name]])
            for name in self._WHEEL_JOINT_NAMES
        }
        self._target_root_z = 0.0

        if target_body_height is None:
            self._target_root_z = self._find_wheel_supported_root_height()
            self._target_body_height = float(self.data.xipos[self._root_body_id, 2])
        else:
            self._target_body_height = target_body_height
            mj_resetData(self.model, self.data)
            self._set_target_pose()
            self.data.qpos[self._root_qpos_adr + 2] = 0.0
            mj_forward(self.model, self.data)
            self._target_root_z = target_body_height - float(
                self.data.xipos[self._root_body_id, 2]
            )

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32
        )
        self._step_count = 0
        self._viewer: typing.Any | None = None
        self._spawn_xy = np.zeros(2, dtype=np.float64)

    def _named_objects(self, object_type: mjtObj, count: int) -> tuple[str, ...]:
        names = tuple(
            mj_id2name(self.model, object_type, index) for index in range(count)
        )
        if any(name is None for name in names):
            raise ValueError(f"All {object_type.name} entries must have XML names")
        return tuple(name for name in names if name is not None)

    def _make_target_joint_positions(self) -> dict[str, float]:
        height = self.target_stance_height
        height_l = float(np.clip(height + self.target_roll / 2.0, 0.0, 1.0))
        height_r = float(np.clip(height - self.target_roll / 2.0, 0.0, 1.0))
        # The mirrored MuJoCo leg frames require opposite left/right joint signs.
        targets = {
            "flexor_l": -self.target_roll,
            "flexor_r": -self.target_roll,
            "hip_l": np.pi * height_l / 2.0 + 0.6 * self.target_pitch,
            "hip_r": -np.pi * height_r / 2.0 - 0.6 * self.target_pitch,
            "knee_l": np.pi * height_l,
            "knee_r": -np.pi * height_r,
        }
        for name, position in targets.items():
            joint_id = self._joint_ids[name]
            if self.model.jnt_limited[joint_id]:
                low, high = self.model.jnt_range[joint_id]
                if not low <= position <= high:
                    raise ValueError(
                        f"Target {name}={position:.3f} is outside its joint range "
                        f"[{low:.3f}, {high:.3f}]"
                    )
        return targets

    def _make_target_quaternion(self) -> NDArray[np.float64]:
        half_pitch = self.target_pitch / 2.0
        half_roll = self.target_roll / 2.0
        cp, sp = np.cos(half_pitch), np.sin(half_pitch)
        cr, sr = np.cos(half_roll), np.sin(half_roll)
        return np.asarray(
            [cp * cr, cp * sr, sp * cr, -sp * sr], dtype=np.float64
        )

    def _set_target_pose(self) -> None:
        self.data.qpos[self._root_qpos_adr : self._root_qpos_adr + 3] = (
            0.0,
            0.0,
            self._target_root_z,
        )
        self.data.qpos[
            self._root_qpos_adr + 3 : self._root_qpos_adr + 7
        ] = self._target_quaternion
        for name, position in self._target_joint_positions.items():
            joint_id = self._joint_ids[name]
            self.data.qpos[int(self.model.jnt_qposadr[joint_id])] = position

    def _find_wheel_supported_root_height(self) -> float:
        mj_resetData(self.model, self.data)
        self._set_target_pose()
        floor_ids = set(int(i) for i in self._floor_geom_ids)
        best: tuple[float, float] | None = None
        for root_z in np.arange(-0.25, 0.501, 0.002):
            self.data.qpos[self._root_qpos_adr + 2] = root_z
            mj_forward(self.model, self.data)
            wheel_distances: dict[int, list[float]] = {
                body_id: [] for body_id in self._wheel_body_ids
            }
            other_floor_contact = False
            for contact_index in range(self.data.ncon):
                contact = self.data.contact[contact_index]
                geom1, geom2 = int(contact.geom1), int(contact.geom2)
                if geom1 not in floor_ids and geom2 not in floor_ids:
                    continue
                other_geom = geom2 if geom1 in floor_ids else geom1
                body_id = int(self.model.geom_bodyid[other_geom])
                if body_id in wheel_distances:
                    wheel_distances[body_id].append(float(contact.dist))
                else:
                    other_floor_contact = True
            if other_floor_contact or any(
                not distances for distances in wheel_distances.values()
            ):
                continue
            score = max(abs(min(distances)) for distances in wheel_distances.values())
            if best is None or score < best[0]:
                best = (score, float(root_z))

        if best is None or best[0] > 0.005:
            raise ValueError(
                "Could not find a root height where both wheels support the target "
                "stance without other robot parts touching the floor. Adjust "
                "target_stance_height/target_pitch/target_roll or set "
                "target_body_height explicitly."
            )
        mj_resetData(self.model, self.data)
        self._set_target_pose()
        self.data.qpos[self._root_qpos_adr + 2] = best[1]
        mj_forward(self.model, self.data)
        return best[1]

    def set_difficulty(self, difficulty: float) -> None:
        """Set reset perturbation strength from 0 (easy) to 1 (hard)."""
        if not np.isfinite(difficulty) or not 0.0 <= difficulty <= 1.0:
            raise ValueError("difficulty must be between 0 and 1")
        self.difficulty = float(difficulty)

    def _get_obs(self) -> NDArray[np.float32]:
        rotation = self.data.xmat[self._root_body_id].reshape(3, 3)
        gravity_body = rotation.T @ np.asarray([0.0, 0.0, 1.0])
        root_angular_velocity = self.data.qvel[
            self._root_dof_adr : self._root_dof_adr + 3
        ]
        root_linear_velocity = self.data.qvel[
            self._root_dof_adr + 3 : self._root_dof_adr + 6
        ]
        leg_positions = self.data.qpos[self._pose_qpos_addresses]
        leg_velocities = self.data.qvel[self._pose_dof_addresses]
        wheel_velocities = self.data.qvel[self._wheel_dof_addresses]
        root_xy = self.data.qpos[self._root_qpos_adr : self._root_qpos_adr + 2]
        body_height_error = (
            self.data.xipos[self._root_body_id, 2] - self._target_body_height
        )
        observation = np.concatenate(
            (
                gravity_body,
                root_angular_velocity / 5.0,
                root_linear_velocity / 2.0,
                leg_positions / np.pi,
                leg_velocities / 10.0,
                wheel_velocities / 20.0,
                root_xy - self._spawn_xy,
                np.asarray([body_height_error / 0.2]),
                np.asarray(
                    [
                        self.target_stance_height,
                        self.target_pitch / np.pi,
                        self.target_roll / np.pi,
                    ]
                ),
            )
        )
        return observation.astype(np.float32, copy=False)

    def _tilt_angle(self) -> float:
        rotation = self.data.xmat[self._root_body_id].reshape(3, 3)
        cosine = np.clip(rotation[2, 2], -1.0, 1.0)
        return float(np.arccos(cosine))

    def _pitch_roll(self) -> tuple[float, float]:
        rotation = self.data.xmat[self._root_body_id].reshape(3, 3)
        pitch = float(np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0)))
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        return pitch, roll

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, typing.Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, typing.Any]]:
        super().reset(seed=seed)
        mj_resetData(self.model, self.data)
        self._set_target_pose()

        joint_noise = self.reset_noise_scale + (
            self.difficulty * self.max_reset_joint_noise
        )
        if joint_noise:
            self.data.qpos[self._pose_qpos_addresses] += self.np_random.uniform(
                -joint_noise, joint_noise, size=self._pose_qpos_addresses.size
            )
            for name in self._LEG_JOINT_NAMES:
                joint_id = self._joint_ids[name]
                if self.model.jnt_limited[joint_id]:
                    address = int(self.model.jnt_qposadr[joint_id])
                    low, high = self.model.jnt_range[joint_id]
                    self.data.qpos[address] = np.clip(
                        self.data.qpos[address], low, high
                    )

        tilt_noise = self.np_random.normal(
            0.0, self.reset_noise_scale, size=2
        ) + self.np_random.uniform(
            -self.difficulty * self.max_reset_tilt,
            self.difficulty * self.max_reset_tilt,
            size=2,
        )
        quaternion = self.data.qpos[
            self._root_qpos_adr + 3 : self._root_qpos_adr + 7
        ]
        mju_quatIntegrate(
            quaternion, np.asarray([tilt_noise[0], tilt_noise[1], 0.0]), 1.0
        )

        velocity_noise = self.reset_noise_scale
        self.data.qvel += self.np_random.normal(
            0.0, velocity_noise, size=self.model.nv
        )
        self.data.qvel[
            self._root_dof_adr : self._root_dof_adr + 2
        ] += self.np_random.uniform(
            -self.difficulty * self.max_reset_angular_velocity,
            self.difficulty * self.max_reset_angular_velocity,
            size=2,
        )
        self.data.qvel[
            self._root_dof_adr + 3 : self._root_dof_adr + 5
        ] += self.np_random.uniform(
            -self.difficulty * self.max_reset_linear_velocity,
            self.difficulty * self.max_reset_linear_velocity,
            size=2,
        )

        mj_forward(self.model, self.data)
        self._step_count = 0
        if self.render_mode == "human":
            self.render()
        pitch, roll = self._pitch_roll()
        info = {
            "tilt_angle": self._tilt_angle(),
            "pitch": pitch,
            "roll": roll,
            "body_height": float(self.data.xipos[self._root_body_id, 2]),
            "target_body_height": self._target_body_height,
            "target_stance_height": self.target_stance_height,
            "target_pitch": self.target_pitch,
            "target_roll": self.target_roll,
            "actuator_names": self.actuator_names,
            "sensor_names": self.sensor_names,
        }
        return self._get_obs(), info

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, typing.Any]]:
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, "
                f"received {action_array.shape}"
            )
        if not np.all(np.isfinite(action_array)):
            raise ValueError("Actions must contain only finite values")
        normalized_action = np.clip(action_array, -1.0, 1.0)
        for action_index, actuator_name in enumerate(self.actuator_names):
            if actuator_name in self._WHEEL_JOINT_NAMES:
                self.data.ctrl[action_index] = self._ctrl_low[action_index] + (
                    normalized_action[action_index] + 1.0
                ) * 0.5 * (
                    self._ctrl_high[action_index] - self._ctrl_low[action_index]
                )
                continue

            joint_id = self._joint_ids[actuator_name]
            qpos_address = int(self.model.jnt_qposadr[joint_id])
            dof_address = int(self.model.jnt_dofadr[joint_id])
            target_position = (
                self._target_joint_positions[actuator_name]
                + normalized_action[action_index] * self._LEG_TARGET_ACTION_SCALE
            )
            torque = self._LEG_POSITION_GAIN * (
                target_position - self.data.qpos[qpos_address]
            ) - self._LEG_VELOCITY_GAIN * self.data.qvel[dof_address]
            self.data.ctrl[action_index] = np.clip(
                torque, self._ctrl_low[action_index], self._ctrl_high[action_index]
            )

        for _ in range(self.frame_skip):
            mj_step(self.model, self.data)

        self._step_count += 1
        tilt_angle = self._tilt_angle()
        body_height = float(self.data.xipos[self._root_body_id, 2])
        height_error = body_height - self._target_body_height
        tilt_fall = tilt_angle >= self.fall_angle
        height_fall = body_height < max(0.02, self._target_body_height - 0.06)
        terminated = tilt_fall or height_fall
        truncated = self._step_count >= self.max_episode_steps

        pitch, roll = self._pitch_roll()
        pitch_error = float(
            np.arctan2(
                np.sin(pitch - self.target_pitch),
                np.cos(pitch - self.target_pitch),
            )
        )
        roll_error = float(
            np.arctan2(
                np.sin(roll - self.target_roll),
                np.cos(roll - self.target_roll),
            )
        )
        orientation_reward = float(
            np.exp(-0.5 * ((pitch_error / 0.2) ** 2 + (roll_error / 0.2) ** 2))
        )
        pose_error = np.asarray(
            [
                self.data.qpos[int(self.model.jnt_qposadr[self._joint_ids[name]])]
                - self._target_joint_positions[name]
                for name in self._LEG_JOINT_NAMES
            ]
        )
        stance_reward = float(np.exp(-np.mean(np.square(pose_error / 0.25))))
        height_reward = float(np.exp(-((height_error / 0.05) ** 2)))

        root_angular_velocity = self.data.qvel[
            self._root_dof_adr : self._root_dof_adr + 3
        ]
        root_linear_velocity = self.data.qvel[
            self._root_dof_adr + 3 : self._root_dof_adr + 6
        ]
        leg_velocity = self.data.qvel[self._pose_dof_addresses]
        wheel_velocity = self.data.qvel[self._wheel_dof_addresses]
        joint_action_cost = 0.005 * float(
            np.mean(np.square(normalized_action[self._joint_action_indices]))
        )
        wheel_action_cost = 0.0005 * float(
            np.mean(np.square(normalized_action[self._wheel_action_indices]))
        )
        angular_velocity_cost = 0.01 * float(
            np.mean(np.square(root_angular_velocity / 5.0))
        )
        linear_velocity_cost = 0.02 * float(
            np.mean(np.square(root_linear_velocity[:2] / 1.0))
        )
        leg_velocity_cost = 0.001 * float(np.mean(np.square(leg_velocity / 10.0)))
        wheel_velocity_cost = 0.002 * float(np.mean(np.square(wheel_velocity / 20.0)))
        position_error = self.data.qpos[
            self._root_qpos_adr : self._root_qpos_adr + 2
        ] - self._spawn_xy
        position_cost = 0.05 * float(np.mean(np.square(position_error / 0.5)))

        reward = (
            1.0
            + orientation_reward
            + stance_reward
            + height_reward
            - joint_action_cost
            - wheel_action_cost
            - angular_velocity_cost
            - linear_velocity_cost
            - leg_velocity_cost
            - wheel_velocity_cost
            - position_cost
        )
        info = {
            "tilt_angle": tilt_angle,
            "pitch": pitch,
            "roll": roll,
            "body_height": body_height,
            "height_error": height_error,
            "stance_error": float(np.sqrt(np.mean(np.square(pose_error)))),
            "orientation_reward": orientation_reward,
            "stance_reward": stance_reward,
            "height_reward": height_reward,
            "joint_action_cost": joint_action_cost,
            "wheel_action_cost": wheel_action_cost,
            "episode_step": self._step_count,
            "actuator_names": self.actuator_names,
            "termination_reason": (
                "tilt" if tilt_fall else "body_height" if height_fall else None
            ),
        }
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> None:
        if self.render_mode != "human":
            return None
        if self._viewer is None:
            import mujoco.viewer

            def key_callback(key: int) -> None:
                if self._viewer and self.render_nohold and key == ord("q"):
                    self._viewer.close()

            self._viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_right_ui=False,
                key_callback=key_callback,
            )
        self._viewer.sync()
        if self.render_realtime is not None:
            time.sleep(self.model.opt.timestep / self.render_realtime)
        return None

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
