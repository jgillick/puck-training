"""Gymnasium environment for balancing the Puck robot in MuJoCo."""
import pathlib, typing, time
import gymnasium as gym
from mujoco import MjData, MjModel, mjtObj, mjtJoint, mj_id2name, mju_quatIntegrate, mj_forward, mj_step, mj_resetData # type: ignore[attr-defined]
import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "scene.xml"


class PuckBalanceEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """Train a policy to keep Puck upright."""

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        model_path: str | pathlib.Path = DEFAULT_MODEL_PATH,
        render_mode: str | None = None,
        control_timestep: float = 0.02,
        max_episode_steps: int = 500,
        fall_angle: float = 0.8,
        reset_noise_scale: float = 0.01,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError(f"Unsupported render mode: {render_mode!r}")
        if control_timestep <= 0:
            raise ValueError("control_timestep must be positive")
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if not 0 < fall_angle < np.pi:
            raise ValueError("fall_angle must be between 0 and pi radians")
        if reset_noise_scale < 0:
            raise ValueError("reset_noise_scale must not be negative")

        self.render_mode = render_mode
        self.render_realtime: float | None = None
        self.render_nohold: bool = False
        self.max_episode_steps = max_episode_steps
        self.fall_angle = fall_angle
        self.reset_noise_scale = reset_noise_scale
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

        free_joint_ids = np.flatnonzero(
            self.model.jnt_type == mjtJoint.mjJNT_FREE
        )
        if len(free_joint_ids) != 1:
            raise ValueError(
                f"Expected exactly one free root joint, found {len(free_joint_ids)}"
            )
        self._root_joint_id = int(free_joint_ids[0])
        self._root_body_id = int(self.model.jnt_bodyid[self._root_joint_id])
        self._root_qpos_adr = int(self.model.jnt_qposadr[self._root_joint_id])
        self._root_dof_adr = int(self.model.jnt_dofadr[self._root_joint_id])

        self.joint_names = self._named_objects(
            mjtObj.mjOBJ_JOINT, self.model.njnt
        )
        self.actuator_names = self._named_objects(
            mjtObj.mjOBJ_ACTUATOR, self.model.nu
        )
        self.sensor_names = self._named_objects(
            mjtObj.mjOBJ_SENSOR, self.model.nsensor
        )
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

        self._joint_qpos_addresses = np.asarray(
            [
                int(self.model.jnt_qposadr[joint_id])
                for joint_id in range(self.model.njnt)
                if joint_id != self._root_joint_id
            ],
            dtype=np.intp,
        )
        self._joint_dof_addresses = np.asarray(
            [
                int(self.model.jnt_dofadr[joint_id])
                for joint_id in range(self.model.njnt)
                if joint_id != self._root_joint_id
            ],
            dtype=np.intp,
        )

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )
        observation_size = self.model.nq + self.model.nv + self.model.nsensordata
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32,
        )
        self._step_count = 0
        self._viewer: typing.Any | None = None

    def _named_objects(
        self, object_type: mjtObj, count: int
    ) -> tuple[str, ...]:
        names = tuple(
            mj_id2name(self.model, object_type, index)
            for index in range(count)
        )
        if any(name is None for name in names):
            raise ValueError(f"All {object_type.name} entries must have XML names")
        return tuple(name for name in names if name is not None)

    def _get_obs(self) -> NDArray[np.float32]:
        observation = np.concatenate(
            (self.data.qpos, self.data.qvel, self.data.sensordata)
        )
        return observation.astype(np.float32, copy=False)

    def _tilt_angle(self) -> float:
        rotation = self.data.xmat[self._root_body_id].reshape(3, 3)
        cosine = np.clip(rotation[2, 2], -1.0, 1.0)
        return float(np.arccos(cosine))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, typing.Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, typing.Any]]:
        super().reset(seed=seed)
        mj_resetData(self.model, self.data)

        if self.reset_noise_scale:
            if self._joint_qpos_addresses.size:
                self.data.qpos[self._joint_qpos_addresses] += self.np_random.uniform(
                    -self.reset_noise_scale,
                    self.reset_noise_scale,
                    size=self._joint_qpos_addresses.size,
                )
                for joint_id in range(self.model.njnt):
                    if self.model.jnt_limited[joint_id]:
                        address = int(self.model.jnt_qposadr[joint_id])
                        low, high = self.model.jnt_range[joint_id]
                        self.data.qpos[address] = np.clip(
                            self.data.qpos[address], low, high
                        )

            orientation_noise = self.np_random.normal(
                0.0, self.reset_noise_scale, size=3
            )
            quaternion = self.data.qpos[
                self._root_qpos_adr + 3 : self._root_qpos_adr + 7
            ]
            mju_quatIntegrate(quaternion, orientation_noise, 1.0)
            self.data.qvel += self.np_random.normal(
                0.0, self.reset_noise_scale, size=self.model.nv
            )

        mj_forward(self.model, self.data)
        self._step_count = 0
        if self.render_mode == "human":
            self.render()
        info = {
            "tilt_angle": self._tilt_angle(),
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
        self.data.ctrl[:] = self._ctrl_low + (normalized_action + 1.0) * 0.5 * (
            self._ctrl_high - self._ctrl_low
        )

        for _ in range(self.frame_skip):
            mj_step(self.model, self.data)

        self._step_count += 1
        tilt_angle = self._tilt_angle()
        terminated = tilt_angle >= self.fall_angle
        truncated = self._step_count >= self.max_episode_steps

        root_angular_velocity = self.data.qvel[
            self._root_dof_adr : self._root_dof_adr + 3
        ]
        root_linear_velocity = self.data.qvel[
            self._root_dof_adr + 3 : self._root_dof_adr + 6
        ]
        joint_velocity = self.data.qvel[self._joint_dof_addresses]
        reward = (
            float(np.cos(tilt_angle))
            - 0.01 * float(np.mean(np.square(normalized_action)))
            - 0.01 * float(np.mean(np.square(root_angular_velocity)))
            - 0.02 * float(np.mean(np.square(root_linear_velocity)))
            - 0.001 * float(np.mean(np.square(joint_velocity)))
        )
        info = {
            "tilt_angle": tilt_angle,
            "episode_step": self._step_count,
            "actuator_names": self.actuator_names,
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
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data, show_right_ui=False, key_callback=key_callback)
        self._viewer.sync()
        if self.render_realtime is not None:
            time.sleep(self.model.opt.timestep / self.render_realtime)
        return None

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
