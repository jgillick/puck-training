import unittest

import mujoco
import numpy as np

from puck.env import PuckBalanceEnv


class PuckBalanceEnvTests(unittest.TestCase):
    def make_env(
        self,
        *,
        max_episode_steps: int = 500,
        target_pitch: float = 0.0,
        target_roll: float = 0.0,
        target_body_height: float | None = None,
        target_forward_speed: float = 0.0,
        target_sideways_speed: float = 0.0,
        target_yaw_rate: float = 0.0,
    ) -> PuckBalanceEnv:
        return PuckBalanceEnv(
            reset_noise_scale=0.0,
            max_episode_steps=max_episode_steps,
            target_pitch=target_pitch,
            target_roll=target_roll,
            target_body_height=target_body_height,
            target_forward_speed=target_forward_speed,
            target_sideways_speed=target_sideways_speed,
            target_yaw_rate=target_yaw_rate,
        )

    def test_reset_starts_in_supported_stance(self) -> None:
        env = self.make_env()
        try:
            observation, info = env.reset(seed=7)
            self.assertEqual(observation.shape, env.observation_space.shape)
            self.assertTrue(np.isfinite(observation).all())
            self.assertAlmostEqual(info["tilt_angle"], 0.0, places=6)
            self.assertAlmostEqual(info["pitch"], 0.0, places=6)
            self.assertAlmostEqual(info["roll"], 0.0, places=6)
            self.assertAlmostEqual(
                info["body_height"], info["target_body_height"], places=6
            )

            floor_ids = set(np.flatnonzero(env.model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE))
            contact_bodies: set[str | None] = set()
            for index in range(env.data.ncon):
                contact = env.data.contact[index]
                geom1, geom2 = int(contact.geom1), int(contact.geom2)
                if geom1 in floor_ids or geom2 in floor_ids:
                    robot_geom = geom2 if geom1 in floor_ids else geom1
                    body_id = int(env.model.geom_bodyid[robot_geom])
                    contact_bodies.add(
                        mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                    )
            self.assertEqual(contact_bodies, {"wheel", "wheel_2"})
        finally:
            env.close()

    def test_reward_distinguishes_stance_from_zero_pose(self) -> None:
        env = self.make_env()
        try:
            env.reset(seed=5)
            _, stand_reward, _, _, stand_info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            env.reset(seed=5)
            for name in env._LEG_JOINT_NAMES:
                joint_id = env._joint_ids[name]
                env.data.qpos[int(env.model.jnt_qposadr[joint_id])] = 0.0
            mujoco.mj_forward(env.model, env.data)
            _, crouch_reward, _, _, _ = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            self.assertGreater(stand_info["stance_reward"], 0.9)
            self.assertGreater(stand_reward, crouch_reward)
        finally:
            env.close()

    def test_no_wheel_correction_cannot_hold_stand_target(self) -> None:
        env = self.make_env(max_episode_steps=100)
        try:
            env.reset(seed=0)
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            for _ in range(env.max_episode_steps):
                _, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            self.assertTrue(terminated)
            self.assertIn(info["termination_reason"], ("tilt", "body_height"))
        finally:
            env.close()

    def test_wheel_and_leg_effort_have_separate_costs(self) -> None:
        env = self.make_env()
        try:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            action[env._joint_action_indices] = 0.5
            env.reset(seed=11)
            _, _, _, _, joint_info = env.step(action)

            action[:] = 0.0
            action[env._wheel_action_indices] = 0.5
            env.reset(seed=11)
            _, _, _, _, wheel_info = env.step(action)

            self.assertGreater(
                joint_info["joint_action_cost"], joint_info["wheel_action_cost"]
            )
            self.assertGreater(
                wheel_info["wheel_action_cost"], wheel_info["joint_action_cost"]
            )
        finally:
            env.close()

    def test_difficulty_increases_reset_perturbations(self) -> None:
        env = self.make_env()
        try:
            env.set_difficulty(0.0)
            _, easy = env.reset(seed=31)
            env.set_difficulty(1.0)
            observation, hard = env.reset(seed=31)
            self.assertGreater(hard["tilt_angle"], easy["tilt_angle"])
            self.assertTrue(np.isfinite(observation).all())
        finally:
            env.close()

    def test_pitch_and_roll_targets_are_configurable(self) -> None:
        env = self.make_env(
            target_pitch=0.05,
            target_roll=-0.05,
            target_body_height=0.15,
        )
        try:
            observation, info = env.reset(seed=13)
            self.assertAlmostEqual(info["pitch"], info["target_pitch"], places=6)
            self.assertAlmostEqual(info["roll"], info["target_roll"], places=6)
            self.assertAlmostEqual(info["body_height"], 0.15, places=6)
            self.assertTrue(env.observation_space.contains(observation))
        finally:
            env.close()

    def test_body_frame_speed_targets_are_observed_and_penalized(self) -> None:
        env = self.make_env(
            target_forward_speed=0.0,
            target_sideways_speed=0.0,
            target_yaw_rate=0.0,
        )
        try:
            observation, _ = env.reset(seed=19)
            env.data.qvel[env._root_dof_adr + 1] = 1.0
            env.data.qvel[env._root_dof_adr + 5] = 0.5
            mujoco.mj_forward(env.model, env.data)

            angular_velocity, linear_velocity = env._body_velocity()
            self.assertAlmostEqual(linear_velocity[1], 1.0, places=3)
            self.assertAlmostEqual(angular_velocity[2], 0.5, places=3)

            observation = env._get_obs()
            self.assertAlmostEqual(observation[5], 0.1, places=3)
            self.assertAlmostEqual(observation[7], 0.5, places=3)

            _, _, _, _, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            self.assertGreater(info["forward_speed_cost"], 0.0)
            self.assertGreater(info["yaw_rate_cost"], 0.0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
