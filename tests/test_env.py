import unittest

import genesis as gs
import torch

from puck.env import (
    LIMB_JOINT_NAMES,
    STANDING_JOINT_POSITIONS,
    PuckStandEnv,
    _knee_direction_reward,
    _leg_symmetry_reward,
)


class PuckStandEnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        gs.init(backend=gs.cpu, logging_level="warning")
        cls.env = PuckStandEnv(num_envs=2)
        cls.env.build()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def setUp(self) -> None:
        self.observations, _ = self.env.reset()

    def test_position_actions_default_to_the_mirrored_standing_pose(self) -> None:
        self.assertEqual(self.env.action_space.shape, (8,))
        self.assertEqual(self.env.action_manager.num_actions, 6)
        self.assertEqual(self.env.wheel_action_manager.num_actions, 2)

        limb_targets = self.env.action_manager.process_actions(
            torch.zeros((2, 6), device=gs.device)
        )
        for index, name in enumerate(LIMB_JOINT_NAMES):
            self.assertAlmostEqual(
                float(limb_targets[0, index]),
                STANDING_JOINT_POSITIONS[name],
                places=5,
            )

        positions = self.env.actuator_manager.get_dofs_position()
        names = self.env.actuator_manager.dofs_names
        for name, target in STANDING_JOINT_POSITIONS.items():
            self.assertTrue(
                torch.allclose(
                    positions[:, names.index(name)],
                    torch.full((2,), target, device=gs.device),
                    atol=1e-5,
                )
            )
        self.assertAlmostEqual(float(self.env.robot.get_pos()[0, 2]), 0.5, places=5)

    def test_wheel_torque_actions_are_bounded_by_robot_model(self) -> None:
        actions = torch.tensor([[2.0, 2.0], [-2.0, -2.0]], device=gs.device)
        torques = self.env.wheel_action_manager.process_actions(actions)
        lower, upper = self.env.robot.get_dofs_force_range(
            self.env.wheel_action_manager.dofs_idx
        )
        self.assertTrue(torch.allclose(torques[0], upper))
        self.assertTrue(torch.allclose(torques[1], lower))

    def test_commands_and_five_step_observations_are_normalized(self) -> None:
        self.assertEqual(self.observations.shape, (2, 190))
        self.assertTrue(torch.isfinite(self.observations).all())
        self.assertTrue(torch.all(self.observations <= 1.0))
        self.assertTrue(torch.all(self.observations >= -1.0))
        self.assertEqual(self.env.observation_manager._history_len, 5)
        self.assertEqual(self.env.height_command.range, (0.1, 0.5))
        self.assertEqual(
            self.env.velocity_command.range,
            {
                "lin_vel_x": (0.0, 0.1),
                "lin_vel_y": (0.0, 0.0),
                "ang_vel_z": (0.0, 0.0),
            },
        )

    def test_leg_symmetry_uses_mirrored_signs_and_ignores_wheels(self) -> None:
        names = self.env.actuator_manager.dofs_names
        positions = torch.zeros((2, len(names)), device=gs.device)
        paired_positions = {
            "flexor_l": 0.2,
            "flexor_r": 0.2,
            "hip_l": -0.5,
            "hip_r": 0.5,
            "knee_l": -1.0,
            "knee_r": 1.0,
        }
        for name, position in paired_positions.items():
            positions[:, names.index(name)] = position
        positions[:, names.index("wheel_l")] = 100.0
        positions[:, names.index("wheel_r")] = -100.0

        symmetry = _leg_symmetry_reward(positions, names)
        self.assertTrue(torch.allclose(symmetry, torch.ones_like(symmetry)))

        positions[0, names.index("knee_l")] += 0.5
        symmetry = _leg_symmetry_reward(positions, names)
        self.assertLess(float(symmetry[0]), float(symmetry[1]))

    def test_knee_direction_reward_penalizes_reverse_bending(self) -> None:
        names = self.env.actuator_manager.dofs_names
        positions = torch.zeros((2, len(names)), device=gs.device)
        positions[0, names.index("knee_r")] = 0.5
        positions[0, names.index("knee_l")] = -0.5
        positions[1, names.index("knee_r")] = -0.5
        positions[1, names.index("knee_l")] = 0.5

        rewards = _knee_direction_reward(positions, names)
        self.assertAlmostEqual(float(rewards[0]), 0.0)
        self.assertLess(float(rewards[1]), 0.0)

    def test_reset_restores_standing_pose_and_partial_reset(self) -> None:
        positions_before = self.env._as_batch(self.env.robot.get_pos()).clone()
        reset_ids = torch.tensor([1], device=gs.device)
        self.env.robot.set_pos(
            torch.tensor([[0.0, 0.0, 0.7]], device=gs.device),
            envs_idx=reset_ids,
        )
        self.env.reset(reset_ids)

        positions_after = self.env._as_batch(self.env.robot.get_pos())
        self.assertTrue(torch.allclose(positions_after[0], positions_before[0]))
        self.assertAlmostEqual(float(positions_after[1, 2]), 0.5, places=5)
        joints = self.env.actuator_manager.get_dofs_position()
        names = self.env.actuator_manager.dofs_names
        for name, target in STANDING_JOINT_POSITIONS.items():
            self.assertAlmostEqual(
                float(joints[1, names.index(name)]), target, places=5
            )

    def test_step_returns_bounded_rewards_and_observations(self) -> None:
        actions = torch.zeros((2, 8), device=gs.device)
        observations, rewards, terminated, truncated, _ = self.env.step(actions)
        self.assertEqual(observations.shape, (2, 190))
        self.assertTrue(torch.isfinite(observations).all())
        self.assertTrue(torch.all(observations <= 1.0))
        self.assertTrue(torch.all(observations >= -1.0))
        self.assertTrue(torch.isfinite(rewards).all())
        self.assertTrue(torch.all(rewards <= 1.0))
        self.assertTrue(torch.all(rewards >= -1.0))
        self.assertEqual(terminated.shape, (2,))
        self.assertEqual(truncated.shape, (2,))


if __name__ == "__main__":
    unittest.main()
