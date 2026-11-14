import unittest

import genesis as gs
import torch

from puck.env import STAND_HEIGHT, STAND_POSE, PuckStandEnv


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

    def test_imports_all_actuators_and_maps_neutral_action_to_stand_pose(self) -> None:
        self.assertEqual(self.env.actuator_manager.num_dofs, 8)
        self.assertEqual(self.env.action_space.shape, (8,))
        self.assertEqual(self.observations.shape, (2, 34))
        self.assertTrue(torch.isfinite(self.observations).all())

        zero_actions = torch.zeros((2, 8), device=gs.device)
        targets = self.env.action_manager.process_actions(zero_actions)
        lower, upper = self.env.actuator_manager.get_dofs_limits(
            self.env.action_manager.dofs_idx
        )
        for name, position in STAND_POSE.items():
            index = self.env.actuator_manager.dofs_names.index(name)
            expected = min(max(position, float(lower[index])), float(upper[index]))
            if name.startswith("wheel_"):
                expected = 0.0
            self.assertAlmostEqual(float(targets[0, index]), expected)
        self.assertAlmostEqual(STAND_HEIGHT, 0.5)

    def test_wheel_actions_are_bounded_torque_commands(self) -> None:
        actions = torch.zeros((2, 8), device=gs.device)
        wheel_indices = self.env.action_manager._wheel_action_indices
        actions[:, wheel_indices[0]] = 2.0
        actions[:, wheel_indices[1]] = -2.0

        targets = self.env.action_manager.process_actions(actions)
        self.assertTrue(
            torch.allclose(
                targets[:, wheel_indices],
                torch.tensor(
                    [[12.0, -12.0], [12.0, -12.0]], device=gs.device
                ),
            )
        )

    def test_reset_restores_the_passively_settled_crouch(self) -> None:
        root_position = self.env._as_batch(self.env.robot.get_pos())
        joint_positions = self.env.actuator_manager.get_dofs_position()
        self.assertTrue(
            torch.allclose(root_position, self.env._rest_position, atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(
                joint_positions, self.env._rest_dof_positions, atol=1e-5
            )
        )
        self.assertFalse(
            torch.allclose(
                joint_positions,
                self.env.actuator_manager.default_dofs_pos,
                atol=1e-2,
            )
        )

    def test_partial_reset_only_restores_requested_environment(self) -> None:
        positions_before = self.env._as_batch(self.env.robot.get_pos()).clone()
        reset_ids = torch.tensor([1], device=gs.device)
        self.env.reset(reset_ids)
        positions_after = self.env._as_batch(self.env.robot.get_pos())

        self.assertTrue(
            torch.allclose(
                positions_after[0], positions_before[0], atol=1e-5
            )
        )
        self.assertTrue(
            torch.allclose(
                positions_after[1], self.env._rest_position[1], atol=1e-5
            )
        )

    def test_managed_step_returns_finite_rewards_and_observations(self) -> None:
        actions = torch.zeros((2, 8), device=gs.device)
        observations, rewards, terminated, truncated, _ = self.env.step(actions)
        self.assertEqual(observations.shape, (2, 34))
        self.assertTrue(torch.isfinite(observations).all())
        self.assertTrue(torch.isfinite(rewards).all())
        self.assertEqual(terminated.shape, (2,))
        self.assertEqual(truncated.shape, (2,))


if __name__ == "__main__":
    unittest.main()
