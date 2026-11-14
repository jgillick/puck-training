import unittest

import genesis as gs
import torch

from puck.env import (
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

    def test_imports_all_actuators_and_maps_neutral_action_to_zero_torque(self) -> None:
        self.assertEqual(self.env.actuator_manager.num_dofs, 8)
        self.assertEqual(self.env.action_space.shape, (8,))
        self.assertEqual(self.observations.shape, (2, 34))
        self.assertTrue(torch.isfinite(self.observations).all())

        zero_actions = torch.zeros((2, 8), device=gs.device)
        torques = self.env.action_manager.process_actions(zero_actions)
        self.assertTrue(torch.equal(torques, zero_actions))

    def test_actions_are_bounded_by_force_ranges_from_robot_model(self) -> None:
        actions = torch.zeros((2, 8), device=gs.device)
        actions[0] = 2.0
        actions[1] = -2.0
        torques = self.env.action_manager.process_actions(actions)
        lower, upper = self.env.robot.get_dofs_force_range(
            self.env.actuator_manager.dofs_idx
        )
        self.assertTrue(torch.allclose(torques[0], upper))
        self.assertTrue(torch.allclose(torques[1], lower))

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

        rewards = _leg_symmetry_reward(positions, names)
        self.assertTrue(torch.allclose(rewards, torch.ones_like(rewards)))

        positions[0, names.index("knee_l")] += 0.5
        rewards = _leg_symmetry_reward(positions, names)
        self.assertLess(float(rewards[0]), float(rewards[1]))

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
