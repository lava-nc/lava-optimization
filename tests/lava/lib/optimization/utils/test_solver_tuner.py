# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

import numpy as np

from lava.lib.optimization.problems.problems import QUBO
from lava.lib.optimization.solvers.generic.solver import (
    OptimizationSolver, SolverConfig, SolverReport
)
from lava.lib.optimization.utils.solver_tuner import SolverTuner


def prepare_solver_and_config():
    """Generate an example QUBO workload."""
    q = np.asarray([[-5, 2, 4, 0],
                    [2, -3, 1, 0],
                    [4, 1, -8, 5],
                    [0, 0, 5, -6]])
    qubo_problem = QUBO(q=q)

    solver = OptimizationSolver(qubo_problem)
    config = SolverConfig(
        timeout=1000,
        target_cost=-11,
        backend="CPU",
        hyperparameters={
            "neuron_model": "sa"
        }
    )

    return solver, config


class TestSolverTuner(unittest.TestCase):
    """Unit tests for `SolverTuner` class."""

    def setUp(self) -> None:
        """Setup test environment for `SolverTuner` class."""
        self.search_space = [(0,), (10,)]
        self.params_names = ['temperature']
        self.shuffle = True
        self.seed = 41
        self.solver_tuner = SolverTuner(search_space=self.search_space,
                                        params_names=self.params_names,
                                        shuffle=self.shuffle,
                                        seed=self.seed)

    def test_create_obj(self):
        """Tests correct instantiation of `SolverTuner` object."""
        self.assertIsInstance(self.solver_tuner, SolverTuner)

    def test_search_space_property(self):
        """Tests `search_space` property exists and has the correct value."""
        self.assertEqual(len(self.solver_tuner.search_space),
                         len(self.search_space))
        self.assertTrue(all(x in self.solver_tuner.search_space
                            for x in self.search_space))

    def test_params_names_property(self):
        """Tests `params_names` property exists and has the correct value."""
        self.assertEqual(len(self.solver_tuner.params_names),
                         len(self.params_names))
        self.assertTrue(all(x in self.solver_tuner.params_names
                            for x in self.params_names))

    def test_shuffle_property(self):
        """Tests `shuffle` property exists and has the correct value."""
        self.assertEqual(self.shuffle, self.solver_tuner.shuffle)

    def test_seed_property(self):
        """Tests `seed` property exists and has the correct value."""
        self.assertEqual(self.seed, self.solver_tuner.seed)

    def test_results_property(self):
        """Tests `results` property exists and returns the correct type."""
        self.assertIsInstance(self.solver_tuner.results, np.ndarray)

    def test_shuffle_setter(self):
        """Tests `shuffle` setter sets the correct value."""
        self.solver_tuner.shuffle = not self.shuffle
        self.assertEqual(self.solver_tuner.shuffle, not self.shuffle)

    def test_seed_setter(self):
        """Tests `seed` setter sets the correct value."""
        self.solver_tuner.seed = self.seed + 1
        self.assertEqual(self.solver_tuner.seed, self.seed + 1)

    def test_search_space_setter(self):
        """Tests `search_space` setter sets the correct value."""
        self.solver_tuner.search_space = [self.search_space[0]]
        self.assertEqual(self.solver_tuner.search_space, [self.search_space[0]])

    def test_generate_grid_util(self):
        """Tests `generate_grid` method produces the correct search space."""
        params_domains = {
            'temperature': (0, 10),
        }
        gen_search_space, gen_params_names = SolverTuner.generate_grid(
            params_domains=params_domains
        )
        self.assertEqual(len(gen_search_space), len(self.search_space))
        self.assertTrue(all(x in gen_search_space for x in self.search_space))
        self.assertEqual(len(gen_params_names), len(self.params_names))
        self.assertTrue(all(x in gen_params_names for x in self.params_names))

    @unittest.skip("CPU backend of QUBO solver temporarily disabled.")
    def test_tune_success(self):
        """Tests the correct set of hyper-parameters is found, for a known
        problem."""

        solver, config = prepare_solver_and_config()

        def fitness(report: SolverReport) -> float:
            if report.best_cost <= config.target_cost:
                return -report.best_timestep
            else:
                return -float("inf")

        fitness_target = -16

        hyperparams, success = self.solver_tuner.tune(
            solver=solver,
            fitness_fn=fitness,
            fitness_target=fitness_target,
            config=config
        )

        self.assertIsNotNone(hyperparams)

        correct_best_temperature = 10

        self.assertEqual(hyperparams['temperature'], correct_best_temperature)
        self.assertTrue(success)


if __name__ == "__main__":
    unittest.main()
