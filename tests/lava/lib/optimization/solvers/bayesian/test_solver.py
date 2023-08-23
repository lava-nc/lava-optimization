# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: LGPL-2.1-or-later
# See: https://spdx.org/licenses/

import numpy as np
import random
from schema import SchemaError
import unittest

from lava.lib.optimization.problems.bayesian.models import (
    SingleInputFunction
)
from lava.lib.optimization.solvers.bayesian.solver import BayesianSolver


@unittest.skip("Failing due to a change in numpy, to be investaget further")
class TeatSolvers(unittest.TestCase):
    """Test initialization and runtime of the BayesianSolver class

    Refer to Bayesian solver.py for more information
    """

    def setUp(self) -> None:
        """set up general parameters for process tests"""

        # set default seed for consistent test runs
        random.seed(0)

        # create a variety of valid acquisition function configs
        valid_acq_funcs: list[str] = [
            "LCB", "EI", "PI", "gp_hedge", "EIps", "PIps"
        ]
        self.valid_acq_func_configs: list[dict] = [
            {"type": t} for t in valid_acq_funcs
        ]

        # create a variety of valid acquisition optimizer configs
        valid_acq_opts: list[str] = ["sampling", "lbfgs"]
        self.valid_acq_opt_configs: list[dict] = [
            {"type": t} for t in valid_acq_opts
        ]

        # create valid examples for all types of search space dimensions
        self.valid_continuous_dimension = np.array([
            "continuous", -20.0, 20.0, 0, "continuous_var0"
        ], dtype=object)
        self.valid_integer_dimension = np.array([
            "integer", -10, 10, 0, "discrete_var0"
        ], dtype=object)
        self.valid_categorical_dimension = np.array([
            "categorical", 0, 0, [x / 4 for x in range(10)], "cat_var0"
        ], dtype=object)
        self.valid_ss = np.array([
            self.valid_continuous_dimension,
            self.valid_integer_dimension,
            self.valid_categorical_dimension
        ], dtype=object)

        # create a variety of valid estimator configs
        valid_ests: list[str] = ["GP"]
        self.valid_est_configs: list[dict] = [
            {"type": t} for t in valid_ests
        ]

        # create a variety of valid initial point generator configs
        valid_ips: list[str] = [
            "random", "sobol", "halton", "hammersly", "lhs", "grid"
        ]
        self.valid_ip_configs: list[dict] = [
            {"type": t} for t in valid_ips
        ]

    def test_solver_bayesian_valid(self) -> None:
        """test BayesianSolver initialization with valid parameters"""

        # test all of the valid acquisition functions
        for config in self.valid_acq_func_configs:
            opt = BayesianSolver(
                acq_func_config=config,
                acq_opt_config=self.valid_acq_opt_configs[0],
                ip_gen_config=self.valid_ip_configs[0],
                num_ips=10,
                seed=10,
                est_config=self.valid_est_configs[0],
                num_objectives=1
            )
            self.assertEqual(
                opt.acquisition_function_config['type'],
                config['type']
            )
            del opt

        # test all of the valid acquisition optimizers
        for config in self.valid_acq_opt_configs:
            opt = BayesianSolver(
                acq_func_config=self.valid_acq_func_configs[0],
                acq_opt_config=config,
                ip_gen_config=self.valid_ip_configs[0],
                num_ips=10,
                seed=10,
                est_config=self.valid_est_configs[0],
                num_objectives=1
            )
            self.assertEqual(
                opt.acquisition_optimizer_config['type'],
                config['type']
            )
            del opt

        # test all of the valid enable plotting settings
        for flag in [True, False]:
            opt = BayesianSolver(
                acq_func_config=self.valid_acq_func_configs[0],
                acq_opt_config=self.valid_acq_opt_configs[0],
                ip_gen_config=self.valid_ip_configs[0],
                num_ips=10,
                seed=10,
                est_config=self.valid_est_configs[0],
                num_objectives=1
            )
            del opt

        # test all valid initial point generator configs
        for config in self.valid_ip_configs:
            opt = BayesianSolver(
                acq_func_config=self.valid_acq_func_configs[0],
                acq_opt_config=self.valid_acq_opt_configs[0],
                ip_gen_config=config,
                num_ips=10,
                seed=10,
                est_config=self.valid_est_configs[0],
                num_objectives=1
            )
            self.assertEqual(opt.ip_gen_config['type'], config['type'])
            del opt

        # test log directory, number of initial points, seed,
        # and number of objectives
        opt = BayesianSolver(
            acq_func_config=self.valid_acq_func_configs[0],
            acq_opt_config=self.valid_acq_opt_configs[0],
            ip_gen_config=config,
            num_ips=10,
            seed=10,
            est_config=self.valid_est_configs[0],
            num_objectives=1
        )
        self.assertEqual(opt.num_initial_points, 10)
        self.assertEqual(opt.seed, 10)
        self.assertEqual(opt.num_objectives, 1)
        del opt

        # test all valid est configs
        for config in self.valid_est_configs:
            opt = BayesianSolver(
                acq_func_config=self.valid_acq_func_configs[0],
                acq_opt_config=self.valid_acq_opt_configs[0],
                ip_gen_config=self.valid_ip_configs[0],
                num_ips=10,
                seed=10,
                est_config=config,
                num_objectives=1
            )
            self.assertEqual(opt.est_config['type'], config['type'])
            del opt

    def test_solver_bayesian_invalid(self) -> None:
        """test BayesianSolver initialization with invalid parameters"""

        # test initialization with invalid acquisition function config
        with self.assertRaises(SchemaError):
            BayesianSolver(
                acq_func_config={"type": "neuro"},
                acq_opt_config=self.valid_acq_opt_configs[0],
                ip_gen_config=self.valid_ip_configs[0],
                num_ips=10,
                seed=10,
                est_config=self.valid_est_configs[0],
                num_objectives=1
            )
            BayesianSolver(
                acq_func_config=int,
                acq_opt_config=self.valid_acq_opt_configs[0],
                ip_gen_config=self.valid_ip_configs[0],
                num_ips=10,
                seed=10,
                est_config=self.valid_est_configs[0],
                num_objectives=1
            )

        # test initialization with invalid acquisition optimizer config
        with self.assertRaises(SchemaError):
            BayesianSolver(
                acq_func_config=self.valid_acq_func_configs[0],
                acq_opt_config={"type": "neuro"},
                ip_gen_config=self.valid_ip_configs[0],
                num_ips=10,
                seed=10,
                est_config=self.valid_est_configs[0],
                num_objectives=1
            )
            BayesianSolver(
                acq_func_config=self.valid_acq_func_configs[0],
                acq_opt_config=float,
                ip_gen_config=self.valid_ip_configs[0],
                num_ips=10,
                seed=10,
                est_config=self.valid_est_configs[0],
                num_objectives=1
            )

        # test initialization with invalid initial point generator config
        with self.assertRaises(SchemaError):
            BayesianSolver(
                acq_func_config=self.valid_acq_func_configs[0],
                acq_opt_config=self.valid_acq_opt_configs[0],
                ip_gen_config={"type": 0},
                num_ips=10,
                seed=10,
                est_config=self.valid_est_configs[0],
                num_objectives=1
            )
            BayesianSolver(
                acq_func_config=self.valid_acq_func_configs[0],
                acq_opt_config=self.valid_acq_opt_configs[0],
                ip_gen_config=str,
                num_ips=10,
                seed=10,
                est_config=self.valid_est_configs[0],
                num_objectives=1
            )

        # test initialization with invalid number of initial points
        with self.assertRaises(SchemaError):
            BayesianSolver(
                acq_func_config=self.valid_acq_func_configs[0],
                acq_opt_config=self.valid_acq_opt_configs[0],
                ip_gen_config=self.valid_ip_configs[0],
                num_ips=0,
                seed=10,
                est_config=self.valid_est_configs[0],
                num_objectives=1
            )
            BayesianSolver(
                acq_func_config=self.valid_acq_func_configs[0],
                acq_opt_config=self.valid_acq_opt_configs[0],
                ip_gen_config=self.valid_ip_configs[0],
                num_ips=dict,
                seed=10,
                est_config=self.valid_est_configs[0],
                num_objectives=1
            )

        # test initialization with invalid number of objectives
        with self.assertRaises(SchemaError):
            BayesianSolver(
                acq_func_config=self.valid_acq_func_configs[0],
                acq_opt_config=self.valid_acq_opt_configs[0],
                ip_gen_config=self.valid_ip_configs[0],
                num_ips=10,
                seed=10,
                est_config=self.valid_est_configs[0],
                num_objectives=0
            )

        # test initialization with invalid seed
        with self.assertRaises(SchemaError):
            BayesianSolver(
                acq_func_config=self.valid_acq_func_configs[0],
                acq_opt_config=self.valid_acq_opt_configs[0],
                ip_gen_config=self.valid_ip_configs[0],
                num_ips=10,
                seed=dict,
                est_config=self.valid_est_configs[0],
                num_objectives=1
            )

    def test_valid_solve(self) -> None:
        """test the solving method with valid arguments"""

        search_space: np.ndarray = np.array([
            ["categorical", np.nan, np.nan, [x / 4 for x in range(10)], "x1"],
        ], dtype=object)

        problem = SingleInputFunction()

        solver = BayesianSolver(
            acq_func_config={"type": "gp_hedge"},
            acq_opt_config={"type": "auto"},
            ip_gen_config={"type": "random"},
            num_ips=1,
            seed=1
        )

        solver.solve(
            name="testing",
            num_iter=2,
            problem=problem,
            search_space=search_space,
        )

    def test_invalid_solve(self) -> None:
        """test the solving method with invalid arguments"""

        search_space: np.ndarray = np.array([
            ["categorical", np.nan, np.nan, [x / 4 for x in range(10)], "x1"],
        ], dtype=object)

        problem = SingleInputFunction()

        solver = BayesianSolver(
            acq_func_config={"type": "gp_hedge"},
            acq_opt_config={"type": "auto"},
            ip_gen_config={"type": "random"},
            num_ips=1,
            seed=1
        )

        with self.assertRaises(SchemaError):
            # should catch problem that doesn't extend the base problem
            solver.solve(
                name="testing",
                num_iter=2,
                problem=int,
                search_space=search_space,
            )

            # should catch number of iterations less than 1
            solver.solve(
                name="testing",
                num_iter=0,
                problem=problem,
                search_space=search_space,
            )

            # should catch an experiment name that isn't a string
            solver.solve(
                name=float,
                num_iter=2,
                problem=problem,
                search_space=search_space,
            )

            # should catch the invalid search space type
            search_space: np.ndarray = np.array([
                ["dsdsds", np.nan, np.nan, [x / 4 for x in range(10)], "x1"],
            ], dtype=object)
            solver.solve(
                name="testing",
                num_iter=2,
                problem=problem,
                search_space=search_space,
            )

            # should catch error where search space size doesn't match the
            # problem
            search_space: np.ndarray = np.array([
                ["categorical", np.nan, np.nan, [x for x in range(10)], "x1"],
                ["categorical", np.nan, np.nan, [x for x in range(10)], "x1"]
            ], dtype=object)
            solver.solve(
                name="testing",
                num_iter=2,
                problem=problem,
                search_space=search_space,
            )

            # should catch error where the identifier isn't a string
            search_space: np.ndarray = np.array([
                ["categorical", np.nan, np.nan, [x for x in range(10)], int],
            ], dtype=object)
            solver.solve(
                name="testing",
                num_iter=2,
                problem=problem,
                search_space=search_space,
            )

            # should catch error where each dimension doesn't contain all
            # fields
            search_space: np.ndarray = np.array([
                ["categorical", np.nan, np.nan, [x / 4 for x in range(10)]],
            ], dtype=object)
            solver.solve(
                name="testing",
                num_iter=2,
                problem=problem,
                search_space=search_space,
            )

            # should catch error where categories are empty
            search_space: np.ndarray = np.array([
                ["categorical", np.nan, np.nan, int, "id"],
            ], dtype=object)
            solver.solve(
                name="testing",
                num_iter=2,
                problem=problem,
                search_space=search_space,
            )

            # should catch error where min and max are not floats
            search_space: np.ndarray = np.array([
                ["int", 0, 10, np.nan, "id"],
            ], dtype=object)
            solver.solve(
                name="testing",
                num_iter=2,
                problem=problem,
                search_space=search_space,
            )
            search_space: np.ndarray = np.array([
                ["float", 0, 10, np.nan, "id"],
            ], dtype=object)
            solver.solve(
                name="testing",
                num_iter=2,
                problem=problem,
                search_space=search_space,
            )


if __name__ == "__main__":
    unittest.main()
