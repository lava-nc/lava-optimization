# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import unittest

import numpy as np
from lava.lib.optimization.problems.problems import QUBO
from lava.lib.optimization.solvers.generic.solver import (
	OptimizationSolver, SolverConfig
	)


class TestParallelOptimizationSolver(unittest.TestCase):

	def test_parallel_run(self):
		q = np.array([[-5, 2, 4, 0],
		              [2, -3, 1, 0],
		              [4, 1, -8, 5],
		              [0, 0, 5, -6]])
		problem = QUBO(q)
		solution = np.asarray([1, 0, 0, 1]).astype(int)
		solver = OptimizationSolver(problem=problem)
		solution_cost = solution @ q @ solution

		np.random.seed(2)

		config = SolverConfig(
				timeout=50,
				target_cost=-11,
				backend="CPU",
				hyperparameters=[{
						"neuron_model"    : "scif",
						'noise_amplitude' : 2,
						'noise_precision' : 5,
						'sustained_on_tau': -3
						},
						{
						"neuron_model": "nebm"
						},
						{
						"neuron_model"    : "scif",
						'noise_amplitude' : 1,
						'noise_precision' : 5,
						'sustained_on_tau': -3
						},
						]
				)
		report = solver.solve(config=config)
		print(report)
		self.assertEqual(report.best_cost, solution_cost)


if __name__ == "__main__":
	unittest.main()
