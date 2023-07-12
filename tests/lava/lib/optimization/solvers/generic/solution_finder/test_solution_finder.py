# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import unittest

import numpy as np
from lava.lib.optimization.problems.problems import QUBO
from lava.lib.optimization.solvers.generic.read_gate.models import (
    get_read_gate_model_class,
)
from lava.lib.optimization.solvers.generic.read_gate.process import ReadGate
from lava.lib.optimization.solvers.generic.solution_finder.process import (
    SolutionFinder,
)
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg


class TestSolutionFinder(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = QUBO(
            np.array(
                [[-5, 2, 4, 0], [2, -3, 1, 0], [4, 1, -8, 5], [0, 0, 5, -6]]
            )
        )
        self.solution = np.asarray([1, 0, 0, 1]).astype(int)
        from lava.magma.core.process.variable import Var

        cc = {
            2: Var(
                shape=self.problem.cost.coefficients[2].shape,
                init=self.problem.cost.coefficients[2],
            ),
        }

        # Create processes.
        self.solution_finder = SolutionFinder(
            discrete_var_shape=(4,),
            cost_diagonal=self.problem.cost.coefficients[2].diagonal(),
            cost_coefficients=cc,
            constraints=None,
            hyperparameters={},
            continuous_var_shape=None,
            backend="CPU",
            problem=self.problem,
        )

        # Execution configurations.
        ReadGatePyModel = get_read_gate_model_class(1)
        pdict = {ReadGate: ReadGatePyModel}
        self.run_cfg = Loihi2SimCfg(exception_proc_model_map=pdict)
        self.solution_finder._log_config.level = 20

    def test_create_process(self):
        self.assertIsInstance(self.solution_finder, SolutionFinder)

    def test_run(self):
        self.solution_finder.run(RunSteps(5), run_cfg=self.run_cfg)
        self.solution_finder.stop()

    def test_cost_checker_is_connected_to_variables_population(self):
        self.solution_finder.run(RunSteps(5), run_cfg=self.run_cfg)
        self.solution_finder.stop()
        pm = self.solution_finder.model_class(self.solution_finder)
        self.assertIs(
            pm.cost_convergence_check.cost_components.in_connections[
                0
            ].process,
            pm.variables.discrete,
        )

    def test_qubo_cost_defines_num_vars_in_discrete_variables_process(self):
        self.solution_finder.run(RunSteps(5), run_cfg=self.run_cfg)
        self.solution_finder.stop()
        pm = self.solution_finder.model_class(self.solution_finder)
        self.assertEqual(
            pm.variables.discrete.num_variables,
            self.problem.variables.discrete.num_variables,
        )
        self.assertEqual(
            self.solution_finder.variables_assignment.size,
            self.problem.variables.discrete.num_variables,
        )

    def test_qubo_cost_defines_biases(self):
        self.solution_finder.run(RunSteps(5), run_cfg=self.run_cfg)
        self.solution_finder.stop()
        pm = self.solution_finder.model_class(self.solution_finder)
        condition = (
            pm.variables.discrete.cost_diagonal
            == self.problem.cost.get_coefficient(2).diagonal()
        ).all()
        self.assertTrue(condition)


if __name__ == "__main__":
    unittest.main()
