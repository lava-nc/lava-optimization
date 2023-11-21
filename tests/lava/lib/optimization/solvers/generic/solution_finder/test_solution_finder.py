# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import unittest
import os

import numpy as np
from lava.lib.optimization.problems.problems import QP
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
        root = os.path.dirname(os.path.abspath(__file__))
        root = os.path.join(root, os.pardir)
        qp_data = np.load(root + "/data/qp/ex_qp_small.npz")
        Q, self.A, p, k = [qp_data[i] for i in qp_data]
        p, self.k = np.squeeze(p), np.squeeze(k)
        self.problem = QP(
            hessian=Q,
            linear_offset=p,
            equality_constraints_weights=self.A,
            equality_constraints_biases=self.k,
        )

        # self.problem = QUBO(
        #    np.array(
        #        [[-5, 2, 4, 0], [2, -3, 1, 0], [4, 1, -8, 5], [0, 0, 5, -6]]
        #    )
        # )
        from lava.magma.core.process.variable import Var

        cc = {
            2: Var(
                shape=self.problem.cost.coefficients[2].shape,
                init=self.problem.cost.coefficients[2],
            ),
        }

        # Create processes.
        self.solution_finder = SolutionFinder(
            continuous_var_shape=(
                self.problem.variables._continuous.num_variables,),
            cost_diagonal=self.problem.cost.coefficients[2].diagonal(),
            cost_coefficients=cc,
            constraints=None,
            hyperparameters={},
            discrete_var_shape=None,
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

    @unittest.skip("cost_convergence_checker is only available for discrete "
                   "variable models such as QUBO. And CPU backend of QUBO "
                   "solver is temporarily deactivated.")
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

    def test_qp_cost_defines_num_vars_in_discrete_variables_process(self):
        self.solution_finder.run(RunSteps(5), run_cfg=self.run_cfg)
        self.solution_finder.stop()
        pm = self.solution_finder.model_class(self.solution_finder)
        self.assertEqual(
            pm.variables.continuous.num_variables,
            self.problem.variables.continuous.num_variables,
        )
        self.assertEqual(
            self.solution_finder.variables_assignment.size,
            self.problem.variables.continuous.num_variables,
        )

    @unittest.skip("CPU backend of QUBO solver is temporarily deactivated.")
    def test_qubo_cost_defines_biases(self):
        self.solution_finder.run(RunSteps(5), run_cfg=self.run_cfg)
        self.solution_finder.stop()
        pm = self.solution_finder.model_class(self.solution_finder)
        condition = (
            pm.variables.continuous.cost_diagonal
            == self.problem.cost.get_coefficient(2).diagonal()
        ).all()
        self.assertTrue(condition)


if __name__ == "__main__":
    unittest.main()
