# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

# Initialization tests for all the processes in QP
import unittest
import numpy as np

from lava.lib.optimization.solvers.generic.qp.processes import (
    ConstraintCheck,
    ConstraintNeurons,
    QPDense,
    SolutionNeurons,
    GradientDynamics,
    ProjectedGradientNeuronsPIPGeq,
    ProportionalIntegralNeuronsPIPGeq,
)


class TestProcessesFloatingPoint(unittest.TestCase):
    """Initializations Tests of all processes of the QP solver in floating
    point
    All tests check if the vars are properly assigned in the processes and if
    the ports are shaped properly. Refer to QP models.py in qp/solver repo to
    understand behaviours.
    """

    def test_process_constraint_directions(self):
        process = QPDense()
        self.assertEqual(process.vars.weights.get() == 0, True)

        weights = np.array([[2, 3, 6], [43, 3, 2]])
        process = QPDense(shape=weights.shape, weights=weights)
        self.assertEqual(np.all(process.vars.weights.get() == weights), True)
        self.assertEqual(
            np.all(process.s_in.shape == (weights.shape[1], 1)), True
        )
        self.assertEqual(
            np.all(process.a_out.shape == (weights.shape[0], 1)), True
        )

    def test_process_constraint_neurons(self):
        process = ConstraintNeurons()
        self.assertEqual(process.vars.thresholds.get() == 0, True)
        inp_bias = np.array([[2, 4, 6]]).T
        process = ConstraintNeurons(shape=inp_bias.shape, thresholds=inp_bias)
        self.assertEqual(
            np.all(process.vars.thresholds.get() == inp_bias), True
        )
        self.assertEqual(
            np.all(process.a_in.shape == (inp_bias.shape[0], 1)), True
        )

    def test_process_solution_neurons(self):
        init_sol = np.array([[2, 4, 6, 4, 1]]).T
        p = np.array([[4, 3, 2, 1, 1]]).T
        alpha, beta, alpha_d, beta_g = 3, 2, 100, 100
        process = SolutionNeurons(
            shape=init_sol.shape,
            qp_neurons_init=init_sol,
            grad_bias=p,
            alpha=alpha,
            beta=beta,
            alpha_decay_schedule=alpha_d,
            beta_growth_schedule=beta_g,
        )
        self.assertEqual(
            np.all(process.vars.qp_neuron_state.get() == init_sol), True
        )
        self.assertEqual(np.all(process.vars.grad_bias.get() == p), True)
        self.assertEqual(np.all(process.vars.alpha.get() == alpha), True)
        self.assertEqual(np.all(process.vars.beta.get() == beta), True)
        self.assertEqual(
            process.vars.alpha_decay_schedule.get() == alpha_d, True
        )
        self.assertEqual(
            process.vars.beta_growth_schedule.get() == beta_g, True
        )
        self.assertEqual(process.vars.decay_counter.get() == 0, True)
        self.assertEqual(process.vars.growth_counter.get() == 0, True)
        self.assertEqual(
            np.all(process.a_in_qc.shape == (p.shape[0], 1)), True
        )
        self.assertEqual(
            np.all(process.a_in_cn.shape == (p.shape[0], 1)), True
        )
        self.assertEqual(
            np.all(process.s_out_qc.shape == (p.shape[0], 1)), True
        )
        self.assertEqual(
            np.all(process.s_out_cc.shape == (p.shape[0], 1)), True
        )

    def test_process_constraint_check(self):
        A = np.array([[2, 3, 6], [43, 3, 2]])

        b = np.array([[2, 4]]).T
        process = ConstraintCheck(constraint_matrix=A, constraint_bias=b)
        self.assertEqual(
            np.all(process.vars.constraint_matrix.get() == A), True
        )
        self.assertEqual(np.all(process.vars.constraint_bias.get() == b), True)
        self.assertEqual(np.all(process.s_in.shape == (A.shape[1], 1)), True)
        self.assertEqual(np.all(process.s_out.shape == (A.shape[0], 1)), True)

    def test_process_gradient_dynamics(self):
        P = np.array([[2, 43, 2], [43, 3, 4], [2, 4, 1]])

        A_T = np.array([[2, 3, 6], [43, 3, 2]]).T

        init_sol = np.array([[2, 4, 6]]).T
        p = np.array([[4, 3, 2]]).T
        alpha, beta, alpha_d, beta_g = 3, 2, 100, 100
        process = GradientDynamics(
            hessian=P,
            constraint_matrix_T=A_T,
            qp_neurons_init=init_sol,
            grad_bias=p,
            alpha=alpha,
            beta=beta,
            alpha_decay_schedule=alpha_d,
            beta_growth_schedule=beta_g,
        )
        self.assertEqual(
            np.all(process.vars.constraint_matrix_T.get() == A_T), True
        )
        self.assertEqual(np.all(process.vars.hessian.get() == P), True)
        self.assertEqual(
            np.all(process.vars.qp_neuron_state.get() == init_sol), True
        )
        self.assertEqual(np.all(process.vars.grad_bias.get() == p), True)
        self.assertEqual(np.all(process.vars.alpha.get() == alpha), True)
        self.assertEqual(np.all(process.vars.beta.get() == beta), True)
        self.assertEqual(
            process.vars.alpha_decay_schedule.get() == alpha_d, True
        )
        self.assertEqual(
            process.vars.beta_growth_schedule.get() == beta_g, True
        )
        self.assertEqual(np.all(process.s_in.shape == (A_T.shape[1], 1)), True)
        self.assertEqual(np.all(process.s_out.shape == (P.shape[0], 1)), True)

    def test_process_projected_gradient_pipgeq_neurons(self):
        init_sol = np.array([[2, 4, 6, 4, 1]]).T
        p = np.array([[4, 3, 2, 1, 1]]).T
        alpha, alpha_d = 3, 100
        process = ProjectedGradientNeuronsPIPGeq(
            shape=init_sol.shape,
            qp_neurons_init=init_sol,
            grad_bias=p,
            alpha=alpha,
            alpha_decay_schedule=alpha_d,
        )
        self.assertEqual(
            np.all(process.vars.qp_neuron_state.get() == init_sol), True
        )
        self.assertEqual(np.all(process.vars.grad_bias.get() == p), True)
        self.assertEqual(np.all(process.vars.alpha.get() == alpha), True)
        self.assertEqual(
            process.vars.alpha_decay_schedule.get() == alpha_d, True
        )
        # self.assertEqual(process.decay_counter.get() == 0, True)
        self.assertEqual(np.all(process.a_in.shape == (p.shape[0],)), True)
        self.assertEqual(np.all(process.s_out.shape == (p.shape[0],)), True)

    def test_process_proportional_integral_pipgeq_neurons(self):
        init_sol = np.array([[2, 4, 6, 4, 1]]).T
        p = np.array([[4, 3, 2, 1, 1]]).T
        beta, beta_g = 2, 100
        process = ProportionalIntegralNeuronsPIPGeq(
            shape=init_sol.shape,
            constraint_neurons_init=init_sol,
            thresholds=p,
            beta=beta,
            beta_growth_schedule=beta_g,
        )
        self.assertEqual(
            np.all(process.vars.constraint_neuron_state.get() == init_sol),
            True,
        )
        self.assertEqual(np.all(process.vars.constraint_bias.get() == p), True)
        self.assertEqual(np.all(process.vars.beta.get() == beta), True)
        self.assertEqual(
            process.vars.beta_growth_schedule.get() == beta_g, True
        )
        # self.assertEqual(process.growth_counter.get() == 0, True)
        self.assertEqual(np.all(process.a_in.shape == (p.shape[0],)), True)
        self.assertEqual(np.all(process.s_out.shape == (p.shape[0],)), True)


if __name__ == "__main__":
    unittest.main()
