# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.lib.optimization.problems.problems import QP
from lava.lib.optimization.solvers.qp.models import (
    ConstraintCheck,
    GradientDynamics,
)


# Future: inheritance from OptimizationSolver class
class QPSolver:
    """Solve Full QP by connecting two Lava processes, GradDynamics and
    ConstraintCheck

        Parameters
        ----------
        alpha : 1-D np.array
            The learning rate for gradient descent
        beta : 1-D np.array
            The learning rate for constraint correction
        alpha_decay_schedule : int, default 10000
            Number of iterations after which one right shift takes place for
            alpha
        beta_growth_schedule : int, default 10000
            Number of iterations after which one left shift takes place for
            beta

    """

    def __init__(
        self,
        alpha: np.ndarray,
        beta: np.ndarray,
        alpha_decay_schedule: int,
        beta_growth_schedule: int,
    ):
        self.alpha = alpha
        self.beta = beta
        self.beta_g = beta_growth_schedule
        self.alpha_d = alpha_decay_schedule

    def solve(self, problem: QP, iterations: int = 400):
        """solves the supplied QP problem

        Parameters
        ----------
        problem : QP
            The QP containing the matrices that set up the problem
        iterations : int, optional
            Number of iterations for which QP has to run, by default 400

        Returns
        --------
        sol : 1-D np.array
            Solution to the quadratic program
        """
        (
            hessian,
            linear_offset,
        ) = (problem.get_hessian, problem.get_linear_offset)
        if problem.get_constraint_hyperplanes is not None:
            constraint_hyperplanes, constraint_biases = (
                problem.get_constraint_hyperplanes,
                problem.get_constraint_biases,
            )

        else:
            constraint_hyperplanes, constraint_biases = np.zeros(
                (hessian.shape[0], hessian.shape[1])
            ), np.zeros((hessian.shape[0], 1))

        init_sol = np.random.rand(hessian.shape[0], 1)
        i_max = iterations
        ConsCheck = ConstraintCheck(
            constraint_matrix=constraint_hyperplanes,
            constraint_bias=constraint_biases,
        )
        GradDyn = GradientDynamics(
            hessian=hessian,
            constraint_matrix_T=constraint_hyperplanes.T,
            qp_neurons_init=init_sol,
            grad_bias=linear_offset,
            alpha=self.alpha,
            beta=self.beta,
            alpha_decay_schedule=self.alpha_d,
            beta_growth_schedule=self.beta_g,
        )

        # core solver
        GradDyn.a_out.connect(ConsCheck.s_in)
        ConsCheck.a_out.connect(GradDyn.s_in)

        GradDyn.run(
            condition=RunSteps(num_steps=i_max),
            run_cfg=Loihi1SimCfg(select_sub_proc_model=True),
        )
        sol = GradDyn.vars.qp_neuron_state.get()
        GradDyn.stop()
        return sol
