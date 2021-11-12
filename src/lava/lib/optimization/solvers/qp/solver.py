# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import time
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.lib.optimization.problems.problems import QP

from lava.lib.optimization.solvers.qp.models import (
    ConstraintCheck,
    GradientDynamics,
)


# Future inheritance from OptimizationSolver class
class QPSolver:
    """Solve Full QP by connecting two LAVA processes GradDynamics and
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
        self, alpha, beta, alpha_decay_schedule, beta_growth_schedule
    ):
        self.alpha = alpha
        self.beta = beta
        self.beta_g = beta_growth_schedule
        self.alpha_d = alpha_decay_schedule

    def solve(self, problem: QP, iterations=400):
        """solves the supplied QP problem

        Parameters
        ----------
        problem : QP
            The QP containing the matrices that set up the problem
        iterations : int, optional
            Number of iterations for which QP has to run, by default 400
        """
        Q, p, = (
            problem._Q,
            problem._p,
        )
        if problem._A is not None:
            A, k = problem._A, problem._k
            F = np.diag(1 / np.linalg.norm(A, axis=1))
        else:
            A, k = np.zeros((Q.shape[0], Q.shape[1])), np.zeros(
                (Q.shape[0], 1)
            )
            # Dummy preconditioner
            F = np.diag(np.linalg.norm(A, axis=1))
        # Precondition the problem before feeding it into Loihi
        preconditioner_Q = np.sqrt(np.diag(1 / np.linalg.norm(Q, axis=1)))
        Q_pre = preconditioner_Q @ Q @ preconditioner_Q
        p_pre = preconditioner_Q @ p
        A = F @ A @ preconditioner_Q
        k = F @ k
        Q = Q_pre
        p = p_pre
        #####################################################################
        init_sol = np.random.rand(Q.shape[0], 1)
        i_max = iterations
        ConsCheck = ConstraintCheck(constraint_matrix=A, constraint_bias=k)
        GradDyn = GradientDynamics(
            hessian=Q,
            constraint_matrix_T=A.T,
            qp_neurons_init=init_sol,
            grad_bias=p,
            alpha=self.alpha,
            beta=self.beta,
            alpha_decay_schedule=self.alpha_d,
            beta_growth_schedule=self.beta_g,
        )

        # core solver
        GradDyn.a_out.connect(ConsCheck.s_in)
        ConsCheck.a_out.connect(GradDyn.s_in)

        tic = time.time()
        GradDyn.run(
            condition=RunSteps(num_steps=i_max),
            run_cfg=Loihi1SimCfg(select_sub_proc_model=True),
        )
        GradDyn.pause()
        pre_sol = GradDyn.vars.qp_neuron_state.get()
        GradDyn.stop()
        toc = time.time()
        print(
            "[LavaQpOpt][INFO]: The solution after {} iterations is \n \
            {}".format(
                i_max, preconditioner_Q @ pre_sol
            )
        )
        print(
            "\n [LavaQpOpt][INFO]: QP Solver ran in {} seconds".format(
                toc - tic
            )
        )
