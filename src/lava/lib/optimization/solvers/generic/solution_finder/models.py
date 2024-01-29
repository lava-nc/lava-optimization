# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import numpy as np
from numpy import typing as npty
from lava.lib.optimization.solvers.generic.dataclasses import (
    ConstraintEnforcing,
    VariablesImplementation,
    CostMinimizer,
)
from lava.lib.optimization.solvers.generic.hierarchical_processes import (
    ContinuousConstraintsProcess,
    DiscreteVariablesProcess,
    ContinuousVariablesProcess,
    CostConvergenceChecker,
)
from lava.lib.optimization.solvers.generic.solution_finder.process import (
    SolutionFinder,
)
from lava.magma.core.resources import (
    CPU,
    Loihi2NeuroCore,
    NeuroCore,
)
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.dense.process import Dense
from lava.proc.sparse.process import Sparse
from lava.lib.optimization.utils.datatype_converter import convert_to_fp
from scipy.sparse import csr_matrix

CPUS = [CPU, "CPU"]
NEUROCORES = [Loihi2NeuroCore, NeuroCore, "Loihi2"]
BACKEND_MSG = f""" was requested as backend. However,
the solver currently supports only Loihi 2 and CPU backends.
These can be specified by calling solve with any of the following:
backend = "CPU"
backend = "Loihi2"
backend = CPU
backend = Loihi2NeuroCore
backend = NeuroCoreS
The explicit resource classes can be imported from
lava.magma.core.resources"""


@implements(proc=SolutionFinder, protocol=LoihiProtocol)
@requires(CPU)
class SolutionFinderModel(AbstractSubProcessModel):
    def __init__(self, proc):
        cost_diagonal = proc.proc_params.get("cost_diagonal")
        cost_coefficients = proc.proc_params.get("cost_coefficients")
        hyperparameters = proc.proc_params.get("hyperparameters")
        discrete_var_shape = proc.proc_params.get("discrete_var_shape")
        continuous_var_shape = proc.proc_params.get("continuous_var_shape")
        problem = proc.proc_params.get("problem")
        backend = proc.proc_params.get("backend")

        # Subprocesses
        self.variables = VariablesImplementation()
        if discrete_var_shape is not None:

            self.variables.discrete = DiscreteVariablesProcess(
                shape=discrete_var_shape,
                cost_diagonal=cost_diagonal,
                cost_off_diagonal=self._get_q_off(cost_coefficients),
                hyperparameters=hyperparameters,
            )

            self.cost_minimizer = None
            self.cost_convergence_check = None
            if cost_coefficients is not None:
                weights = self._get_q_off(cost_coefficients)

                self.cost_minimizer = CostMinimizer(
                    Sparse(
                        weights=csr_matrix(weights),
                        num_message_bits=24,
                    )
                )
                if 1 in cost_coefficients.keys():
                    q_diag = (
                        cost_coefficients[1].init
                        + cost_coefficients[2].init.diagonal()
                    )
                else:
                    q_diag = cost_coefficients[2].init.diagonal()
                self.variables.importances = q_diag
                self.cost_convergence_check = CostConvergenceChecker(
                    shape=discrete_var_shape
                )

                # Connect processes
                self.cost_minimizer.gradient_out.connect(
                    self.variables.gradient_in
                )
                self.variables.state_out.connect(self.cost_minimizer.state_in)
                self.variables.local_cost.connect(
                    self.cost_convergence_check.cost_components
                )
            proc.vars.variables_assignment.alias(
                self.variables.variables_assignment
            )
            # Note: Total min cost=cost_min_first_byte<<24+cost_min_last_bytes
            proc.vars.cost_last_bytes.alias(
                self.cost_convergence_check.cost_last_bytes
            )
            proc.vars.cost_first_byte.alias(
                self.cost_convergence_check.cost_first_byte
            )
            self.cost_convergence_check.cost_out_last_bytes.connect(
                proc.out_ports.cost_out_last_bytes
            )
            self.cost_convergence_check.cost_out_first_byte.connect(
                proc.out_ports.cost_out_first_byte
            )

        elif continuous_var_shape:
            self.constraints = ConstraintEnforcing()
            self.variables.continuous = ContinuousVariablesProcess(
                shape=continuous_var_shape,
                hyperparameters=hyperparameters,
                backend=backend,
                problem=problem,
            )

            self.constraints.continuous = ContinuousConstraintsProcess(
                shape_in=continuous_var_shape,
                shape_out=continuous_var_shape,
                backend=backend,
                hyperparameters=hyperparameters,
                problem=problem,
            )
            self.cost_minimizer = None
            Q_pre = problem.hessian
            if cost_coefficients is not None:
                if backend in CPUS:
                    self.cost_minimizer = CostMinimizer(
                        Sparse(
                            weights=csr_matrix(Q_pre),
                            num_message_bits=64,
                        )
                    )
                elif backend in NEUROCORES:
                    A_pre = problem.constraint_hyperplanes_eq
                    # weights need to converted to fixed_pt first
                    Q_pre_fp_man, Q_pre_fp_exp = convert_to_fp(Q_pre, 8)
                    _, A_pre_fp_exp = convert_to_fp(A_pre, 8)
                    Q_pre_fp_man = (Q_pre_fp_man // 2) * 2
                    correction_exp = min(A_pre_fp_exp, Q_pre_fp_exp)
                    Q_exp_new = -correction_exp + Q_pre_fp_exp
                    self.cost_minimizer = CostMinimizer(
                        Sparse(
                            weights=csr_matrix(Q_pre_fp_man),
                            weight_exp=Q_exp_new,
                            num_message_bits=24,
                        )
                    )
                else:
                    raise NotImplementedError(str(backend) + BACKEND_MSG)
            # Connect processes
            self.cost_minimizer.gradient_out.connect(
                self.variables.gradient_in_cont
            )
            self.variables.state_out_cont.connect(self.cost_minimizer.state_in)
            self.variables.state_out_cont.connect(self.constraints.state_in)
            self.constraints.state_out.connect(self.variables.gradient_in_cont)

            proc.vars.variables_assignment.alias(
                self.variables.variables_assignment_cont
            )

    def _get_q_off(self, cost_coefficients) -> npty.ArrayLike:
        """Returns the off-diagonal elements of the Q matrix"""

        q_off_diag = cost_coefficients[2].init * np.logical_not(
            np.eye(*cost_coefficients[2].init.shape))
        return q_off_diag
