# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import numpy as np
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
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.dense.process import Dense
from lava.proc.sparse.process import Sparse
from lava.lib.optimization.utils.qp_processing import convert_to_fp
from scipy.sparse import csr_matrix

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
        
        # Subprocesses
        self.variables = VariablesImplementation()
        if discrete_var_shape is not None:
            hyperparameters.update(
                dict(
                    init_state=self._get_init_state(
                        hyperparameters, cost_coefficients, discrete_var_shape
                    )
                )
            )
            self.variables.discrete = DiscreteVariablesProcess(
                shape=discrete_var_shape,
                cost_diagonal=cost_diagonal,
                hyperparameters=hyperparameters,
            )

            self.cost_minimizer = None
            self.cost_convergence_check = None
            if cost_coefficients is not None:
                weights =  cost_coefficients[2].init*np.logical_not(
                np.eye(*cost_coefficients[2].init.shape))
                self.cost_minimizer = CostMinimizer(
                    Dense(
                        # todo just using the last coefficient for now
                        weights=weights,
                        num_message_bits=24,
                    )
                )
                if 1 in cost_coefficients.keys():
                    q_diag = cost_coefficients[1].init + cost_coefficients[2].init.diagonal()
                else:
                    q_diag = cost_coefficients[2].init.diagonal()
                self.variables.importances = q_diag
                self.cost_convergence_check = CostConvergenceChecker(
                    shape=discrete_var_shape
                )

                # Connect processes
                self.cost_minimizer.gradient_out.connect(self.variables.gradient_in)
                self.variables.state_out.connect(self.cost_minimizer.state_in)
                self.variables.local_cost.connect(
                    self.cost_convergence_check.cost_components
                )

                proc.vars.variables_assignment.alias(
                    self.variables.variables_assignment
                )
                proc.vars.cost.alias(
                    self.cost_convergence_check.cost
                )
                self.cost_convergence_check.update_buffer.connect(
                    proc.out_ports.cost_out)

        elif continuous_var_shape:
            self.constraints = ConstraintEnforcing()
            self.variables.continuous = ContinuousVariablesProcess(
                shape=continuous_var_shape,
                hyperparameters=hyperparameters,
                problem=problem,
            )

            self.constraints.continuous = ContinuousConstraintsProcess(
                shape_in=continuous_var_shape,
                shape_out=continuous_var_shape,
                hyperparameters=hyperparameters,
                problem=problem,
            )
            self.cost_minimizer = None
            # weights need to converted to fixed_pt first
            A_pre = problem.constraint_hyperplanes_eq
            Q_pre = problem.hessian
            Q_pre_man, Q_pre_fp_exp = convert_to_fp(Q_pre, 8)
            _, A_pre_fp_exp = convert_to_fp(A_pre, 8)
            correction_exp = min(A_pre_fp_exp, Q_pre_fp_exp) 
            Q_exp_new = -correction_exp + Q_pre_fp_exp
            
            if cost_coefficients is not None:
                self.cost_minimizer = CostMinimizer(
                    Sparse(
                        # todo just using the last coefficient for now
                        weights=csr_matrix(Q_pre_man),
                        weight_exp=Q_exp_new,
                        num_message_bits=24,
                    )
                )
            # Connect processes
            self.cost_minimizer.gradient_out.connect(self.variables.gradient_in_cont)
            self.variables.state_out_cont.connect(self.cost_minimizer.state_in)
            self.variables.state_out_cont.connect(self.constraints.state_in)
            self.constraints.state_out.connect(self.variables.gradient_in_cont)
            
            proc.vars.variables_assignment.alias(
                    self.variables.variables_assignment_cont
                )
            
    def _get_init_state(
        self, hyperparameters, cost_coefficients, discrete_var_shape
    ):
        init_value = hyperparameters.get(
            "init_value", np.zeros(discrete_var_shape, dtype=int)
        )
        q_off_diag = cost_coefficients[2].init * np.logical_not(
                np.eye(*cost_coefficients[2].init.shape))
        if 1 in cost_coefficients.keys():
            q_diag = cost_coefficients[1].init + cost_coefficients[2].init.diagonal()
        else:
            q_diag = cost_coefficients[2].init.diagonal()

        return q_off_diag @ init_value + q_diag

    # @staticmethod
    # def _get_initial_value_for_var(
    #     coefficient: npt.ArrayLike, rank: int
    # ) -> npt.ArrayLike:
    #     """Get the value for initializing the coefficient's Var.

    #     Parameters
    #     ----------
    #     coefficient: npt.ArrayLike
    #         A tensor representing one of the coefficients of a cost or
    #         constraints function.
    #     rank: int
    #         The rank of the tensor coefficient.
    #     """
    #     if rank == 1:
    #         return coefficient
    #     if rank == 2:
    #         quadratic_component = coefficient * np.logical_not(
    #             np.eye(*coefficient.shape)
    #         )
    #         return quadratic_component
        
    # @staticmethod
    # def _update_linear_component_var(
    #     vars: ty.Dict[int, AbstractProcessMember],
    #     quadratic_coefficient: npt.ArrayLike,
    # ):
    #     """Update a linear coefficient's Var given a quadratic coefficient.

    #     Parameters
    #     ----------
    #     vars: ty.Dict[int, AbstractProcessMember]
    #         A dictionary where keys are ranks and values are the Lava Vars
    #         corresponding to ranks' coefficients.
    #     quadratic_coefficient: npt.ArrayLike
    #         An array-like tensor of rank 2, corresponds to the coefficient of
    #         the quadratic term on a cost or constraint function.
    #     """
    #     linear_component = quadratic_coefficient.diagonal()
    #     if 1 in vars.keys():
    #         vars[1].init = vars[1].init + linear_component
    #     else:
    #         vars[1] = Var(shape=linear_component.shape, init=linear_component)