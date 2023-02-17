# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import numpy as np
from lava.lib.optimization.solvers.generic.dataclasses import \
	(
	VariablesImplementation, CostMinimizer
	)
from lava.lib.optimization.solvers.generic.hierarchical_processes import \
	(
	DiscreteVariablesProcess, ContinuousVariablesProcess,
	CostConvergenceChecker
	)
from lava.lib.optimization.solvers.generic.solution_finder.process import \
	SolutionFinder
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.dense.process import Dense


@implements(proc=SolutionFinder, protocol=LoihiProtocol)
@requires(CPU)  # todo enable also Loihi2
class SolutionFinderModel(AbstractSubProcessModel):
	def __init__(self, proc):
		cost_diagonal = proc.proc_params.get("cost_diagonal")
		cost_coefficients = proc.proc_params.get("cost_coefficients")
		hyperparameters = proc.proc_params.get("hyperparameters")
		discrete_var_shape = proc.proc_params.get("discrete_var_shape")
		continuous_var_shape = proc.proc_params.get("continuous_var_shape")

		# Subprocesses
		self.variables = VariablesImplementation()
		if discrete_var_shape:
			hyperparameters.update(dict(
					init_state=self._get_init_state(hyperparameters,
													cost_coefficients,
													discrete_var_shape)))
			self.variables.discrete = DiscreteVariablesProcess(
					shape=discrete_var_shape,
					cost_diagonal=cost_diagonal,
					hyperparameters=hyperparameters)

		if continuous_var_shape:
			self.variables.continuous = ContinuousVariablesProcess(
					shape=continuous_var_shape
					)

		self.cost_minimizer = None
		self.cost_convergence_check = None
		if cost_coefficients is not None:
			self.cost_minimizer = CostMinimizer(
					Dense(
							# todo just using the last coefficient for now
							weights=cost_coefficients[2].init,
							num_message_bits=24
							)
					)
			self.variables.importances = cost_coefficients[1].init
			self.cost_convergence_check = CostConvergenceChecker(
					shape=discrete_var_shape)

		# Connect processes
		self.cost_minimizer.gradient_out.connect(self.variables.gradient_in)
		self.variables.state_out.connect(self.cost_minimizer.state_in)
		self.variables.local_cost.connect(
				self.cost_convergence_check.cost_components)

		proc.vars.variables_assignment.alias(
				self.variables.variables_assignment)
		self.cost_convergence_check.update_buffer.connect(
				proc.out_ports.cost_out)

	def _get_init_state(self, hyperparameters, cost_coefficients,
						discrete_var_shape):
		init_value = hyperparameters.get("init_value", np.zeros(
				discrete_var_shape, dtype=int))
		q_off_diag = cost_coefficients[2].init
		q_diag = cost_coefficients[1].init
		return q_off_diag @ init_value + q_diag
