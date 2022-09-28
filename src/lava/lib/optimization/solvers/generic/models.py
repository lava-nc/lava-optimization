# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


import numpy as np
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort, PyRefPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.dense.process import Dense

from lava.lib.optimization.solvers.generic.dataclasses import (
    CostMinimizer,
    VariablesImplementation,
    MacroStateReader,
)
from lava.lib.optimization.solvers.generic.hierarchical_processes import (
    SatConvergenceChecker,
    DiscreteVariablesProcess,
    ContinuousVariablesProcess,
)
from lava.lib.optimization.solvers.generic.hierarchical_processes import \
    CostConvergenceChecker
from lava.proc.read_gate.process import ReadGate
from lava.proc.cost_integrator.process import CostIntegrator
from lava.lib.optimization.solvers.generic.hierarchical_processes import \
    StochasticIntegrateAndFire
from lava.lib.optimization.solvers.generic.monitoring_processes \
    .solution_readout.process import SolutionReadout


class SolverModelBuilder:
    def __init__(self, solver_process):
        self.constructor = None
        self.solver_process = solver_process

    def create_constructor(self, target_cost):
        def constructor(self, proc):
            variables = VariablesImplementation()
            if hasattr(proc, "discrete_variables"):
                variables.discrete = DiscreteVariablesProcess(
                    shape=proc.discrete_variables.shape,
                    cost_diagonal=proc.cost_diagonal)
            if hasattr(proc, "continuous_variables"):
                variables.continuous = ContinuousVariablesProcess(
                    shape=proc.continuous_variables.shape
                )

            macrostate_reader = MacroStateReader(
                ReadGate(target_cost=target_cost),
                SolutionReadout(shape=proc.variable_assignment.shape,
                                target_cost=target_cost),
            )
            if proc.problem.constraints:
                macrostate_reader.sat_convergence_check = SatConvergenceChecker(
                    shape=proc.variable_assignment.shape
                )
                proc.vars.feasibility.alias(macrostate_reader.satisfaction)
            if hasattr(proc, "cost_coefficients"):
                cost_minimizer = CostMinimizer(
                    Dense(
                        # todo just using the last coefficient for now
                        weights=proc.cost_coefficients[2].init
                    )
                )
                variables.importances = proc.cost_coefficients[1].init
                macrostate_reader.cost_convergence_check = CostConvergenceChecker(
                    shape=proc.variable_assignment.shape
                )
                variables.local_cost.connect(macrostate_reader.cost_in)
                proc.vars.optimality.alias(macrostate_reader.min_cost)

            # Variable aliasing
            proc.vars.variable_assignment.alias(macrostate_reader.solution)
            # Connect processes
            macrostate_reader.update_buffer.connect(
                macrostate_reader.read_gate_in_port
            )
            # macrostate_reader.cost_convergence_check.s_out.connect(
            #     variables.discrete.)
            macrostate_reader.read_gate_do_readout.connect(
                macrostate_reader.solution_readout.read_solution
            )
            macrostate_reader.ref_port.connect_var(
                variables.variables_assignment

            )
            cost_minimizer.gradient_out.connect(variables.gradient_in)
            variables.state_out.connect(cost_minimizer.state_in)
            self.macrostate_reader = macrostate_reader
            self.variables = variables
            self.cost_minimizer = cost_minimizer

        self.constructor = constructor

    @property
    def solver_model(self):
        SolverModel = type(
            "OptimizationSolverModel",
            (AbstractSubProcessModel,),
            {"__init__": self.constructor},
        )
        setattr(SolverModel, "implements_process", self.solver_process)
        # setattr(SolverModel, 'implements_protocol', protocol)
        # Get requirements of parent class
        super_res = SolverModel.required_resources.copy()
        # Set new requirements on this cls to not overwrite parent class
        # requirements
        setattr(SolverModel, "required_resources", super_res + [CPU])
        setattr(SolverModel, "implements_protocol", LoihiProtocol)
        return SolverModel

