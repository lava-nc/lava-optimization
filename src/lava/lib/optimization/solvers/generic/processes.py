# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var


class OptimizationSolverProcess(AbstractProcess):
    """Hierarchical process implementing an optimization solver."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        problem = kwargs.get("problem", None)
        self.continuous_variables = InPort(
            shape=(problem.variables.continuous.num_vars, 2))
        self.discrete_variables = InPort(
            shape=(problem.variables.discrete.num_vars,))
        self.cost_coefficients = []
        for coeff in problem.cost.coefficients:
            self.cost_coefficients.append(InPort(shape=coeff.shape))
        self.cost_augmented_terms = []
        for coeff in problem.cost.augmented_terms:
            self.cost_augmented_terms.append(InPort(shape=coeff.shape))
        self.ineq_constraints = []
        for coeff in problem.constraints.inequality.coefficients:
            self.ineq_constraints.append(InPort(shape=coeff.shape))
        self.eq_constraints = []
        for coeff in problem.constraints.equality.coefficients:
            self.eq_constraints.append(InPort(shape=coeff.shape))
        if problem.discrete.constraints is not None:
            d_constraints = problem.constraints.discrete
            num_dconstraints = len(d_constraints.var_subsets)
            max_arity = len(d_constraints.var_subsets[0])
            self.discrete_constraints_relations = InPort(
                shape=(len(d_constraints.relations), max_arity))
            subsets_shape = (num_dconstraints,) + (
            problem.variables.discrete.num_vars,) * max_arity
            self.discrete_constraints_subsets = InPort(shape=subsets_shape)
        self.solution = OutPort(shape=(problem.num_vars,))
        self.variable_assignment = Var(shape=(problem.num_vars,))
        self.optimality = Var(shape=(1,))
        self.feasibility = Var(shape=(1,))


class Monitor(AbstractProcess):
    """Process which reads solution upon Readout process notification."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError


class Readout(AbstractProcess):
    """Monitor Integrator spikes to identify when a solution is found."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError


class Integrator(AbstractProcess):
    """Integrate satisfiability signal from SolverNet, spike if solution found.

    Integration and weights are calibrated to reach spiking threshold when
    optimal and feasible variable assignment is found."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError


class SolverNet(AbstractProcess):
    """Network that represents a problem and whose dynamics solves it."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
