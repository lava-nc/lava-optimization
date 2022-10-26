# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty

import numpy as np
from lava.lib.optimization.problems.coefficients import CoefficientTensorsMixin
from lava.lib.optimization.problems.problems import OptimizationProblem
from lava.lib.optimization.problems.variables import ContinuousVariables, \
    DiscreteVariables
from lava.lib.optimization.solvers.generic.dataclasses import CostMinimizer, \
    MacroStateReader, VariablesImplementation
from lava.lib.optimization.solvers.generic.hierarchical_processes import \
    ContinuousVariablesProcess, CostConvergenceChecker, \
    DiscreteVariablesProcess, SatConvergenceChecker
from lava.lib.optimization.solvers.generic.monitoring_processes \
    .solution_readout.process import SolutionReadout
from lava.magma.core.model.model import AbstractProcessModel
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.process.interfaces import AbstractProcessMember
from lava.magma.core.process.ports.ports import InPort
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import AbstractComputeResource
from lava.magma.core.sync.protocol import AbstractSyncProtocol
from lava.proc.dense.process import Dense
from lava.proc.read_gate.process import ReadGate
from numpy import typing as npt


class SolverProcessBuilder:
    """Builder to dynamically create the process and model for the solver of an
    optimization problem."""

    def __init__(self):
        self._process_constructor = None
        self._model_constructor = None
        self._process = None
        self._model = None

    @property
    def solver_process(self) -> AbstractProcess:
        """Returns the solver process if already created."""
        self.verify_process_exists()
        return self._process

    @property
    def solver_model(self) -> AbstractProcessModel:
        """Returns the solver process model if already created."""
        self.verify_model_exists()
        return self._model

    def verify_process_exists(self):
        """Assert the solver process has already been created."""
        msg = """Process has not been created yet. Make sure the
        create_solver_process method was already called."""
        if self._process is None:
            raise Exception(msg)

    def verify_model_exists(self):
        """Assert the solver process model has already been created."""
        msg = """Process model has not been created yet. Make sure the
        create_solver_model method was already called."""
        if self._model is None:
            raise Exception(msg)

    def create_solver_process(self,
                              problem: OptimizationProblem,
                              hyperparameters: ty.Dict[
                                  str, ty.Union[int, npt.ArrayLike]]):
        """Create and set a solver process for the specified optimization
        problem.

        Parameters
        ----------
        problem: Optimization problem defined by cost and constraints which
        will be used to ensemble the necessary variables and ports with their
        shape and initial values deriving from the problem specification.
        hyperparameters: A dictionary specifying values for steps_to_fire,
        noise_amplitude, step_size and init_value. All but the last are
        integers, the initial value is an array-like of initial values for the
        variables defining the problem.
        """
        self._create_process_constructor(problem, hyperparameters)
        SolverProcess = type("OptimizationSolverProcess",
                             (AbstractProcess,),
                             {"__init__": self._process_constructor}
                             )
        self._process = SolverProcess(hyperparameters)

    def create_solver_model(self,
                            target_cost: int,
                            requirements: ty.List[AbstractComputeResource],
                            protocol: AbstractSyncProtocol):
        """Create and set the model class for the solver process in the
        building pipeline.

        Parameters
        ----------
        target_cost: A cost value provided by the user as a target for the
        solution to be found by the solver, when a solution with such cost is
        found and read, execution ends.
        requirements: specifies which resources the ProcessModel requires.
        protocol: The SyncProtocol that the ProcessModel implements.

        """
        self.verify_process_exists()
        self._create_model_constructor(target_cost)
        SolverModel = type(
            "OptimizationSolverModel",
            (AbstractSubProcessModel,),
            {"__init__": self._model_constructor},
        )
        self._decorate_process_model(SolverModel, requirements, protocol)
        self._model = SolverModel

    def _create_process_constructor(self,
                                    problem: OptimizationProblem,
                                    hyperparameters: ty.Dict[
                                        str, ty.Union[int, npt.ArrayLike]]):
        """Create __init__ method for the OptimizationSolverProcess class.

        Parameters
        ----------

        problem: an instance of an OptimizationProblem defined by cost and
        constraints which will be used to ensemble the necessary variables
        and ports with their shape and initial values deriving from the
        problem specification.
        hyperparameters: A dictionary specifying values for steps_to_fire,
        noise_amplitude, step_size and init_value. All but the last are
        integers, the initial value is an array-like of initial values for the
        variables defining the problem.

        """

        def constructor(self,
                        hyperparameters: ty.Dict[
                            str, ty.Union[int, npt.ArrayLike]],
                        name: ty.Optional[str] = None,
                        log_config: ty.Optional[LogConfig] = None) -> None:
            super(type(self), self).__init__(hyperparameters=hyperparameters,
                                             name=name,
                                             log_config=log_config)
            self.problem = problem
            self.hyperparameters = hyperparameters
            if not hasattr(problem, "variables"):
                raise Exception(
                    "An optimization problem must contain " "variables."
                )
            if hasattr(problem.variables, "continuous") or isinstance(
                    problem.variables, ContinuousVariables
            ):
                self.continuous_variables = Var(
                    shape=(problem.variables.continuous.num_vars, 2)
                )
            if hasattr(problem.variables, "discrete") or isinstance(
                    problem.variables, DiscreteVariables
            ):
                self.discrete_variables = Var(
                    shape=(
                        problem.variables.num_variables,
                        # problem.variables.domain_sizes[0]
                    )
                )
            self.cost_diagonal = None
            if hasattr(problem, "cost"):
                mrcv = SolverProcessBuilder._map_rank_to_coefficients_vars
                self.cost_coefficients = mrcv(problem.cost.coefficients)
                self.cost_diagonal = problem.cost.coefficients[
                    2].diagonal()
            self.variable_assignment = Var(
                shape=(problem.variables.num_variables,)
            )
            self.optimality = Var(shape=(1,))
            self.feasibility = Var(shape=(1,))
            self.solution_step = Var(shape=(1,))

        self._process_constructor = constructor

    def _create_model_constructor(self, target_cost: int):
        """Create __init__ method for the OptimizationSolverModel
        corresponding to the process in the building pipeline.

        Parameters
        ----------
        target_cost: A cost value provided by the user as a target for the
        solution to be found by the solver, when a solution with such cost is
        found and read, execution ends.

        """

        def constructor(self, proc):
            variables = VariablesImplementation()
            if hasattr(proc, "discrete_variables"):
                variables.discrete = DiscreteVariablesProcess(
                    shape=proc.discrete_variables.shape,
                    cost_diagonal=proc.cost_diagonal,
                    hyperparameters=proc.hyperparameters)
            if hasattr(proc, "continuous_variables"):
                variables.continuous = ContinuousVariablesProcess(
                    shape=proc.continuous_variables.shape
                )

            macrostate_reader = MacroStateReader(
                ReadGate(shape=proc.variable_assignment.shape,
                         target_cost=target_cost),
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
                        weights=proc.cost_coefficients[2].init,
                        num_message_bits=24
                    )
                )
                variables.importances = proc.cost_coefficients[1].init
                c = CostConvergenceChecker(shape=proc.variable_assignment.shape)
                macrostate_reader.cost_convergence_check = c
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
            macrostate_reader.read_gate_cost_out.connect(
                macrostate_reader.solution_readout_cost_in
            )
            macrostate_reader.read_gate_req_stop.connect(
                macrostate_reader.solution_readout_req_stop_in
            )
            macrostate_reader.ref_port.connect_var(
                variables.variables_assignment

            )
            macrostate_reader.read_gate_solution_out.connect(
                macrostate_reader.solution_readout_solution_in)
            macrostate_reader.solution_readout.acknowledgemet.connect(
                macrostate_reader.read_gate.acknowledgemet)
            cost_minimizer.gradient_out.connect(variables.gradient_in)
            variables.state_out.connect(cost_minimizer.state_in)
            self.macrostate_reader = macrostate_reader
            self.variables = variables
            self.cost_minimizer = cost_minimizer
            proc.vars.solution_step.alias(macrostate_reader.solution_step)
            for container in [macrostate_reader, cost_minimizer, variables]:
                SolverProcessBuilder.update_parent_process(data_class=container,
                                                           parent=proc)

        self._model_constructor = constructor

    @staticmethod
    def update_parent_process(data_class: ty.Any,
                              parent: AbstractProcess):
        for child in list(data_class.__dict__.values()):
            if child is not None:
                child.parent_proc = parent

    @staticmethod
    def _map_rank_to_coefficients_vars(coefficients: CoefficientTensorsMixin) \
            -> ty.Dict[int, AbstractProcessMember]:
        """Creates a dictionary of Lava variables from a coefficients object.

        Parameters
        ----------

        coefficients: a dictionary-like structure of rank -> tensor pairs. The
        tensors represent the coefficients of a cost or constraints function.

        """
        vars = dict()
        for rank, coefficient in coefficients.items():
            initial_value = SolverProcessBuilder._get_initial_value_for_var(
                coefficient, rank)
            if rank == 2:
                SolverProcessBuilder._update_linear_component_var(vars,
                                                                  coefficient)
            vars[rank] = Var(shape=coefficient.shape, init=initial_value)
        return vars

    @staticmethod
    def _get_initial_value_for_var(coefficient: npt.ArrayLike,
                                   rank: int) -> npt.ArrayLike:
        """Get the value for initializing the coefficient's Var.

        Parameters
        ----------

        coefficient: A tensor representing one of the coefficients of a cost
        or constraints function.
        rank: the rank of the tensor coefficient.

        """
        if rank == 1:
            return coefficient
        if rank == 2:
            quadratic_component = coefficient * np.logical_not(
                np.eye(*coefficient.shape))
            return quadratic_component

    @staticmethod
    def _update_linear_component_var(vars: ty.Dict[int, AbstractProcessMember],
                                     quadratic_coefficient: npt.ArrayLike):
        """Update a linear coefficient's Var given a quadratic coefficient.

        Parameters
        ----------

        vars: A dictionary where keys are ranks and values are the Lava Vars
        corresponding to ranks' coefficients.
        quadratic_coefficient: an array-like tensor of rank 2, corresponds to
        the coefficient of the quadratic term on a cost or constraint function.

        """
        linear_component = quadratic_coefficient.diagonal()
        if 1 in vars.keys():
            vars[1].init = vars[1].init + linear_component
        else:
            vars[1] = Var(shape=linear_component.shape, init=linear_component)

    def _in_ports_from_coefficients(self,
                                    coefficients: CoefficientTensorsMixin) -> \
            ty.Dict[int, AbstractProcessMember]:
        """Create input ports for the various ranks of a coefficients object.

        Parameters
        ----------

        coefficients: A tensor representing one of the coefficients of a cost
        or constraints function.

        """
        in_ports = {coeff.ndim: InPort(shape=coeff.shape) for coeff in
                    coefficients.coefficients}
        return in_ports

    def _decorate_process_model(self,
                                solver_model: "OptimizationSolverModel",
                                requirements: ty.List[AbstractComputeResource],
                                protocol: AbstractSyncProtocol) -> \
            AbstractProcessModel:
        """Replicate the functionality of the decorators for creating Lava
        ProcessModels.

        This is necessary when dynamically creating Lava process model classes.

        Parameters
        ----------
        solver_model: A Lava process created as a solver from the specification
        of an optimization problem.
        requirements: specifies which resources the ProcessModel requires.
        protocol: The SyncProtocol that the ProcessModel implements.

        """
        setattr(solver_model, "implements_process", self.solver_process)
        # Get requirements of parent class
        super_res = solver_model.required_resources.copy()
        # Set new requirements not overwriting parent class requirements.
        setattr(solver_model, "required_resources", super_res + requirements)
        setattr(solver_model, "implements_protocol", protocol)
