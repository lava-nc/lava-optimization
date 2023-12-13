# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
from lava.magma.core.resources import (
    CPU,
    AbstractComputeResource,
    Loihi2NeuroCore,
    NeuroCore,
)
from lava.lib.optimization.problems.coefficients import CoefficientTensorsMixin
from lava.lib.optimization.problems.problems import OptimizationProblem
from lava.lib.optimization.solvers.generic.solution_finder.process import (
    SolutionFinder,
)
from lava.lib.optimization.solvers.generic.solution_reader.process import (
    SolutionReader,
)
from lava.magma.core.model.model import AbstractProcessModel
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.process.interfaces import AbstractProcessMember
from lava.magma.core.process.ports.ports import InPort
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import AbstractComputeResource
from lava.magma.core.sync.protocol import AbstractSyncProtocol
from numpy import typing as npt

BACKENDS = ty.Union[CPU, Loihi2NeuroCore, NeuroCore, str]


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

    def create_solver_process(
            self,
            problem: OptimizationProblem,
            config: "SolverConfig"
    ):
        """Create and set a solver process for the specified optimization
        problem.

        Parameters
        ----------
        problem: OptimizationProblem
            Optimization problem defined by cost and constraints which will be
            used to ensemble the necessary variables and ports with their shape
            and initial values deriving from the problem specification.
        hyperparameters: dict
            A dictionary specifying values for temperature and init_value.
            Both are array-like of. init_value defines initial values for the
            variables defining the problem. The temperature provides the
            level of noise.
        """
        self._create_process_constructor(problem=problem)
        SolverProcess = type(
            "OptimizationSolverProcess",
            (AbstractProcess,),
            {"__init__": self._process_constructor},
        )
        self._process = SolverProcess(config=config)

    def verify_model_exists(self):
        """Assert the solver process model has already been created."""
        msg = """Process model has not been created yet. Make sure the
        create_solver_model method was already called."""
        if self._model is None:
            raise Exception(msg)

    def create_solver_model(
            self,
            requirements: ty.List[AbstractComputeResource],
            protocol: AbstractSyncProtocol,
    ):
        """Create and set the model class for the solver process in the
        building pipeline.

        Parameters
        ----------
        requirements: ty.List[AbstractComputeResource]
            Specifies which resources the ProcessModel requires.
        protocol: AbstractSyncProtocol
            The SyncProtocol that the ProcessModel implements.
        """
        self.verify_process_exists()
        self._create_model_constructor()
        SolverModel = type(
            "OptimizationSolverModel",
            (AbstractSubProcessModel,),
            {"__init__": self._model_constructor},
        )
        self._decorate_process_model(SolverModel, requirements, protocol)
        self._model = SolverModel

    def _create_process_constructor(
            self,
            problem: OptimizationProblem,
    ):
        """Create __init__ method for the OptimizationSolverProcess class.

        Parameters
        ----------
        problem: OptimizationProblem
            An instance of an OptimizationProblem defined by cost and
            constraints which will be used to ensemble the necessary variables
            and ports with their shape and initial values deriving from the
            problem specification.
        """

        def constructor(
                selfi,
                config: "SolverConfig",
                name: ty.Optional[str] = None
        ) -> None:
            super(type(selfi), selfi).__init__(config=config, name=name)
            selfi.config = config
            selfi.problem = problem
            self._create_vars(selfi, problem)
            selfi.finders = None

        self._process_constructor = constructor

    def _create_vars(self, selfi, problem):
        self._check_problem_has_variables(problem)
        self._create_variables_vars(selfi, problem)
        self._create_cost_vars(selfi, problem)

    def _check_problem_has_variables(self, problem: OptimizationProblem):
        if not hasattr(problem, "variables"):
            raise Exception(
                "An optimization problem must contain " "variables."
            )

    def _create_variables_vars(self, selfi, prob):
        selfi.is_discrete = False
        selfi.is_continuous = False
        if self._has_continuous_vars(prob):
            self._create_continuous_vars_vars(selfi, prob)
            selfi.is_continuous = True
        if self._has_discrete_vars(prob):
            self._create_discrete_vars_vars(selfi, prob)
            selfi.is_discrete = True

    def _create_cost_vars(self, selfi, problem):
        selfi.cost_diagonal = None
        if hasattr(problem, "cost"):
            mrcv = SolverProcessBuilder._map_rank_to_coefficients_vars
            selfi.cost_coefficients = mrcv(problem.cost.coefficients)
            if selfi.is_discrete:
                selfi.cost_diagonal = problem.cost.coefficients[
                    2
                ].diagonal()

    def _has_continuous_vars(self, problem):
        return self._is_continuous(problem) and self._num_cont_vars(
            problem) > 0

    def _create_continuous_vars_vars(self, selfi, prob):
        selfi.continuous_variables = Var(shape=(self._num_cont_vars(prob),))
        if selfi.is_continuous:
            selfi.variable_assignment = Var(
                shape=(prob.variables.continuous.num_variables,)
            )

    def _has_discrete_vars(self, problem):
        return self._is_discrete(problem) and self._num_disc_vars(
            problem) > 0

    def _create_discrete_vars_vars(self, selfi, prob):
        selfi.discrete_variables = Var(shape=(self._num_disc_vars(prob),))
        if not selfi.is_continuous:
            selfi.best_variable_assignment = Var(
                shape=(prob.variables.discrete.num_variables,))
            # Total cost=optimality_first_byte<<24+optimality_last_bytes
            for idx in range(selfi.config.num_replicas):
                self._create_indexed_variables(selfi, idx, prob)
            selfi.optimum = Var(shape=(2,))
            selfi.feasibility = Var(shape=(1,))
            selfi.solution_step = Var(shape=(1,))
            selfi.cost_monitor = Var(shape=(1,))

    def _create_indexed_variables(self, selfi, idx, prob):
        setattr(selfi, f"variable_assignment_{idx}",
                Var(shape=(prob.variables.discrete.num_variables,)))
        setattr(selfi, f"optimality_last_bytes_{idx}", Var(shape=(1,)))
        setattr(selfi, f"optimality_first_byte_{idx}", Var(shape=(1,)))
        setattr(selfi, f"internal_state_{idx}",
                Var(shape=(prob.variables.discrete.num_variables,)))

    def _num_cont_vars(self, problem):
        return problem.variables.continuous.num_variables

    def _num_disc_vars(self, problem):
        return problem.variables.discrete.num_variables

    def _is_continuous(self, problem):
        return hasattr(problem.variables, "continuous")

    def _is_discrete(self, problem):
        return hasattr(problem.variables, "discrete")

    def _create_model_constructor(self):
        """Create __init__ method for the OptimizationSolverModel
        corresponding to the process in the building pipeline.
        """
        def constructor(selfi, proc):
            hps = self._get_hps(proc.config)
            proc.finders = self._create_finders(hps, selfi, proc)
            if not proc.is_continuous:
                reader = self._create_solution_reader(selfi, proc, len(hps))
                self._connect_finder_ref_ports(proc.finders, reader)
                self._connect_finder_to_reader(proc.finders, reader)
        self._model_constructor = constructor

    def _get_hps(self, config):
        return (config.hyperparameters if isinstance(
            config.hyperparameters, list) else [config.hyperparameters])

    def _create_solution_reader(self, selfi, proc, num_in_ports):
        selfi.solution_reader = SolutionReader(
            var_shape=proc.discrete_variables.shape,
            target_cost=proc.config.target_cost,
            num_in_ports=num_in_ports,
            num_steps=proc.config.num_steps
        )
        self._create_reader_aliases(proc, selfi)
        return selfi.solution_reader

    def _create_finders(self, hps, selfi, proc):
        finder_params = self._get_fixed_finder_params(proc)
        finders = []
        for idx, hp in enumerate(hps):
            finder = self._create_finder(selfi, finder_params, idx, hp)
            setattr(selfi, f"finder_{idx}", finder)
            finders.append(finder)

        if not proc.is_continuous:
            self._create_finders_aliases(finders, proc, proc.config, selfi)
        return finders

    def _connect_finder_ref_ports(self, finders, reader):
        reader.ref_port.connect_var(finders[0].variables_assignment)

    def _get_fixed_finder_params(self, proc):
        config = proc.config
        discrete_var_shape = None
        if hasattr(proc, "discrete_variables"):
            discrete_var_shape = proc.discrete_variables.shape
        continuous_var_shape = None
        if hasattr(proc, "continuous_variables"):
            continuous_var_shape = proc.continuous_variables.shape
        params = dict(cost_diagonal=proc.cost_diagonal,
                      cost_coefficients=proc.cost_coefficients,
                      constraints=proc.problem.constraints,
                      backend=config.backend,
                      discrete_var_shape=discrete_var_shape,
                      continuous_var_shape=continuous_var_shape,
                      problem=proc.problem,
                      idx=0)
        return params

    def _connect_finder_to_reader(self, finders, reader):
        for idx, finder in enumerate(finders):
            finder_port1 = getattr(finder, f"cost_out_first_byte")
            finder_port2 = getattr(finder, f"cost_out_last_bytes")
            reader_port1 = getattr(reader, f"read_gate_in_port_first_byte_{idx}", )
            reader_port2 = getattr(reader, f"read_gate_in_port_last_bytes_{idx}", )
            finder_port1.connect(reader_port1)
            finder_port2.connect(reader_port2)

    def _create_finder(self, selfi, finder_params, idx, hp):
        finder_params['idx'] = idx
        finder_params['hyperparameters'] = hp
        finder = SolutionFinder(**finder_params)
        return finder

    def _create_finders_aliases(self, finders, proc, config, selfi):
        if hasattr(proc, "cost_coefficients"):
            for idx, finder in enumerate(finders):
                self._create_finder_aliases(idx, finder, proc, selfi)


    def _create_finder_aliases(self, idx, finder, proc, selfi):
        getattr(proc.vars, f"optimality_last_bytes_{idx}").alias(finder.cost_last_bytes)
        getattr(proc.vars, f"optimality_first_byte_{idx}").alias(finder.cost_first_byte)
        getattr(proc.vars, f"variable_assignment_{idx}").alias(finder.variables_assignment)
        getattr(proc.vars, f"internal_state_{idx}").alias(finder.internal_state)

    def _create_reader_aliases(self, proc, selfi):
        proc.vars.optimum.alias(selfi.solution_reader.min_cost)
        proc.vars.best_variable_assignment.alias(selfi.solution_reader.solution)
        proc.vars.solution_step.alias(selfi.solution_reader.solution_step)


    @staticmethod
    def _map_rank_to_coefficients_vars(
            coefficients: CoefficientTensorsMixin,
    ) -> ty.Dict[int, AbstractProcessMember]:
        """Creates a dictionary of Lava variables from a coefficients object.

        Parameters
        ----------
        coefficients: CoefficientTensorsMixin
            A dictionary-like structure of rank -> tensor pairs. The tensors
            represent the coefficients of a cost or constraints function.
        """
        vars = dict()
        for rank, coefficient in coefficients.items():
            vars[rank] = Var(shape=coefficient.shape, init=coefficient)
        return vars

    def _in_ports_from_coefficients(
            self, coefficients: CoefficientTensorsMixin
    ) -> ty.Dict[int, AbstractProcessMember]:
        """Create input ports for the various ranks of a coefficients object.

        Parameters
        ----------
        coefficients: CoefficientTensorsMixin
            A tensor representing one of the coefficients of a cost or
            constraints function.
        """
        in_ports = {
            coeff.ndim: InPort(shape=coeff.shape)
            for coeff in coefficients.coefficients
        }
        return in_ports

    def _decorate_process_model(
            self,
            solver_model: "OptimizationSolverModel",
            requirements: ty.List[AbstractComputeResource],
            protocol: AbstractSyncProtocol,
    ) -> AbstractProcessModel:
        """Replicate the functionality of the decorators for creating Lava
        ProcessModels.

        This is necessary when dynamically creating Lava process model classes.

        Parameters
        ----------
        solver_model: "OptimizationSolverModel"
            A Lava process created as a solver from the specification of an
            optimization problem.
        requirements: ty.List[AbstractComputeResource]
            Specifies which resources the ProcessModel requires.
        protocol: AbstractSyncProtocol
            The SyncProtocol that the ProcessModel implements.
        """
        setattr(solver_model, "implements_process", self.solver_process)
        # Get requirements of parent class
        super_res = solver_model.required_resources.copy()
        # Set new requirements not overwriting parent class requirements.
        setattr(solver_model, "required_resources", super_res + requirements)
        setattr(solver_model, "implements_protocol", protocol)
