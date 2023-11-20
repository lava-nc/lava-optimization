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
from lava.lib.optimization.solvers.generic.sub_process_models import (
    DiscreteVariablesModel
)
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

    def verify_model_exists(self):
        """Assert the solver process model has already been created."""
        msg = """Process model has not been created yet. Make sure the
        create_solver_model method was already called."""
        if self._model is None:
            raise Exception(msg)

    def create_solver_process(
        self,
        problem: OptimizationProblem,
        backend: BACKENDS,
        hyperparameters: ty.Dict[str, ty.Union[int, npt.ArrayLike]],
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
        self._create_process_constructor(backend, problem, hyperparameters)
        SolverProcess = type(
            "OptimizationSolverProcess",
            (AbstractProcess,),
            {"__init__": self._process_constructor},
        )
        self._process = SolverProcess(backend, hyperparameters)

    def create_solver_model(
        self,
        target_cost: int,
        requirements: ty.List[AbstractComputeResource],
        protocol: AbstractSyncProtocol,
    ):
        """Create and set the model class for the solver process in the
        building pipeline.

        Parameters
        ----------
        target_cost: int
            A cost value provided by the user as a target for the solution to be
            found by the solver, when a solution with such cost is found and
            read, execution ends.
        requirements: ty.List[AbstractComputeResource]
            Specifies which resources the ProcessModel requires.
        protocol: AbstractSyncProtocol
            The SyncProtocol that the ProcessModel implements.
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

    def _create_process_constructor(
        self,
        backend: BACKENDS,
        problem: OptimizationProblem,
        hyperparameters: ty.Dict[str, ty.Union[int, npt.ArrayLike]],
    ):
        """Create __init__ method for the OptimizationSolverProcess class.

        Parameters
        ----------
        problem: OptimizationProblem
            An instance of an OptimizationProblem defined by cost and
            constraints which will be used to ensemble the necessary variables
            and ports with their shape and initial values deriving from the
            problem specification.
        hyperparameters: dict
            A dictionary specifying values for temperature and init_value.
            Both are array-like of. init_value defines initial values for the
            variables defining the problem. The temperature provides the
            level of noise.
        """

        def constructor(
            self,
            backend: BACKENDS,
            hyperparameters: ty.Dict[str, ty.Union[int, npt.ArrayLike]],
            name: ty.Optional[str] = None,
            log_config: ty.Optional[LogConfig] = None,
        ) -> None:
            super(type(self), self).__init__(
                backend=backend,
                hyperparameters=hyperparameters,
                name=name,
                log_config=log_config,
            )
            self.problem = problem
            self.hyperparameters = hyperparameters
            self.backend = backend
            self.is_continuous = 0
            self.is_discrete = 0
            if not hasattr(problem, "variables"):
                raise Exception(
                    "An optimization problem must contain " "variables."
                )
            if (
                hasattr(problem.variables, "continuous")
                and problem.variables.continuous.num_variables is not None
            ):
                self.continuous_variables = Var(
                    shape=(problem.variables.continuous.num_variables,)
                )
                self.is_continuous = 1
            if (
                hasattr(problem.variables, "discrete")
                and problem.variables.discrete.num_variables is not None
            ):
                self.discrete_variables = Var(
                    shape=(
                        problem.variables.discrete.num_variables,
                    )
                )
                self.is_discrete = 1
            self.cost_diagonal = None
            if hasattr(problem, "cost"):
                mrcv = SolverProcessBuilder._map_rank_to_coefficients_vars
                self.cost_coefficients = mrcv(problem.cost.coefficients)
                if self.is_discrete:
                    self.cost_diagonal = problem.cost.coefficients[
                        2
                    ].diagonal()
            if not self.is_continuous:
                self.variable_assignment = Var(
                    shape=(problem.variables.discrete.num_variables,)
                )
                self.best_variable_assignment = Var(
                    shape=(problem.variables.discrete.num_variables,)
                )
                # Total cost=optimality_first_byte<<24+optimality_last_bytes
                self.optimality_last_bytes = Var(shape=(1,))
                self.optimality_first_byte = Var(shape=(1,))
                self.optimum = Var(shape=(2,))
                self.feasibility = Var(shape=(1,))
                self.solution_step = Var(shape=(1,))
                self.cost_monitor = Var(shape=(1,))
            elif self.is_continuous:
                self.variable_assignment = Var(
                    shape=(problem.variables.continuous.num_variables,)
                )
            self.finders = None

        self._process_constructor = constructor

    def _create_model_constructor(self, target_cost: int):
        """Create __init__ method for the OptimizationSolverModel
        corresponding to the process in the building pipeline.

        Parameters
        ----------
        target_cost: int
            A cost value provided by the user as a target for the solution to be
            found by the solver, when a solution with such cost is found and
            read, execution ends.
        """

        def constructor(self, proc):
            discrete_var_shape = None
            if hasattr(proc, "discrete_variables"):
                discrete_var_shape = proc.discrete_variables.shape
            continuous_var_shape = None
            if hasattr(proc, "continuous_variables"):
                continuous_var_shape = proc.continuous_variables.shape
            cost_diagonal = proc.cost_diagonal
            cost_coefficients = proc.cost_coefficients
            constraints = proc.problem.constraints
            hyperparameters = proc.hyperparameters
            problem = proc.problem
            backend = proc.backend

            hps = (
                hyperparameters
                if isinstance(hyperparameters, list)
                else [hyperparameters]
            )
            #
            if not proc.is_continuous:
                self.solution_reader = SolutionReader(
                    var_shape=discrete_var_shape,
                    target_cost=target_cost,
                    num_in_ports=len(hps),
                    time_steps_per_algorithmic_step=DiscreteVariablesModel.
                    get_neuron_process(
                        proc.hyperparameters).
                    time_steps_per_algorithmic_step
                )
            finders = []
            for idx, hp in enumerate(hps):
                finder = SolutionFinder(
                    cost_diagonal=cost_diagonal,
                    cost_coefficients=cost_coefficients,
                    constraints=constraints,
                    backend=backend,
                    hyperparameters=hp,
                    discrete_var_shape=discrete_var_shape,
                    continuous_var_shape=continuous_var_shape,
                    problem=problem,
                )
                setattr(self, f"finder_{idx}", finder)
                finders.append(finder)
                if not proc.is_continuous:
                    finder.cost_out_last_bytes.connect(
                        getattr(
                            self.solution_reader,
                            f"read_gate_in_port_last_bytes_{idx}",
                        )
                    )
                    finder.cost_out_first_byte.connect(
                        getattr(
                            self.solution_reader,
                            f"read_gate_in_port_first_byte_{idx}",
                        )
                    )
            proc.finders = finders
            # Variable aliasing
            if not proc.is_continuous:
                if hasattr(proc, "cost_coefficients"):
                    proc.vars.optimum.alias(self.solution_reader.min_cost)
                    # Cost = optimality_first_byte << 24 + optimality_last_bytes
                    proc.vars.optimality_last_bytes.alias(
                        proc.finders[0].cost_last_bytes
                    )
                    proc.vars.optimality_first_byte.alias(
                        proc.finders[0].cost_first_byte
                    )
                proc.vars.variable_assignment.alias(
                    proc.finders[0].variables_assignment
                )
                proc.vars.best_variable_assignment.alias(
                    self.solution_reader.solution
                )
                proc.vars.solution_step.alias(
                    self.solution_reader.solution_step
                )

                # Connect processes
                self.solution_reader.ref_port.connect_var(
                    finders[0].variables_assignment
                )

            proc.vars.variable_assignment.alias(
                proc.finders[0].variables_assignment
            )

        self._model_constructor = constructor

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
