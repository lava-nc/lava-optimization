# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
import numpy as np

from dataclasses import dataclass
import warnings

from lava.lib.optimization.solvers.generic.qp.models import (
    PyPIneurPIPGeqModel,
    PyProjGradPIPGeqModel,
)
from lava.lib.optimization.solvers.generic.qp.processes import (
    ProjectedGradientNeuronsPIPGeq,
    ProportionalIntegralNeuronsPIPGeq,
)

from lava.magma.core.resources import (
    CPU,
    AbstractComputeResource,
    Loihi2NeuroCore,
    NeuroCore,
)
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg, Loihi2HwCfg
from lava.magma.core.sync.protocol import AbstractSyncProtocol
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.dense.models import PyDenseModelFloat
from lava.proc.sparse.models import PySparseModelFloat
from lava.proc.dense.process import Dense
from lava.proc.sparse.process import Sparse
from lava.proc.monitor.process import Monitor
from lava.utils.profiler import Profiler

from lava.lib.optimization.problems.problems import QUBO
from lava.lib.optimization.problems.problems import OptimizationProblem
from lava.lib.optimization.solvers.generic.builder import SolverProcessBuilder
from lava.lib.optimization.solvers.generic.cost_integrator.process import (
    CostIntegrator,
)
from lava.lib.optimization.solvers.generic.hierarchical_processes import (
    NEBMAbstract,
    SimulatedAnnealingAbstract,
    SimulatedAnnealingLocalAbstract,
)
from lava.lib.optimization.solvers.generic.monitoring_processes. \
    solution_readout.models import (
        SolutionReadoutPyModel,
    )
from lava.lib.optimization.solvers.generic.nebm.models import NEBMPyModel
from lava.lib.optimization.solvers.generic.nebm.process import (
    NEBM,
    SimulatedAnnealing,
    SimulatedAnnealingLocal,
)
from lava.lib.optimization.solvers.generic.annealing.process import Annealing
from lava.lib.optimization.solvers.generic.scif.models import (
    PyModelQuboScifFixed,
)
from lava.lib.optimization.solvers.generic.scif.process import QuboScif
from lava.lib.optimization.solvers.generic.sub_process_models import (
    NEBMAbstractModel,
    SimulatedAnnealingAbstractModel,
    SimulatedAnnealingLocalAbstractModel,
)

try:
    from lava.proc.dense.ncmodels import NcModelDense
    from lava.proc.sparse.ncmodel import NcModelSparse

    from lava.lib.optimization.solvers.generic.cost_integrator.ncmodels import (
        CostIntegratorNcModel,
    )
    from lava.lib.optimization.solvers.generic.nebm.ncmodels import (
        NEBMNcModel,
        SimulatedAnnealingNcModel,
        SimulatedAnnealingLocalNcModel,
    )
    from lava.lib.optimization.solvers.generic.annealing.ncmodels import (
        AnnealingNcModel,
    )
    from lava.lib.optimization.solvers.generic.read_gate.ncmodels import (
        ReadGateCModel,
    )

    from lava.lib.optimization.solvers.generic.qp.ncmodels import (
        NcL2ModelPG,
        NcL2ModelPI,
    )
    from lava.lib.optimization.solvers.generic.sub_process_models import (
        DiscreteVariablesModel
    )
except ImportError:

    class ReadGateCModel:
        pass

    class NcModelDense:
        pass

    class NEBMNcModel:
        pass

    class SimulatedAnnealingNcModel:
        pass

    class SimulatedAnnealingLocalNcModel:
        pass

    class AnnealingNcModel:
        pass

    class CostIntegratorNcModel:
        pass

    class NcModelSparse:
        pass

    class NcL2ModelPG:
        pass

    class NcL2ModelPI:
        pass


from lava.lib.optimization.solvers.generic.read_gate.models import (
    ReadGatePyModel,
)

BACKENDS = ty.Union[CPU, Loihi2NeuroCore, NeuroCore, str]
HP_TYPE = ty.Union[ty.Dict, ty.List[ty.Dict]]
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


@dataclass
class SolverConfig:
    """
    Dataclass to store and validate OptimizationSolver configurations.

    Parameters
    ----------
    timeout: int
        Maximum number of iterations (timesteps) to be run. If set to -1
        then the solver will run continuously in non-blocking mode until a
        solution is found.
    target_cost: int, optional
        A cost value provided by the user as a target for the solution to be
        found by the solver, when a solution with such cost is found and
        read, execution ends.
    backend: BACKENDS, optional
        Specifies the backend where the main solver network will be
        deployed.
    hyperparameters:
        ty.Union[ty.Dict, ty.Dict[str, ty.Union[int, npt.ArrayLike]]],
        optional.
        A dictionary specifying values for steps_to_fire, noise_amplitude,
        step_size and init_value. All but the last are integers, the initial
        value is an array-like of initial values for the variables defining
        the problem.
    probe_cost: bool
        A boolean flag to request cost tracking through time.
    probe_time: bool
        A boolean flag to request time profiling, available only on "Loihi2"
        backend.
    probe_energy: bool
        A boolean flag to request time profiling, available only on "Loihi2"
        backend.
    log_level: int
        Select log verbosity (40: default, 20: verbose).
    folded_compilation: bool
        A boolean flag to enable folded compilation, available only on
        "Loihi2" backend. Default value is False.
    """

    timeout: int = 1e3
    target_cost: int = 0
    backend: BACKENDS = CPU
    hyperparameters: HP_TYPE = None
    probe_cost: bool = False
    probe_state: bool = False
    probe_time: bool = False
    probe_energy: bool = False
    log_level: int = 40
    folded_compilation: bool = False


@dataclass(frozen=True)
class SolverReport:
    """
    Dataclass to store OptimizationSolver results.

    Parameters
    ----------
    best_cost: int
        Best cost found during the execution.
    best_state: np.ndarray
        Candidate solution associated to the best cost.
    best_timestep: int
        Execution timestep during which the best solution was found.
    solver_config: SolverConfig
        Solver configuraiton used. Refers to SolverConfig documentation.
    profiler: Profiler
        Profiler instance containing time, energy and activity measurements.
    """

    problem: OptimizationProblem = None
    best_cost: int = None
    best_state: np.ndarray = None
    best_timestep: int = None
    cost_timeseries: np.ndarray = None
    state_timeseries: np.ndarray = None
    solver_config: SolverConfig = None
    profiler: Profiler = None


def solve(
    problem: OptimizationProblem, config: SolverConfig = SolverConfig()
) -> np.ndarray:
    """
    Solve the given optimization problem using the passed configuration, and
    returns the best candidate solution.

    Parameters
    ----------
    problem: OptimizationProblem
        Optimization problem to be solved.
    config: SolverConfig, optional
        Solver configuraiton used. Refers to SolverConfig documentation.
    """
    solver = OptimizationSolver(problem)
    report = solver.solve(config=config)
    return report.best_state


class OptimizationSolver:
    """
    Generic solver for constrained optimization problems defined by
    variables, cost and constraints.

    The problem should behave according to the OptimizationProblem's
    interface so that the Lava solver can be built correctly.

    A Lava OptimizationSolverProcess and a Lava OptimizationSolverModel will
    be created from the problem specification. The dynamics of such process
    implements the algorithms that search a solution to the problem and
    reports it to the user.
    """

    def __init__(self, problem: OptimizationProblem):
        """
        Constructor for the OptimizationSolver class.

        Parameters
        ----------
        problem: OptimizationProblem
            Optimization problem to be solved.
        """
        self.problem = problem
        self._process_builder = SolverProcessBuilder()
        self.solver_process = None
        self.solver_model = None
        self._profiler = None
        self._cost_tracker_first_byte = None
        self._cost_tracker_last_bytes = None
        self._state_tracker = None

    def solve(self, config: SolverConfig = SolverConfig()) -> SolverReport:
        """
        Create solver from problem spec and run until it has
        either minimized the cost to the target_cost or ran for a number of
        time steps provided by the timeout parameter.

        Parameters
        ----------
        config: SolverConfig, optional
            Solver configuration used. Refers to SolverConfig documentation.

        Returns
        ----------
        report: SolverReport
            An object containing all the data generated by the execution.
        """
        self._validate_input_solve(config)
        run_condition, run_cfg = self._prepare_solver(config)
        folded_compile_config = None
        if config.folded_compilation:
            folded_compile_config = {'folded_view' : ['SolutionFinder']}
        self.solver_process.run(condition=run_condition, run_cfg=run_cfg,
                                compile_config=folded_compile_config)
        best_state, best_cost, best_timestep = self._get_results(config)
        cost_timeseries, state_timeseries = self._get_probing(config)
        self.solver_process.stop()
        return SolverReport(
            problem=self.problem,
            best_cost=best_cost,
            best_state=best_state,
            best_timestep=best_timestep,
            solver_config=config,
            profiler=self._profiler,
            cost_timeseries=cost_timeseries,
            state_timeseries=state_timeseries,
        )

    def _prepare_solver(self, config: SolverConfig):
        self._create_solver_process(config=config)
        hps = config.hyperparameters
        num_in_ports = len(hps) if isinstance(hps, list) else 1
        probes = []
        if config.backend in NEUROCORES:
            from lava.utils.loihi2_state_probes import StateProbe

            if config.probe_cost:
                self._cost_tracker_last_bytes = StateProbe(
                    self.solver_process.optimality_last_bytes
                )
                self._cost_tracker_first_byte = StateProbe(
                    self.solver_process.optimality_first_byte
                )
                probes.append(self._cost_tracker_last_bytes)
                probes.append(self._cost_tracker_first_byte)
            if config.probe_state:
                self._state_tracker = StateProbe(
                    self.solver_process.variable_assignment
                )
                probes.append(self._state_tracker)
        elif config.backend in CPUS:
            if config.probe_cost:
                self._cost_tracker_first_byte = Monitor()
                self._cost_tracker_first_byte.probe(
                    target=self.solver_process.optimality_first_byte,
                    num_steps=config.timeout,
                )
                self._cost_tracker_last_bytes = Monitor()
                self._cost_tracker_last_bytes.probe(
                    target=self.solver_process.optimality_last_bytes,
                    num_steps=config.timeout,
                )
                probes.append(self._cost_tracker_first_byte)
                probes.append(self._cost_tracker_last_bytes)
            if config.probe_state:
                self._state_tracker = Monitor()
                self._state_tracker.probe(
                    target=self.solver_process.variable_assignment,
                    num_steps=config.timeout,
                )
                probes.append(self._state_tracker)
        run_cfg = self._get_run_config(
            backend=config.backend,
            probes=probes,
            num_in_ports=num_in_ports,
        )
        run_condition = RunSteps(num_steps=config.timeout)
        self._prepare_profiler(config=config, run_cfg=run_cfg)
        return run_condition, run_cfg

    def _create_solver_process(self, config: SolverConfig) -> None:
        """
        Create process and model class as solver for the given problem.

        Parameters
        ----------
        config: SolverConfig
            Solver configuration used. Refers to SolverConfig documentation.
        """
        requirements, protocol = self._get_requirements_and_protocol(
            backend=config.backend
        )
        self._process_builder.create_solver_process(
            problem=self.problem,
            backend=config.backend,
            hyperparameters=config.hyperparameters or dict(),
        )
        self._process_builder.create_solver_model(
            target_cost=config.target_cost,
            requirements=requirements,
            protocol=protocol,
        )
        self.solver_process = self._process_builder.solver_process
        self.solver_model = self._process_builder.solver_model
        self.solver_process._log_config.level = config.log_level

    def _get_requirements_and_protocol(
        self, backend: BACKENDS
    ) -> ty.Tuple[AbstractComputeResource, AbstractSyncProtocol]:
        """
        Figure out requirements and protocol for a given backend.

        Parameters
        ----------
        backend: BACKENDS
            Specifies the backend for which requirements and protocol classes
            will be returned.
        """
        return [CPU] if backend in CPUS else [Loihi2NeuroCore], LoihiProtocol

    def _get_probing(
        self, config: SolverConfig()
    ) -> ty.Tuple[np.ndarray, np.ndarray]:
        """
        Return the cost and state timeseries if probed.

        Parameters
        ----------
        config: SolverConfig
            Solver configuration used. Refers to SolverConfig documentation.
        """
        cost_timeseries_last_bytes = self._get_probed_data(
            tracker=self._cost_tracker_last_bytes,
            var_name="optimality_last_bytes",
        )
        cost_timeseries_first_byte = self._get_probed_data(
            tracker=self._cost_tracker_first_byte,
            var_name="optimality_first_byte",
        )
        if self._cost_tracker_first_byte is not None:
            cost_timeseries = (
                cost_timeseries_first_byte << 24
            ) + cost_timeseries_last_bytes
        else:
            cost_timeseries = None
        state_timeseries = self._get_probed_data(
            tracker=self._state_tracker, var_name="variable_assignment"
        )
        if state_timeseries is not None:
            time_steps_per_algorithmic_step = 1
            # Some solvers for discrete optimization require more than 1 time
            # step per algorithmic step
            if self._is_problem_discrete():
                time_steps_per_algorithmic_step = \
                    DiscreteVariablesModel.get_neuron_process(
                        config.hyperparameters).\
                    time_steps_per_algorithmic_step
            state_timeseries = SolutionReadoutPyModel.decode_solution(
                raw_solution=state_timeseries,
                time_steps_per_algorithmic_step=time_steps_per_algorithmic_step
            )
        return cost_timeseries, state_timeseries

    def _get_probed_data(self, tracker, var_name):
        if tracker is None:
            return None
        if isinstance(tracker, Monitor):
            return tracker.get_data()[self.solver_process.name][
                getattr(self.solver_process, var_name).name
            ].astype(np.int32)
        else:
            return tracker.time_series

    def _get_run_config(
        self, backend: BACKENDS, probes=None, num_in_ports: int = None
    ):
        from lava.lib.optimization.solvers.generic.read_gate.process import (
            ReadGate
        )
        from lava.lib.optimization.solvers.generic.read_gate.models import (
            get_read_gate_model_class,
        )

        if backend in CPUS:
            ReadGatePyModel = get_read_gate_model_class(num_in_ports)
            pdict = {
                self.solver_process: self.solver_model,
                ReadGate: ReadGatePyModel,
                Dense: PyDenseModelFloat,
                Sparse: PySparseModelFloat,
                NEBMAbstract: NEBMAbstractModel,
                NEBM: NEBMPyModel,
                QuboScif: PyModelQuboScifFixed,
                ProportionalIntegralNeuronsPIPGeq: PyPIneurPIPGeqModel,
                ProjectedGradientNeuronsPIPGeq: PyProjGradPIPGeqModel,
            }
            return Loihi1SimCfg(exception_proc_model_map=pdict,
                                select_sub_proc_model=True)
        elif backend in NEUROCORES:
            pdict = {
                self.solver_process: self.solver_model,
                ReadGate: ReadGateCModel,
                Dense: NcModelDense,
                Sparse: NcModelSparse,
                NEBMAbstract: NEBMAbstractModel,
                NEBM: NEBMNcModel,
                SimulatedAnnealingAbstract:
                SimulatedAnnealingAbstractModel,
                SimulatedAnnealingLocalAbstract:
                    SimulatedAnnealingLocalAbstractModel,
                SimulatedAnnealingLocal: SimulatedAnnealingLocalNcModel,
                Annealing: AnnealingNcModel,
                CostIntegrator: CostIntegratorNcModel,
                ProportionalIntegralNeuronsPIPGeq: NcL2ModelPI,
                ProjectedGradientNeuronsPIPGeq: NcL2ModelPG,
            }
            return Loihi2HwCfg(
                exception_proc_model_map=pdict,
                select_sub_proc_model=True,
                callback_fxs=probes,
            )
        else:
            raise NotImplementedError(str(backend) + BACKEND_MSG)

    def _prepare_profiler(self, config: SolverConfig, run_cfg) -> None:
        if config.probe_time or config.probe_energy:
            self._profiler = Profiler.init(run_cfg)
            if config.probe_time:
                self._profiler.execution_time_probe(num_steps=config.timeout)
            if config.probe_energy:
                self._profiler.energy_probe(num_steps=config.timeout)
        else:
            self._profiler = None

    def _get_results(self, config: SolverConfig):
        idx = 0
        if self.solver_process.is_discrete:
            best_cost, idx = self.solver_process.optimum.get()
            best_cost = SolutionReadoutPyModel.decode_cost(best_cost)
            best_timestep = (
                self.solver_process.solution_step.aliased_var.get() - 2
            )
        best_state = self._get_best_state(config, int(idx))

        if self.solver_process.is_discrete:
            return best_state, int(best_cost), int(best_timestep)
        else:
            return best_state, None, None

    def _get_best_state(self, config: SolverConfig, idx: int):
        if self._is_problem_discrete():
            discrete_values = self._get_and_decode_discrete_vars(config, idx)
            return discrete_values

        if self._is_problem_continuous():
            continuous_values = self._get_and_decode_continuous_vars(idx)
            return continuous_values

    def _is_problem_discrete(self):
        return (
            hasattr(self.problem.variables, "discrete")
            and self.problem.variables.discrete.num_variables is not None
        )

    def _get_and_decode_discrete_vars(self, config: SolverConfig, idx: int):
        if isinstance(config.hyperparameters, list):
            raw_solution = np.asarray(
                self.solver_process.finders[idx]
                .variables_assignment.get()
                .astype(np.int32)
            )
            time_steps_per_algorithmic_step = \
                DiscreteVariablesModel.get_neuron_process(
                    config.hyperparameters).\
                time_steps_per_algorithmic_step
            raw_solution = SolutionReadoutPyModel.decode_solution(
                raw_solution=raw_solution,
                time_steps_per_algorithmic_step=time_steps_per_algorithmic_step
            )
            return raw_solution
        else:
            best_assignment = self.solver_process.best_variable_assignment
            return best_assignment.aliased_var.get()

    def _is_problem_continuous(self):
        return (
            hasattr(self.problem.variables, "continuous")
            and self.problem.variables.continuous.num_variables is not None
        )

    def _get_and_decode_continuous_vars(self, idx: int):
        solution = np.asarray(
            self.solver_process.finders[idx].variables_assignment.get()
        )
        return solution

    def _validate_input_solve(self, config: SolverConfig) -> None:
        """
        Validates the user input provided to the solve() method of the
        OptimizationSolver.

        Parameters
        ----------
        config: SolverConfig
            Solver configuration used.

        Raises
        ----------
        """

        if (config.backend in CPUS) and isinstance(self.problem, QUBO):
            warnings.warn("Please note that we are currently advancing only "
                          "the Loihi 2 backend of the QUBO solver. Until "
                          "further notice, the CPU implementation uses "
                          "an outdated algorithm. We thus recommend running "
                          "your workloads on Loihi 2 to leverage the latest "
                          "algorithms for optimal latency, energy "
                          "consumption, and solution accuracy. ")
