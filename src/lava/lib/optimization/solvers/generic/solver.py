# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import numpy as np
import typing as ty

from dataclasses import dataclass
from lava.lib.optimization.problems.problems import OptimizationProblem
from lava.lib.optimization.solvers.generic.builder import SolverProcessBuilder
from lava.lib.optimization.solvers.generic.hierarchical_processes import (
    BoltzmannAbstract,
)
from lava.lib.optimization.solvers.generic.scif.models import (
    BoltzmannFixed,
    PyModelQuboScifFixed,
)
from lava.lib.optimization.solvers.generic.scif.process import (
    Boltzmann,
    QuboScif,
)
from lava.lib.optimization.solvers.generic.sub_process_models import (
    BoltzmannAbstractModel,
)
from lava.magma.core.resources import (
    AbstractComputeResource,
    CPU,
    Loihi2NeuroCore,
    NeuroCore,
)
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg, Loihi2HwCfg
from lava.magma.core.sync.protocol import AbstractSyncProtocol
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.dense.models import PyDenseModelFloat
from lava.proc.dense.process import Dense
from lava.proc.monitor.process import Monitor
from lava.utils.profiler import Profiler

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
    """

    timeout: int = 1e3
    target_cost: int = 0
    backend: BACKENDS = CPU
    hyperparameters: HP_TYPE = None
    probe_cost: bool = False
    probe_time: bool = False
    probe_energy: bool = False
    log_level: int = 40


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

    best_cost: int = None
    best_state: np.ndarray = None
    best_timestep: int = None
    cost_timeseries: np.ndarray = None
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
        self._cost_tracker = None

    def solve(self, config: SolverConfig = SolverConfig()) -> SolverReport:
        """
        Create solver from problem spec and run until target_cost or timeout.

        Parameters
        ----------
        config: SolverConfig, optional
            Solver configuraiton used. Refers to SolverConfig documentation.

        Returns
        ----------
        report: SolverReport
            An object containing all the data geenrated by the execution.
        """
        run_condition, run_cfg = self._prepare_solver(config)
        self.solver_process.run(condition=run_condition, run_cfg=run_cfg)
        best_state, best_cost, best_timestep = self._get_results(
            config.hyperparameters
        )
        cost_timeseries = self._get_cost_tracking()
        self.solver_process.stop()
        return SolverReport(
            best_cost=best_cost,
            best_state=best_state,
            best_timestep=best_timestep,
            solver_config=config,
            profiler=self._profiler,
            cost_timeseries=cost_timeseries
        )

    def _prepare_solver(self, config: SolverConfig):
        self._create_solver_process(config=config)
        hps = config.hyperparameters
        num_in_ports = len(hps) if type(hps) is list else 1
        if config.probe_cost:
            if config.backend in NEUROCORES:
                # from lava.utils.loihi2_state_probes import StateProbe
                # self._cost_tracker = StateProbe()
                raise NotImplementedError
            if config.backend in CPUS:
                self._cost_tracker = Monitor()
                self._cost_tracker.probe(
                    target=self.solver_process.optimality,
                    num_steps=config.timeout)
        run_cfg = self._get_run_config(backend=config.backend,
                                       probes=[self._cost_tracker]
                                       if self._cost_tracker else None,
                                       num_in_ports=num_in_ports)
        run_condition = RunSteps(num_steps=config.timeout)
        self._prepare_profiler(config=config, run_cfg=run_cfg)
        return run_condition, run_cfg

    def _create_solver_process(self, config: SolverConfig) -> None:
        """
        Create process and model class as solver for the given problem.

        Parameters
        ----------
        config: SolverConfig
            Solver configuraiton used. Refers to SolverConfig documentation.
        """
        requirements, protocol = self._get_requirements_and_protocol(
            backend=config.backend
        )
        self._process_builder.create_solver_process(
            problem=self.problem,
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

    def _get_cost_tracking(self):
        if self._cost_tracker is None:
            return None
        if isinstance(self._cost_tracker, Monitor):
            return self._cost_tracker.get_data()[self.solver_process.name][
                self.solver_process.optimality.name].T.astype(np.int32)
        else:
            return self._cost_tracker.time_series

    def _get_run_config(self, backend: BACKENDS, probes=None,
                        num_in_ports: int = None):
        if backend in CPUS:
            from lava.lib.optimization.solvers.generic.read_gate.process \
                import ReadGate
            from lava.lib.optimization.solvers.generic.read_gate.models import \
                get_read_gate_model_class
            ReadGatePyModel = get_read_gate_model_class(num_in_ports)
            pdict = {
                self.solver_process: self.solver_model,
                ReadGate: ReadGatePyModel,
                Dense: PyDenseModelFloat,
                BoltzmannAbstract: BoltzmannAbstractModel,
                Boltzmann: BoltzmannFixed,
                QuboScif: PyModelQuboScifFixed,
            }
            return Loihi1SimCfg(
                exception_proc_model_map=pdict, select_sub_proc_model=True
            )
        elif backend in NEUROCORES:
            pdict = {self.solver_process: self.solver_model,
                     BoltzmannAbstract:
                         BoltzmannAbstractModel,
                     }
            return Loihi2HwCfg(exception_proc_model_map=pdict,
                               select_sub_proc_model=True,
                               callback_fxs=probes
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

    def _get_results(self, hyperparameters):
        best_cost, idx = self.solver_process.optimality.get()
        best_cost = (np.asarray([best_cost]).astype(np.int32) << 8) >> 8
        best_state = self._get_best_state(hyperparameters, idx)
        best_timestep = self.solver_process.solution_step.aliased_var.get()
        return best_state, int(best_cost), int(best_timestep)

    def _get_best_state(self, hyperparameters: HP_TYPE, idx: int):
        if type(hyperparameters) is list:
            raw_solution = np.asarray(
                self.solver_process.finders[int(idx)].variables_assignment.get()
            ).astype(np.int32)
            raw_solution &= 0x3F
            return raw_solution.astype(np.int8) >> 5
        else:
            return self.solver_process.variable_assignment.aliased_var.get()
