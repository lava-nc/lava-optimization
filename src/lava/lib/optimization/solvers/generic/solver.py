# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
from dataclasses import dataclass
from lava.utils.profiler import Profiler

import numpy.typing as npt
import numpy as np
from lava.lib.optimization.problems.problems import OptimizationProblem
from lava.lib.optimization.solvers.generic.builder import SolverProcessBuilder
from lava.lib.optimization.solvers.generic.hierarchical_processes import (
    BoltzmannAbstract,
)
from lava.lib.optimization.solvers.generic import sconfig

from lava.lib.optimization.solvers.generic.sub_process_models import (
    BoltzmannAbstractModel,
)
from lava.magma.core.resources import (
    AbstractComputeResource,
    CPU,
    Loihi2NeuroCore,
    NeuroCore,
)
from lava.magma.core.run_conditions import RunContinuous, RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg, Loihi2HwCfg
from lava.magma.core.sync.protocol import AbstractSyncProtocol
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.dense.models import PyDenseModelFloat
from lava.proc.dense.process import Dense

from lava.lib.optimization.solvers.generic.read_gate.process import ReadGate
from lava.lib.optimization.solvers.generic.scif.models import BoltzmannFixed, \
    PyModelQuboScifFixed
from lava.lib.optimization.solvers.generic.scif.process import Boltzmann, \
    QuboScif

BACKENDS = ty.Union[CPU, Loihi2NeuroCore, NeuroCore, str]
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
    hyperparameters: ty.Dict[str, ty.Union[int, npt.ArrayLike]], optional
        A dictionary specifying values for steps_to_fire, noise_amplitude,
        step_size and init_value. All but the last are integers, the initial
        value is an array-like of initial values for the variables defining
        the problem.
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
    hyperparameters: ty.Union[ty.Dict, ty.List[ty.Dict]] = None
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
    solver_config: SolverConfig = None
    profiler: Profiler = None


def solve(problem: OptimizationProblem,
          config: SolverConfig = SolverConfig()) -> np.ndarray:
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
        best_state, best_cost, best_timestep = self._get_results()
        if type(config.hyperparameters) is list:
            optimality, idx = self.solver_process.optimality.get()
            is_nebm = config.hyperparameters[int(idx)]['neuron_model'] == "nebm"
            raw_solution =np.asarray(self.solver_process.finders[
                int(idx)].variables_assignment.get()).astype(np.int32)
            raw_solution &= (0xFF if is_nebm else 0x3F)
            best_state = (raw_solution.astype(np.int8) >> (6 if is_nebm else 5))
        self.solver_process.stop()
        report = SolverReport(
            best_cost=best_cost,
            best_state=best_state,
            best_timestep=best_timestep,
            solver_config=config,
            profiler=self._profiler
        )
        return report

    def _prepare_solver(self, config: SolverConfig):
        hyperparameters=config.hyperparameters
        sconfig.num_in_ports = len(hyperparameters) if type(hyperparameters) \
                                                       is list else 1
        self._create_solver_process(config=config)
        run_cfg = self._get_run_config(backend=config.backend)
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
            hyperparameters=config.hyperparameters or dict()
        )
        self._process_builder.create_solver_model(
            target_cost=config.target_cost,
            requirements=requirements,
            protocol=protocol
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

    def _get_run_config(self, backend: BACKENDS):
        if backend in CPUS:
            from lava.lib.optimization.solvers.generic.read_gate.models import ReadGatePyModel
            pdict = {self.solver_process: self.solver_model,
                     ReadGate: ReadGatePyModel,
                     Dense: PyDenseModelFloat,
                     BoltzmannAbstract:
                         BoltzmannAbstractModel,
                     Boltzmann: BoltzmannFixed,
                     QuboScif: PyModelQuboScifFixed
                     }
            return Loihi1SimCfg(exception_proc_model_map=pdict,
                                select_sub_proc_model=True)
        elif backend in NEUROCORES:
            pdict = {self.solver_process: self.solver_model,
                     BoltzmannAbstract:
                         BoltzmannAbstractModel,
                     }
            return Loihi2HwCfg(exception_proc_model_map=pdict,
                               select_sub_proc_model=True)
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

    def _get_results(self):
        best_state = self.solver_process.variable_assignment.aliased_var.get()
        best_cost, idx = self.solver_process.optimality.get()
        best_cost = (np.asarray([best_cost]).astype(np.int32) << 8) >> 8
        best_timestep = self.solver_process.solution_step.aliased_var.get()
        return best_state, int(best_cost), int(best_timestep)
