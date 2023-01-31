# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
from dataclasses import dataclass

import numpy.typing as npt
import numpy as np
from lava.lib.optimization.problems.problems import OptimizationProblem
from lava.lib.optimization.solvers.generic.builder import SolverProcessBuilder
from lava.lib.optimization.solvers.generic.hierarchical_processes import (
    BoltzmannAbstract,
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
from lava.magma.core.run_conditions import RunContinuous, RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg, Loihi2HwCfg
from lava.magma.core.sync.protocol import AbstractSyncProtocol
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.dense.models import PyDenseModelFloat
from lava.proc.dense.process import Dense
from lava.lib.optimization.solvers.generic.read_gate.models import \
    ReadGatePyModel
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
    """Dataclass to store and validate OptimizationSolver configurations."""

    timeout: int = 1e3
    target_cost: int = 0
    backend: BACKENDS = CPU
    hyperparameters: dict = None
    probe_time: bool = False
    probe_energy: bool = False
    log_level: int = 40


@dataclass(frozen=True)
class SolverReport:
    best_cost: int = None
    best_state: np.ndarray = None
    best_timestep: int = None
    solver_config: SolverConfig = None
    profiler = None


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


        Returns
        ----------
        report: SolverReport
            An object containing all the data geenrated by the execution.
        """
        run_condition, run_cfg = self._prepare_solver(config)
        self.solver_process.run(condition=run_condition, run_cfg=run_cfg)
        self.solver_process.stop()

        best_state, best_cost, best_timestep = self._get_results()

        report = SolverReport(
            best_cost=best_cost,
            best_state=best_state,
            best_timestep=best_timestep,
            solver_config=config,
            profiler=self._profiler
        )

        return report

    def _prepare_solver(self, config: SolverConfig):
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
            from lava.utils.profiler import Profiler
            self._profiler = Profiler.init(run_cfg)
            if config.probe_time:
                self._profiler.execution_time_probe(num_steps=config.timeout)
            if config.probe_energy:
                self._profiler.energy_probe(num_steps=config.timeout)

    def _get_results(self):
        best_state = self.solver_process.variable_assignment.aliased_var.get()
        best_cost = self.solver_process.optimality.aliased_var.get()
        best_cost = (best_cost.astype(np.int32) << 8) >> 8
        best_timestep = self.solver_process.solution_step.aliased_var.get()
        return best_state, int(best_cost), int(best_timestep)
