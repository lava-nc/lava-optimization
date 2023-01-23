# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
from dataclasses import dataclass
import typing as ty
from lava.proc.monitor.process import Monitor

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
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg, Loihi2HwCfg
from lava.magma.core.sync.protocol import AbstractSyncProtocol
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.dense.models import PyDenseModelFloat
from lava.proc.dense.process import Dense
from lava.lib.optimization.solvers.generic.read_gate.models import \
    ReadGatePyModel
from lava.lib.optimization.solvers.generic.read_gate.process import ReadGate
from lava.lib.optimization.solvers.generic.scif.models import BoltzmannFixed
from lava.lib.optimization.solvers.generic.scif.process import Boltzmann

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
    timeout: int = 1e3
    target_cost: int = None
    backend: BACKENDS = CPU
    hyperparameters: dict = None
    probe_cost: bool = False
    probe_time: bool = False
    probe_energy: bool = False
    log_level: int = 40

    @timeout.setter
    def timeout(self, value) -> None:
        self._validate(timeout=value)

    @staticmethod
    def _validated_cost(target_cost: int) -> int:
        if target_cost != int(target_cost):
            raise ValueError(f"target_cost has to be an integer, received "
                             f"{target_cost}")
        return int(target_cost)

    @staticmethod
    def _validated_timeout(timeout: int) -> int:
        if timeout < 0:
            raise NotImplementedError("The timeout must be > 0.")
        return int(timeout)


@dataclass(frozen=True)
class SolverReport:
    best_cost: int = None
    best_state: np.ndarray = None
    best_timestep: int = None
    cost_timeseries: np.ndarray = None
    solver_config: SolverConfig = None

    def plot_cost_timeseries(self, filename: str = None) -> None:
        if self.cost_timeseries is None:
            # what to do?
        from matplotlib import pyplot as plt
        plt.plot(self.cost_timeseries, "ro")
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)



def solve(problem: OptimizationProblem,
          timeout: int,
          target_cost: int = None,
          backend: BACKENDS = CPU,
          ) -> npt.ArrayLike:
    """Create solver from problem spec and run until target_cost or timeout.

    Parameters
    ----------
    problem: OptimizationProblem
        Optimization problem to be solved.
    timeout: int
        Maximum number of iterations (timesteps) to be run. If set to -1 then
        the solver will run continuously in non-blocking mode until a solution
        is found.
    target_cost: int, optional
        A cost value provided by the user as a target for the solution to be
        found by the solver, when a solution with such cost is found and read,
        execution ends.
    backend: BACKENDS, optional
        Specifies the backend where the main solver network will be deployed.

    Returns
    ----------
    solution: npt.ArrayLike
        Candidate solution to the input optimization problem.
    """
    solver = OptimizationSolver(problem)
    report = solver.solve(
        config=SolverConfig(
            timeout=timeout,
            target_cost=target_cost,
            backend=backend
        )
    )
    return report.best_state


class OptimizationSolver:
    """Generic solver for constrained optimization problems defined by
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

    def solve(self, config: SolverConfig = SolverConfig()) -> SolverReport:
        """
        Create solver from problem spec and run until target_cost or timeout.

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
        config: SolverConfig, optional

        Returns
        ----------
        report: SolverReport

        """
        self._create_solver_process(config)
        run_cfg = self._get_run_config(config.backend)
        run_condition = RunSteps(num_steps=timeout)

        # from lava.utils.profiler import Profiler
        # self._profiler = Profiler.init(run_cfg)
        # self._profiler.execution_time_probe(num_steps=timeout)
        # self._profiler.energy_probe(num_steps=timeout)

        if config.probe_cost:
            monitor = Monitor()
            monitor.probe(target=self.solver_process.optimality, num_steps=timeout)

        self.solver_process.run(condition=run_condition, run_cfg=run_cfg)
        best_state = self.solver_process.variable_assignment.aliased_var.get()
        best_cost = self.solver_process.optimality.aliased_var.get()
        best_cost = (best_cost.astype(np.int32) << 8) >> 8
        best_timestep = self.solver_process.solution_step.aliased_var.get()
        if config.probe_cost:
            cost_timeseries = monitor.get_data()[self.solver_process.name][
                self.solver_process.optimality.name
            ]
            print(cost_timeseries)
        self.solver_process.stop()

        report = SolverReport(best_cost=best_cost,
                              best_state=best_state,
                              best_timestep=best_timestep)

        return report

    def _create_solver_process(self, config: SolverConfig) -> None:
        """
        Create process and model class as solver for the given problem.

        Parameters
        ----------
        config: SolverConfig

        """
        requirements, protocol = self._get_requirements_and_protocol(
            config.backend
        )
        self._process_builder.create_solver_process(
            self.problem, config.hyperparameters or dict()
        )
        self._process_builder.create_solver_model(
            config.target_cost, requirements, protocol
        )
        self.solver_process = self._process_builder.solver_process
        self.solver_model = self._process_builder.solver_model
        self.solver_process._log_config.level = config.log_level

    def _get_requirements_and_protocol(
            self, backend: BACKENDS
    ) -> ty.Tuple[AbstractComputeResource, AbstractSyncProtocol]:
        """Figure out requirements and protocol for a given backend.

        Parameters
        ----------
        backend: BACKENDS
            Specifies the backend for which requirements and protocol classes
            will be returned.
        """
        protocol = LoihiProtocol
        if backend in CPUS:
            return [CPU], protocol
        elif backend in NEUROCORES:
            return [Loihi2NeuroCore], protocol
        else:
            raise NotImplementedError(str(backend) + BACKEND_MSG)

    def _get_run_config(self, backend):
        if backend in CPUS:
            pdict = {self.solver_process: self.solver_model,
                     ReadGate: ReadGatePyModel,
                     Dense: PyDenseModelFloat,
                     BoltzmannAbstract:
                         BoltzmannAbstractModel,
                     Boltzmann: BoltzmannFixed
                     }
            run_cfg = Loihi1SimCfg(exception_proc_model_map=pdict,
                                   select_sub_proc_model=True)
        elif backend in NEUROCORES:
            pdict = {self.solver_process: self.solver_model,
                     BoltzmannAbstract:
                         BoltzmannAbstractModel,
                     }
            run_cfg = Loihi2HwCfg(exception_proc_model_map=pdict,
                                  select_sub_proc_model=True)
        else:
            raise NotImplementedError(str(backend) + BACKEND_MSG)
        return run_cfg
