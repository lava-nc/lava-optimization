# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty

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


class SolverConfig:

    def __init__(self,
                 backend: BACKENDS = "CPU",
                 hyperparameters: dict = {},
                 probe_cost: bool = False,
                 probe_time: bool = False,
                 probe_energy: bool = False) -> None:
        if backend not in NEUROCORES + CPUS:
            raise ValueError(f"{backend} {BACKEND_MSG}")
        if backend in CPUS and (probe_time or probe_energy):
            raise ValueError(
                f"Time and energy probing are only enabled on Loihi backend.")
        if probe_time and probe_energy:
            raise ValueError("Time and energy probing should be executed in"
                             "in different runs for better accuracy.")
        self._backend = backend
        self._hyperparameters = hyperparameters
        self._probe_cost = probe_cost
        self._probe_time = probe_time
        self._probe_energy = probe_energy

    @property
    def backend(self) -> BACKENDS:
        return self._backend

    @property
    def hyperparameters(self) -> dict:
        return self._hyperparameters

    @property
    def probe_cost(self) -> bool:
        return self._probe_cost

    @property
    def probe_time(self) -> bool:
        return self._probe_time

    @property
    def probe_energy(self) -> bool:
        return self._probe_energy


class SolverReport:
    pass


def solve(problem: OptimizationProblem,
          timeout: int,
          target_cost: int = None,
          backend: BACKENDS = Loihi2NeuroCore,
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
    solution = solver.solve(
        timeout=timeout, target_cost=target_cost, backend=backend
    )
    return solution


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

    def solve(self,
              timeout: int,
              target_cost: int = 0,
              config: SolverConfig = SolverConfig()) -> SolverReport:
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
        timeout = self._validated_timeout(timeout)
        target_cost = self._validated_cost(target_cost)

        self._create_solver_process(target_cost, config)
        run_cfg = self._get_run_config(config.backend)
        run_condition = RunSteps(num_steps=timeout)

        # from lava.utils.profiler import Profiler
        # self._profiler = Profiler.init(run_cfg)
        # self._profiler.execution_time_probe(num_steps=timeout)
        # self._profiler.energy_probe(num_steps=timeout)

        self.solver_process.run(condition=run_condition, run_cfg=run_cfg)
        best_state = self.solver_process.variable_assignment.aliased_var.get()
        raw_cost = self.solver_process.optimality.aliased_var.get()
        cost = (raw_cost.astype(np.int32) << 8) >> 8
        steps_to_solution = self.solver_process.solution_step.aliased_var.get()

        self.solver_process.stop()
        report = SolverReport()
        return best_state

    def _create_solver_process(self,
                               target_cost: ty.Optional[int] = None,
                               config: SolverConfig = None):
        """
        Create process and model class as solver for the given problem.

        Parameters
        ----------
        target_cost: int, optional
            A cost value provided by the user as a target for the solution to be
            found by the solver, when a solution with such cost is found and
            read, execution ends.
        config: SolverConfig

        """
        requirements, protocol = self._get_requirements_and_protocol(
            config.backend
        )
        self._process_builder.create_solver_process(
            self.problem, config.hyperparameters
        )
        self._process_builder.create_solver_model(
            target_cost, requirements, protocol
        )
        self.solver_process = self._process_builder.solver_process
        self.solver_model = self._process_builder.solver_model
        self.solver_process._log_config.level = 40

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
