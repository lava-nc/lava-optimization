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
from lava.lib.optimization.utils.solver_tuner import SolverTuner

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


def solve(
        problem: OptimizationProblem,
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

    def __init__(self, problem: OptimizationProblem, run_cfg=None):
        """
        Constructor for the OptimizationSolver class.

        Parameters
        ----------
        problem: OptimizationProblem
            Optimization problem to be solved.
        run_cfg: Any
            Run configuration for the OptimizationSolverProcess.
        """
        self.problem = problem
        self._run_cfg = run_cfg
        self._process_builder = SolverProcessBuilder()
        self.solver_process = None
        self.solver_model = None
        self._hyperparameters = dict(temperature=10,
                                     refract=1,
                                     init_value=0)
        self._report = dict(solved=None,
                            best_state=None,
                            cost=None,
                            target_cost=None,
                            steps_to_solution=None,
                            time_to_solution=None)
        self._profiler = None

    @property
    def run_cfg(self):
        """Run configuration for process model selection."""
        return self._run_cfg

    @run_cfg.setter
    def run_cfg(self, value):
        self._run_cfg = value

    @property
    def hyperparameters(self):
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self,
                        value: ty.Dict[str, ty.Union[int, npt.ArrayLike]]):
        self._hyperparameters = value

    @property
    def last_run_report(self):
        return self._report

    def solve(self,
              timeout: int,
              target_cost: int = 0,
              backend: BACKENDS = CPU,
              hyperparameters: ty.Dict[
                  str, ty.Union[int, npt.ArrayLike]] = None) \
            -> npt.ArrayLike:
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
        backend: BACKENDS, optional
            Specifies the backend where the main solver network will be
            deployed.
        hyperparameters: ty.Dict[str, ty.Union[int, npt.ArrayLike]], optional
            A dictionary specifying values for steps_to_fire, noise_amplitude,
            step_size and init_value. All but the last are integers, the initial
            value is an array-like of initial values for the variables defining
            the problem.

        Returns
        ----------
        solution: npt.ArrayLike
            Candidate solution to the input optimization problem.
        """
        target_cost = self._validated_cost(target_cost)
        hyperparameters = hyperparameters or self.hyperparameters
        self._create_solver_process(self.problem,
                                    target_cost,
                                    backend,
                                    hyperparameters)
        run_cfg = self._get_run_config(backend)
        run_condition = self._get_run_condition(timeout)
        self.solver_process._log_config.level = 40
        self.solver_process.run(condition=run_condition, run_cfg=run_cfg)
        if timeout == -1:
            self.solver_process.wait()
        self._update_report(target_cost=target_cost)

        debug_var = self.solver_process.debug.get()

        self.solver_process.stop()

        return self._report["best_state"]

    def measure_time_to_solution(
            self,
            timeout: int,
            target_cost: int,
            backend: BACKENDS,
            hyperparameters: ty.Dict[str, ty.Union[int, npt.ArrayLike]] = None,
    ):
        """
        Run solver until target_cost or timeout and returns total time to
        solution.

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
        backend: BACKENDS
            At the moment, only the Loihi2 backend can be used.
        hyperparameters: ty.Dict[str, ty.Union[int, npt.ArrayLike]], optional
            A dictionary specifying values for steps_to_fire, noise_amplitude,
            step_size and init_value. All but the last are integers, the initial
            value is an array-like of initial values for the variables defining
            the problem.

        Returns
        ----------
        time_to_solution: npt.ArrayLike
            Total time to solution in seconds.
        """
        if timeout == -1:
            raise ValueError("For time measurements timeout " "cannot be -1")
        if backend not in NEUROCORES:
            raise ValueError(f"Time measurement can only be performed on "
                             f"Loihi2 backend, got {backend}.")

        target_cost = self._validated_cost(target_cost)
        hyperparameters = hyperparameters or self.hyperparameters
        self._create_solver_process(self.problem,
                                    target_cost,
                                    backend,
                                    hyperparameters)
        run_cfg = self._get_run_config(backend)
        run_condition = self._get_run_condition(timeout)
        self.solver_process._log_config.level = 40

        from lava.utils.profiler import Profiler
        self._profiler = Profiler.init(run_cfg)
        self._profiler.execution_time_probe(num_steps=timeout - 1)

        self.solver_process.run(condition=run_condition, run_cfg=run_cfg)
        time_to_solution = float(np.sum(self._profiler.execution_time))
        self._update_report(target_cost=target_cost,
                            time_to_solution=time_to_solution)
        self.solver_process.stop()
        return time_to_solution

    def measure_energy_to_solution(
            self,
            timeout: int,
            target_cost: int,
            backend: BACKENDS,
            hyperparameters: ty.Dict[str, ty.Union[int, npt.ArrayLike]] = None,
    ):
        """
        Run solver until target_cost or timeout and returns energy to solution.

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
        backend: BACKENDS
            At the moment, only the Loihi2 backend can be used.
        hyperparameters: ty.Dict[str, ty.Union[int, npt.ArrayLike]], optional
            A dictionary specifying values for steps_to_fire, noise_amplitude,
            step_size and init_value. All but the last are integers, the initial
            value is an array-like of initial values for the variables defining
            the problem.

        Returns
        ----------
        energy_to_solution: npt.ArrayLike
            Total energy to solution in Joule.
        """
        if timeout == -1:
            raise ValueError("For time measurements timeout " "cannot be -1")
        if backend not in NEUROCORES:
            raise ValueError(f"Enegy measurement can only be performed on "
                             f"Loihi2 backend, got {backend}.")

        target_cost = self._validated_cost(target_cost)
        hyperparameters = hyperparameters or self.hyperparameters
        self._create_solver_process(self.problem,
                                    target_cost,
                                    backend,
                                    hyperparameters)
        run_cfg = self._get_run_config(backend)
        run_condition = self._get_run_condition(timeout)
        self.solver_process._log_config.level = 40

        from lava.utils.profiler import Profiler
        self._profiler = Profiler.init(run_cfg)
        self._profiler.execution_time_probe(num_steps=timeout - 1)
        self._profiler.energy_probe(num_steps=timeout - 1)

        self.solver_process.run(condition=run_condition, run_cfg=run_cfg)
        energy_to_solution = float(self._profiler.energy)
        self._update_report(target_cost=target_cost,
                            energy_to_solution=energy_to_solution)
        self.solver_process.stop()
        return energy_to_solution

    def _update_report(self, target_cost=None,
                       time_to_solution=None,
                       energy_to_solution=None):
        self._report["target_cost"] = target_cost
        best_state = self.solver_process.variable_assignment.aliased_var.get()
        self._report["best_state"] = best_state
        raw_cost = self.solver_process.optimality.aliased_var.get()
        cost = (raw_cost.astype(np.int32) << 8) >> 8
        self._report["cost"] = cost
        self._report["solved"] = cost == target_cost
        steps_to_solution = self.solver_process.solution_step.aliased_var.get()
        self._report["steps_to_solution"] = steps_to_solution
        self._report["time_to_solution"] = time_to_solution
        self._report["energy_to_solution"] = energy_to_solution
        print(self._report)

    def _create_solver_process(self,
                               problem: OptimizationProblem,
                               target_cost: ty.Optional[int] = None,
                               backend: BACKENDS = None,
                               hyperparameters: ty.Dict[
                                   str, ty.Union[int, npt.ArrayLike]] = None):
        """
        Create process and model class as solver for the given problem.

        Parameters
        ----------
        problem: OptimizationProblem
            Optimization problem defined by cost and constraints which will be
            used to build the process and its model.
        target_cost: int, optional
            A cost value provided by the user as a target for the solution to be
            found by the solver, when a solution with such cost is found and
            read, execution ends.
        backend: BACKENDS, optional
            Specifies the backend where the main solver network will be
            deployed.
        hyperparameters: ty.Dict[str, ty.Union[int, npt.ArrayLike]]
        """
        requirements, protocol = self._get_requirements_and_protocol(backend)
        self._process_builder.create_solver_process(
            problem, hyperparameters or dict()
        )
        self._process_builder.create_solver_model(
            target_cost, requirements, protocol
        )
        self.solver_process = self._process_builder.solver_process
        self.solver_model = self._process_builder.solver_model

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
            pre_run_fxs, post_run_fxs = [], []
            run_cfg = Loihi2HwCfg(exception_proc_model_map=pdict,
                                  select_sub_proc_model=True,
                                  pre_run_fxs=pre_run_fxs,
                                  post_run_fxs=post_run_fxs
                                  )
        else:
            raise NotImplementedError(str(backend) + BACKEND_MSG)
        return run_cfg

    def _validated_cost(self, target_cost):
        if target_cost != int(target_cost):
            raise ValueError(f"target_cost has to be an integer, received "
                             f"{target_cost}")
        return int(target_cost)

    def _get_run_condition(self, timeout):
        if timeout == -1:
            return RunContinuous()
        else:
            return RunSteps(num_steps=timeout + 1)

    def _add_time_to_run_config(self, run_cfg, timeout):
        pre_run_fxs, post_run_fxs = self._benchmarker.get_time_measurement_cfg(
            num_steps=timeout + 1
        )
        run_cfg.pre_run_fxs += pre_run_fxs
        run_cfg.post_run_fxs += post_run_fxs
