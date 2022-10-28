# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty

import numpy.typing as npt
import numpy as np
from lava.lib.optimization.problems.problems import OptimizationProblem
from lava.lib.optimization.solvers.generic.builder import SolverProcessBuilder
from lava.lib.optimization.solvers.generic.hierarchical_processes import \
    StochasticIntegrateAndFire
from lava.lib.optimization.solvers.generic.sub_process_models import \
    StochasticIntegrateAndFireModelSCIF
from lava.magma.core.resources import AbstractComputeResource, CPU, \
    Loihi2NeuroCore, NeuroCore
from lava.magma.core.run_conditions import RunContinuous, RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg, Loihi2HwCfg
from lava.magma.core.sync.protocol import AbstractSyncProtocol
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.dense.models import PyDenseModelFloat
from lava.proc.dense.process import Dense
from lava.lib.optimization.solvers.generic.read_gate.models import \
    ReadGatePyModel
from lava.lib.optimization.solvers.generic.read_gate.process import ReadGate
from lava.lib.optimization.solvers.generic.scif.models import \
    PyModelQuboScifFixed
from lava.lib.optimization.solvers.generic.scif.process import QuboScif
from lava.lib.optimization.utils.solver_tuner import SolverTuner

BACKENDS = ty.Union[CPU, Loihi2NeuroCore, NeuroCore, str]
CPUS = [CPU, "CPU"]
NEUROCORES = [Loihi2NeuroCore, NeuroCore, "Loihi2"]


def solve(problem: OptimizationProblem,
          timeout: int,
          target_cost: int = None,
          backend: BACKENDS = Loihi2NeuroCore) -> \
        npt.ArrayLike:
    """Create solver from problem spec and run until target_cost or timeout.

    Parameters
    ----------
    problem: Optimization problem to be solved.
    timeout: Maximum number of iterations (timesteps) to be run. If set to
    -1 then the solver will run continuously in non-blocking mode until a
     solution is found.
    target_cost: A cost value provided by the user as a target for the
    solution to be found by the solver, when a solution with such cost is
    found and read, execution ends.
    backend: Specifies the backend where the main solver network will be
    deployed.

    Returns
    ----------
    solution: candidate solution to the input optimization problem.
    """
    solver = OptimizationSolver(problem)
    solution = solver.solve(timeout=timeout, target_cost=target_cost,
                            backend=backend)
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

    Parameters
    ----------
    problem: Optimization problem to be solved.
    run_cfg: Run configuration for the OptimizationSolverProcess.0

    """

    def __init__(self,
                 problem: OptimizationProblem,
                 run_cfg=None):
        self.problem = problem
        self._run_cfg = run_cfg
        self._process_builder = SolverProcessBuilder()
        self.solver_process = None
        self.solver_model = None
        shape = (problem.num_variables,)
        self._hyperparameters = dict(step_size=10,
                                     steps_to_fire=10,
                                     noise_amplitude=1,
                                     init_value=np.zeros(shape),
                                     init_state=np.zeros(shape))
        self._report = dict(solved=None,
                            best_state=None,
                            cost=None,
                            target_cost=None,
                            steps_to_solution=None,
                            time_to_solution=None,
                            power_to_solution=None)

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
        """Create solver from problem spec and run until target_cost or timeout.

        Parameters
        ----------
        timeout: Maximum number of iterations (timesteps) to be run. If set to
        -1 then the solver will run continuously in non-blocking mode until a
         solution is found.
        target_cost: A cost value provided by the user as a target for the
        solution to be found by the solver, when a solution with such cost is
        found and read, execution ends.
        backend: Specifies the backend where the main solver network will be
        deployed.
        hyperparameters: A dictionary specifying values for steps_to_fire,
        noise_amplitude, step_size and init_value. All but the last are
        integers, the initial value is an array-like of initial values for the
        variables defining the problem.

        Returns
        ----------
        solution: candidate solution to the input optimization problem.

        """
        target_cost = self._validated_cost(target_cost)
        hyperparameters = hyperparameters or self.hyperparameters

        if not self.solver_process:
            self._create_solver_process(self.problem,
                                        target_cost,
                                        backend,
                                        hyperparameters)
        run_cfg = self._get_run_config(backend, timeout)
        run_condition = self._get_run_condition(timeout)
        self.solver_process._log_config.level = 20
        self.solver_process.run(condition=run_condition,
                                run_cfg=run_cfg)
        if timeout == -1:
            self.solver_process.wait()
        self._update_report(target_cost=target_cost)
        self.solver_process.stop()
        return self._report["best_state"]

    def _update_report(self,
                       target_cost=None):
        self._report["target_cost"] = target_cost
        best_state = self.solver_process.variable_assignment.aliased_var.get()
        self._report["best_state"] = best_state
        raw_cost = self.solver_process.optimality.aliased_var.get()
        cost = (raw_cost.astype(np.int32) << 8) >> 8
        self._report["cost"] = cost
        self._report["solved"] = cost <= target_cost
        steps_to_solution = self.solver_process.solution_step.get()
        self._report["steps_to_solution"] = steps_to_solution
        time_to_solution = None
        power_to_solution = None
        self._report["time_to_solution"] = time_to_solution
        self._report["power_to_solution"] = power_to_solution
        print(self._report)

    def _create_solver_process(self,
                               problem: OptimizationProblem,
                               target_cost: ty.Optional[int] = None,
                               backend: BACKENDS = None,
                               hyperparameters: ty.Dict[
                                   str, ty.Union[int, npt.ArrayLike]] = None):
        """Create process and model class as solver for the given problem.

        Parameters
        ----------
        problem: Optimization problem defined by cost and constraints which
        will be used to build the process and its model.
        target_cost: A cost value provided by the user as a target for the
        solution to be found by the solver, when a solution with such cost is
        found and read, execution ends.
        backend: Specifies the backend where the main solver network will be
        deployed.
        """
        requirements, protocol = self._get_requirements_and_protocol(backend)
        self._process_builder.create_solver_process(problem, hyperparameters
                                                    or dict())
        self._process_builder.create_solver_model(target_cost,
                                                  requirements,
                                                  protocol)
        self.solver_process = self._process_builder.solver_process
        self.solver_model = self._process_builder.solver_model

    def _get_requirements_and_protocol(self,
                                       backend: BACKENDS) -> \
            ty.Tuple[
                AbstractComputeResource, AbstractSyncProtocol]:
        """Figure out requirements and protocol for a given backend.

        Parameters
        ----------
        backend: Specifies the backend for which requirements and protocol
        classes will be returned.

        """
        protocol = LoihiProtocol
        if backend in CPUS:
            return [CPU], protocol
        elif backend in NEUROCORES:
            return [Loihi2NeuroCore], protocol
        else:
            raise NotImplementedError(str(backend) + backend_msg)

    def _get_run_config(self, backend):
        if backend in CPUS:
            pdict = {self.solver_process: self.solver_model,
                     ReadGate: ReadGatePyModel,
                     Dense: PyDenseModelFloat,
                     StochasticIntegrateAndFire:
                         StochasticIntegrateAndFireModelSCIF,
                     QuboScif: PyModelQuboScifFixed
                     }
            run_cfg = Loihi1SimCfg(exception_proc_model_map=pdict,
                                   select_sub_proc_model=True)
        elif backend in NEUROCORES:
            pdict = {self.solver_process: self.solver_model,
                     StochasticIntegrateAndFire:
                         StochasticIntegrateAndFireModelSCIF,
                     }
            pre_run_fxs, post_run_fxs = [], []
            run_cfg = Loihi2HwCfg(exception_proc_model_map=pdict,
                                  select_sub_proc_model=True,
                                  pre_run_fxs=pre_run_fxs,
                                  post_run_fxs=post_run_fxs
                                  )
        else:
            raise NotImplementedError(str(backend) + backend_msg)
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

    def tune(self, params_grid: ty.Dict,
             timeout: int,
             target_cost: int = 0,
             backend: BACKENDS = CPU,
             stopping_condition: ty.Callable[[float, int], bool] = None) -> \
            ty.Tuple[ty.Dict, bool]:
        """
        Provides an interface to SolverTuner to search hyperparameters on the
        specififed grid. Returns the optimized hyperparameters.
        """
        solver_tuner = SolverTuner(params_grid)
        solver_parameters = {"timeout": timeout,
                             "target_cost": target_cost,
                             "backend": backend}
        hyperparameters, success = solver_tuner.tune(self,
                                                     solver_parameters,
                                                     stopping_condition)
        if success:
            self.hyperparameters = hyperparameters
        return hyperparameters, success


# TODO throw an error if L2 is not present and the user tries to use it.
backend_msg = f""" was requested as backend. However,
the solver currently supports only Loihi 2 and CPU backends.
These can be specified by calling solve with any of the following:

    backend = "CPU"
    backend = "Loihi2"
    backend = CPU
    backend = Loihi2NeuroCore
    backend = NeuroCore

The explicit resource classes can be imported from
lava.magma.core.resources
"""
