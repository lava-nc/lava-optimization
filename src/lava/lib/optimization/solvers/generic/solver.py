# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty

import numpy.typing as npt
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

    @property
    def run_cfg(self):
        """Run configuration for process model selection."""
        return self._run_cfg

    @run_cfg.setter
    def run_cfg(self, value):
        self._run_cfg = value

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
        run_cfg = None
        if not self.solver_process:
            self._create_solver_process(self.problem, target_cost, backend,
                                        hyperparameters)
        if backend in CPUS:
            pdict = {self.solver_process: self.solver_model,
                     ReadGate: ReadGatePyModel,
                     Dense: PyDenseModelFloat,
                     StochasticIntegrateAndFire:
                         StochasticIntegrateAndFireModelSCIF,
                     QuboScif: PyModelQuboScifFixed,
                     }
            run_cfg = Loihi1SimCfg(exception_proc_model_map=pdict,
                                   select_sub_proc_model=True)
        elif backend in NEUROCORES:
            pdict = {self.solver_process: self.solver_model,
                     StochasticIntegrateAndFire:
                         StochasticIntegrateAndFireModelSCIF,
                     }
            run_cfg = Loihi2HwCfg(exception_proc_model_map=pdict,
                                  select_sub_proc_model=True)
        else:
            raise NotImplementedError(str(backend) + backend_msg)
        self.solver_process._log_config.level = 20
        self.solver_process.run(
            condition=RunContinuous()
            if timeout == -1
            else RunSteps(num_steps=timeout + 1),
            run_cfg=run_cfg,
        )
        if timeout == -1:
            self.solver_process.wait()
        solution = self.solver_process.variable_assignment.aliased_var.get()
        self.solver_process.stop()
        return solution

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
