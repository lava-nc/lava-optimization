# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
from lava.magma.core.run_conditions import RunSteps, RunContinuous
from lava.magma.core.run_configs import Loihi1SimCfg

from lava.lib.optimization.problems.problems import OptimizationProblem
from lava.lib.optimization.solvers.generic.processes import SolverProcessBuilder
from lava.lib.optimization.solvers.generic.models import SolverModelBuilder


# from lava.utils.profiler import LavaProfiler
class LavaProfiler:
    # todo placeholder while profiler is implemented.
    pass

    def profile(self, process):
        pass


class OptimizationSolver:
    """Wrapper over the actual Lava OptimizationSolverProcess.

    Parameters
    ----------
    configuration: Configuration parameters for the OptimizationProcessSolver.
    """

    def __init__(self, run_cfg=None):
        self._run_cfg = run_cfg
        self._process_builder = SolverProcessBuilder()
        self._solver_process = None
        self._solver_model = None

    @property
    def run_cfg(self):
        """Run configuration for process model selection."""
        return self._run_cfg

    @run_cfg.setter
    def run_cfg(self, value):
        self._run_cfg = value

    def solve(
        self,
        problem: OptimizationProblem,
        timeout: int,
        target_cost: int = None,
        profiling: bool = False,
    ):
        """Create solver from problem specs and run until solution or timeout.

        Parameters
        ----------
        problem: Optimization problem to be solved.
        timeout: Maximum number of iterations/timesteps to be run, if set to
        -1 then the solver will run continuously in non-blocking mode until a
         solution is found.
        profiling: Whether to profile the run. This will measure or estimate
        energy and time depending on the backend.

        Returns
        ----------
        solution: candidate solution to the input optimization problem.

        """
        solver_process = self._create_solver_process(problem)
        solver_model = self._create_solver_model(solver_process, target_cost)
        if profiling:
            profiler = LavaProfiler()
            solver_process = profiler.profile(solver_process)
        pdict = {solver_process: solver_model}
        solver_process.run(
            condition=RunContinuous()
            if timeout == -1
            else RunSteps(num_steps=timeout),
            run_cfg=Loihi1SimCfg(  # select_sub_proc_model=True,
                exception_proc_model_map=pdict
            ),
        )
        if timeout == -1:
            solver_process.wait()
        solution = solver_process.variable_assignment.aliased_var.get()
        solver_process.stop()
        return solution

    def _create_solver_process(self, problem):
        self._process_builder.create_constructor(problem)
        self._solver_process = self._process_builder.solver_process
        return self._solver_process

    def _create_solver_model(self, solver_process, target_cost):
        model_builder = SolverModelBuilder(solver_process)
        model_builder.create_constructor(target_cost)
        self._solver_model = model_builder.solver_model
        return self._solver_model

    @classmethod
    def get_process(cls, spec=None):
        return

    def verify_solution(self, solution):
        pass
