# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
from lava.lib.optimization.problems.problems import OptimizationProblem
from lava.lib.optimization.solvers.generic.processes import \
    OptimizationSolverProcess
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg


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

    @property
    def run_cfg(self):
        """Run configuration for process model selection."""
        return self._run_cfg

    @run_cfg.setter
    def run_cfg(self, value):
        self._run_cfg = value

    def solve(self,
              problem: OptimizationProblem,
              timeout: int,
              profiling: bool = False):
        """Create solver from problem specs and run until solution or timeout.

        Parameters
        ----------
        problem: Optimization problem to be solved.
        timeout: Maximum number of iterations/timesteps to be run.
        profiling: Whether to profile the run. This will measure or estimate
        energy and time depending on the backend.

        Returns
        ----------
        solution: candidate solution to the input optimization problem.

        """
        self.solver_process = OptimizationSolverProcess(problem=problem)
        if profiling:
            profiler = LavaProfiler()
            solver_process = profiler.profile(self.solver_process)
        else:
            solver_process = self.solver_process
        solver_process.run(condition=RunSteps(num_steps=timeout),
                           run_cfg=Loihi1SimCfg(select_sub_proc_model=True))
        solution = self.solver_process.variable_assignment.get()
        self.solver_process.stop()
        return solution
