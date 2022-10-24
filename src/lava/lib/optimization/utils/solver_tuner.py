# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import itertools as it
import typing as ty

from lava.lib.optimization.solvers.generic.solver import OptimizationSolver
import random

class SolverTuner:
    """Class to find and set hyperparameters for an OptimizationSolver."""

    def __init__(self,
                 params_grid: ty.Dict
                 ):
        """
        Instantiate a SolverTuner, defining the admissible search space.

        :param params_grid:
        """
        self._params_grid = params_grid

    def tune(self,
             solver: OptimizationSolver,
             solver_parameters: ty.Dict,
             stopping_condition: ty.Callable[[float, int], bool] = None):
        """Find and set solving hyperparameters for an OptimizationSolver"""
        # TODO : Check that hyperparameters are arguments for solver
        best_hyperparameters, best_cost, best_step_to_sol = self._perform_grid_search(
            solver,
            solver_parameters,
            stopping_condition)
        succeeded = stopping_condition(best_cost, best_step_to_sol)
        return best_hyperparameters, succeeded

    def _perform_grid_search(self,
                             solver: OptimizationSolver,
                             solver_parameters: ty.Dict,
                             stopping_condition: ty.Callable[
                                 [float, int], bool] = None
                             ) -> ty.Union[ty.NoReturn, ty.Dict]:

        best_hyperparameters = None
        best_cost = float("inf")
        best_step_to_sol = float("inf")
        problem = solver.problem
        params_names = self._params_grid.keys()

        for params in it.product(
                *map(lambda k: self._params_grid[k], params_names)):
            hyperparameters = dict(zip(params_names, params))
            solver_parameters["hyperparameters"] = hyperparameters
            solution = solver.solve(**solver_parameters)
            # TODO : Implement logic for CSP problems
            cost = problem.evaluate_cost(solution)
            step_to_sol = random.randint(0, 1000) # TODO : check this
            if cost is not None and cost <= best_cost and step_to_sol < best_step_to_sol:
                best_hyperparameters = hyperparameters
                best_cost = cost
                best_step_to_sol = step_to_sol
                print(f"""Better hyperparameters configuration found!\
                        \nHyperparameters: {best_hyperparameters}\
                        \nBest cost: {best_cost}\
                        \nBest step-to-solution: {best_step_to_sol}""")
            if stopping_condition is not None and stopping_condition(best_cost,
                                                                     best_step_to_sol):
                break
            # TODO : Add internal logging

        return best_hyperparameters, best_cost, best_step_to_sol
