# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import itertools as it
import typing as ty

import random


class SolverTuner:
    """Class to find and set hyperparameters for an OptimizationSolver."""

    def __init__(self, params_grid: ty.Dict):
        """
        Instantiate a SolverTuner, defining the admissible search space.

        Parameters
        ----------
        params_grid: dict
            Dictionary where the keys are the name of the hyperparameters to be
            tuned, and the values are tuples containing all the admissible
            values for the associated hyperparameter.
        """
        self._params_grid = params_grid

    def tune(self,
             solver,
             solver_parameters: ty.Dict,
             stopping_condition: ty.Callable[[float, int], bool] = None):
        """Find and set solving hyperparameters for an OptimizationSolver"""
        # TODO : Check that hyperparameters are arguments for solver
        best_hyperparameters, best_cost, best_step_to_sol = self._random_search(
            solver,
            solver_parameters,
            stopping_condition)
        succeeded = stopping_condition(best_cost, best_step_to_sol)
        return best_hyperparameters, succeeded

    def _random_search(self,
                       solver,
                       solver_parameters: ty.Dict,
                       stopping_condition: ty.Callable[
                           [float, int], bool] = None
                       ) -> ty.Union[ty.NoReturn, ty.Dict]:

        best_hyperparameters = None
        best_cost = float("inf")
        best_step_to_sol = float("inf")
        problem = solver.problem
        params_names = self._params_grid.keys()
        params_grid = list(
            it.product(*map(lambda k: self._params_grid[k], params_names)))
        random.shuffle(params_grid)

        for params in params_grid:
            hyperparameters = dict(zip(params_names, params))
            solver_parameters["hyperparameters"] = hyperparameters
            solution = solver.solve(**solver_parameters)
            # TODO : Implement logic for CSP problems
            cost = solver.last_run_report["cost"]
            step_to_sol = solver.last_run_report["steps_to_solution"]
            if cost is not None and cost <= best_cost and (
                    step_to_sol < best_step_to_sol):
                best_hyperparameters = hyperparameters
                best_cost = cost
                best_step_to_sol = step_to_sol
                print(f"""Better hyperparameters configuration found!\
                        \nHyperparameters: {best_hyperparameters}\
                        \nBest cost: {best_cost}\
                        \nBest step-to-solution: {best_step_to_sol}""")
            if stopping_condition is not None and (
                    stopping_condition(best_cost, best_step_to_sol)):
                break
            # TODO : Add internal logging

        return best_hyperparameters, best_cost, best_step_to_sol
