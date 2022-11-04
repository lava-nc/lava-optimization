# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import itertools as it
import typing as ty

import random
import numpy as np


class SolverTuner:
    """Utility class to optimize hyper-parameters by random search."""

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
        self._params_keys = list(params_grid.keys())
        self._params_grid = list(
            it.product(*map(lambda k: params_grid[k], self._params_keys)))
        self._store_dtype = SolverTuner._build_store_dtype(params_grid,
                                                           self._params_keys)
        self._store = np.array([], dtype=self._store_dtype)

    def tune(self,
             solver,
             solver_params: ty.Dict,
             fitness_fn: ty.Callable[[float, int], float],
             fitness_target: float = None,
             seed: int = 0):
        """
        Perform random search to optimize solver hyper-parameters based on a
        fitness function.

        Parameters
        ----------
        solver: OptimizationSolver
            Optimization solver to use for solving the problem.
        solver_params: ty.Dict
            Parameters for the solver.
        fitness_fn: ty.Callable[[float, int], float]
            Fitness function to evaluate a given set of hyper-parameters,
            taking as input the current cost and number of steps to solution.
            This is the function that is maximized by the SolverTuner.
        fitness_target: float, optional
            Fitness target to reach. If this is not passed, the full grid is
            explored before stopping search.
        seed: int, default=0
            Seed for randomized grid search.

        Returns
        -------
        best_hyperparams: ty.Dict
            Dictionary containing the hyper-parameters with the highest fitness.
        success: bool
            Flag signaling if the fitness_target has been reached. If no
            fitness_target is passed, the flag is True.
        """
        # TODO : Check that hyperparams are arguments for solver

        best_hyperparams = None
        best_fitness = -float('inf')

        random.Random(seed).shuffle(self._params_grid)

        for params in self._params_grid:
            np.random.seed(seed)
            hyperparams = dict(zip(self._params_keys, params))
            solver_params["hyperparameters"] = hyperparams
            solver.solve(**solver_params)
            cost = solver.last_run_report["cost"]
            step_to_sol = solver.last_run_report["steps_to_solution"]
            fitness = fitness_fn(cost, step_to_sol)
            self._store_trial(hyperparams, cost, step_to_sol, fitness)
            if fitness > best_fitness:
                best_hyperparams = hyperparams
                best_fitness = fitness
                print(f"Better hyperparameters configuration found!\n"
                      f"Hyperparameters: {best_hyperparams}")
            if fitness_target is not None and best_fitness >= fitness_target:
                break

        success = best_fitness >= (fitness_target or -float('inf'))
        return best_hyperparams, success

    def _store_trial(self, params, cost, step_to_sol, fitness):
        new_entry = tuple(map(lambda k: params[k], self._params_keys))
        new_entry = new_entry + (cost, step_to_sol, fitness)
        new_entry_array = np.array([new_entry], dtype=self._store_dtype)
        self._store = np.concatenate([self._store, new_entry_array])

    def get_results(self):
        """Returns data on all hyper-parameters evaluations as a structured
        numpy array."""
        return self._store

    @staticmethod
    def _build_store_dtype(params_grid, keys):
        type_conv = {float: 'f4', int: 'i4'}
        dtype = list(
            map(lambda k: (k, type_conv[type(params_grid[k][0])]), keys))
        dtype += [('cost', 'f4'), ('step_to_sol', 'i4'), ('fitness', 'f4')]
        return dtype
