# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import itertools as it
import typing as ty

import random
import numpy as np


class SolverTuner:
    """Utility class to optimize hyper-parameters by random search."""

    def __init__(self,
                 search_space: list,
                 params_names: list,
                 shuffle: bool = False,
                 seed: int = 0):
        """
        Instantiate a SolverTuner, defining a search space and a list of
        hyperparameters names.

        Parameters
        ----------
        search_space
            List of hyperparameter tuple to evaluate.
        params_names
            List of hyperparameters names, one for each tuple dimension.
        shuffle
            Boolean flag to control search space shuffling. If set to False,
            the order of search_space list is preserved, and can be useful to
            prioritize the evaluation of certain hyperparamters tuples.
        seed
            Seed for random shuffling and numpy seeding.
        """
        self._search_space = list(search_space)
        self._params_names = params_names
        self._shuffle = shuffle
        self._seed = seed

        if self._shuffle:
            random.Random(seed).shuffle(self._search_space)

        self._store_dtype = SolverTuner._build_store_dtype(
            self._search_space, self._params_names
        )
        self._store = np.zeros(
            len(self._search_space), dtype=self._store_dtype)

    def tune(self,
             solver,
             solver_params: ty.Dict,
             fitness_fn: ty.Callable[[float, int], float],
             fitness_target: float = None,
             ):
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

        Returns
        -------
        best_hyperparams: ty.Dict
            Dictionary containing the hyper-parameters with the highest fitness.
        success: bool
            Flag signaling if the fitness_target has been reached. If no
            fitness_target is passed, the flag is True.
        """
        # TODO : Check that hyperparams are arguments for solver
        self._stored_rows = 0
        if self._store.shape[0] < len(self._search_space):
            self._store = np.zeros(
                len(self._search_space), dtype=self._store_dtype)

        best_hyperparams = None
        best_fitness = -float("inf")

        for params in self._search_space:
            np.random.seed(self._seed)
            hyperparams = dict(zip(self._params_names, params))
            solver_params["hyperparameters"] = hyperparams
            solver.solve(**solver_params)
            cost = solver.last_run_report["cost"]
            step_to_sol = solver.last_run_report["steps_to_solution"]
            fitness = fitness_fn(cost, step_to_sol)
            self._store_trial(hyperparams, cost, step_to_sol, fitness)
            if fitness > best_fitness:
                best_hyperparams = hyperparams
                best_fitness = fitness
                print(
                    f"Better hyperparameters configuration found!\n"
                    f"Hyperparameters: {best_hyperparams}"
                )
            if fitness_target is not None and best_fitness >= fitness_target:
                break
        self._remove_unused_store()
        success = best_fitness >= (fitness_target or -float("inf"))
        return best_hyperparams, success

    def _store_trial(self, params, cost, step_to_sol, fitness):
        new_entry = tuple(map(lambda k: params[k], self._params_names))
        new_entry = new_entry + (cost, step_to_sol, fitness)
        self._store[self._stored_rows] = new_entry
        self._stored_rows += 1

    def _remove_unused_store(self):
        self._store = self._store[:self._stored_rows]

    @property
    def results(self):
        """Returns data on all hyper-parameters evaluations as a structured
        numpy array."""
        return self._store

    @staticmethod
    def _build_store_dtype(search_space, params_names) -> list:
        type_conv = {float: "f4", int: "i4"}
        dtype = []
        for i, name in enumerate(params_names):
            param_type = type(search_space[0][i])
            if type_conv.get(param_type) is None:
                raise ValueError(f"Search space must contain only "
                                 f"{type_conv.keys()}, passed {param_type}")
            dtype.append((name, type_conv[param_type]))
        dtype += [("cost", "f4"), ("step_to_sol", "i4"), ("fitness", "f4")]
        return dtype

    @staticmethod
    def generate_grid(params_domains: dict):
        params_names = list(params_domains.keys())
        search_space = list(it.product(
            *map(lambda k: params_domains[k], params_names)
        ))
        return search_space, params_names

    @property
    def search_space(self) -> list:
        return self._search_space

    @property
    def params_names(self) -> list:
        return self._params_names

    @property
    def shuffle(self) -> bool:
        return self._shuffle

    @property
    def seed(self) -> int:
        return self._seed

    @shuffle.setter
    def shuffle(self, shuffle: bool) -> None:
        self._shuffle = shuffle
        if self._shuffle:
            random.Random(self._seed).shuffle(self._search_space)

    @seed.setter
    def seed(self, seed: int) -> None:
        self._seed = seed
        if self._shuffle:
            random.Random(self._seed).shuffle(self._search_space)

    @search_space.setter
    def search_space(self, search_space: list) -> None:
        """Provide a new list of hyper-parameters tuples to the SolverTuner,
        with the same order defined by params_names.
        """
        self._search_space = list(search_space)
        if self._shuffle:
            random.Random(self._seed).shuffle(self._search_space)
