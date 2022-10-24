# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import numpy as np
import typing as ty

from lava.lib.optimization.problems.problems import QUBO
from lava.lib.optimization.solvers.generic.solver import OptimizationSolver


class SolverTuner:
    """Class to find and set hyperparameters for an OptimizationSolver."""

    def __init__(self,
                 step_range: ty.Tuple = (1, 2),
                 noise_range: ty.Tuple = (4, 5),
                 steps_to_fire_range: ty.Tuple = (8, 9)
                 ):
        self._step_range = step_range
        self._noise_range = noise_range
        self._steps_to_fire_range = steps_to_fire_range

    def tune(self,
             solver: OptimizationSolver,
             solver_parameters: ty.Dict,
             target_cost: int):
        """Find and set solving hyperparameters for an OptimizationSolver"""
        hyperparameters = self._perform_grid_search(solver,
                                                    solver_parameters,
                                                    target_cost=target_cost)
        succeeded = hyperparameters is not None
        if succeeded:
            solver.hyperparameters = hyperparameters
        return solver, succeeded

    def _perform_grid_search(self,
                             solver: OptimizationSolver,
                             solver_parameters: ty.Dict,
                             target_cost: int = None,
                             ) -> ty.Union[ty.NoReturn, ty.Dict]:
        """Explore hyperparameter space until a solution is found.

        # Todo: allow finding best configuration by tracking step to solution.
        """
        cost = None
        if not isinstance(solver.problem, QUBO):
            raise ValueError("SolverTuner can only be used with QUBO problems.")
        qubo_problem: QUBO = solver.problem
        for steps_to_fire in self._step_range:
            for noise_amplitude in self._noise_range:
                for step_size in self._steps_to_fire_range:
                    hyperparameters = dict(steps_to_fire=steps_to_fire,
                                           noise_amplitude=noise_amplitude,
                                           step_size=step_size
                                           )
                    solver_parameters["hyperparameters"] = hyperparameters
                    solution = solver.solve(**solver_parameters)
                    cost = qubo_problem.compute_cost(solution)
                    self._print_cost_msg(cost, target_cost, solution)
                    if cost is not None and cost <= target_cost:
                        self._print_solution_found_msg(hyperparameters)
                        break
                else:
                    continue
                break
            else:
                continue
            break

        if cost is None or cost > target_cost:
            hyperparameters = None
        return hyperparameters

    @staticmethod
    def _print_cost_msg(cost, cost_ref, solution) -> None:
        print(f"""Solution vector from Loihi {solution} \
          \nNodes in MIS (index starts at 0): {np.where(solution)[0]} \
          \nQUBO cost of solution: {cost} (Lava) vs {cost_ref,} (target)\n""")

    @staticmethod
    def _print_solution_found_msg(hyperparameters: ty.Dict) -> None:
        print("Solution found!")
        print(f"{hyperparameters=}")
