# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import enum
from pprint import pprint

import numpy as np
import networkx as ntx
import typing as ty
from typing import Tuple, Dict, List
import numpy.typing as npty
from dataclasses import dataclass

from lava.lib.optimization.problems.problems import QUBO
from lava.lib.optimization.solvers.generic.solver import OptimizationSolver, \
    SolverReport
from lava.lib.optimization.apps.tsp.problems import TravellingSalesmanProblem
from lava.lib.optimization.apps.tsp.utils.q_matrix_generator import QMatrixTSP

from lava.magma.core.resources import (
    CPU,
    Loihi2NeuroCore,
    NeuroCore,
)
from lava.lib.optimization.solvers.generic.solver import SolverConfig

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


@dataclass
class TSPConfig(SolverConfig):
    """Solver configuration for VRP solver.

    Parameters
    ----------
    core_solver : CoreSolver
        Core algorithm that solves a given VRP. Possible values are
        CoreSolver.VRPY_CPU or CoreSolver.LAVA_QUBO.

    Notes
    -----
    VRPConfig class inherits from `SolverConfig` class at
    `lava.lib.optimization.solvers.generic.solver`. Please refer to the
    documentation for `SolverConfig` to know more about other arguments that
    can be passed.
    """

    profile_q_mat_gen: bool = False
    only_gen_q_mat: bool = False


@dataclass
class TSPSolution:
    """TSP solution holds two lists:
        - `solution_path_ids` holds the ordered list of IDs of the waypoints,
        which forms the path obtained from the QUBO solution. It begins and
        ends with `1`, the ID of the salesman node.
        - `solution_path_coords` holds the ordered list of tuples, which are
        the coordinates of the waypoints.
    """
    solution_path_ids: list = None
    solution_path_coords: list = None


class TSPSolver:
    """Solver for vehicle routing problems.
    """
    def __init__(self, tsp: TravellingSalesmanProblem):
        self.problem = tsp
        self._solver = None
        self._profiler = None
        self.q_gen_time = 0.
        self.raw_solution = None
        self.solution = TSPSolution()

    @property
    def solver(self):
        return self._solver

    @property
    def profiler(self):
        return self._profiler

    def solve(self, scfg: TSPConfig = TSPConfig()):
        """
        Solve a TSP using a given solver configuration.

        Parameters
        ----------
        scfg (TSPConfig) : Configuration parameters.

        """
        q_matsize = self.problem.num_waypts ** 2
        q_mat_obj = QMatrixTSP(
            self.problem.waypt_coords,
            lamda_dist=1,
            lamda_cnstrt=100,
            fixed_pt=False,
            fixed_pt_range=(-128, 127),
            profile_mat_gen=scfg.profile_q_mat_gen
        )
        q_mat = q_mat_obj.matrix.astype(int)
        if scfg.profile_q_mat_gen:
            self.q_gen_time = q_mat_obj.time_to_gen_mat
        if not scfg.only_gen_q_mat:
            tsp = QUBO(q=q_mat)
            tsp_solver = OptimizationSolver(problem=tsp)
            hparams = {
                'neuron_model': 'nebm-sa-refract',
                'refract': 50,
                'refract_scaling': 6,
                'init_state': np.random.randint(0, 2, size=(q_matsize,)),
                'min_temperature': 1,
                'max_temperature': 5,
                'steps_per_temperature': 200
            }
            if not scfg.hyperparameters:
                scfg.hyperparameters.update(hparams)
            report: SolverReport = tsp_solver.solve(config=scfg)
            if report.profiler:
                self._profiler = report.profiler
                pprint(f"TSP execution took"
                       f" {np.sum(report.profiler.execution_time)}s")
            self.raw_solution: npty.NDArray = \
                report.best_state.reshape((self.problem.num_waypts,
                                           self.problem.num_waypts)).T
        else:
            self.raw_solution = -1 * np.ones((self.problem.num_waypts,
                                              self.problem.num_waypts)).T

        self.post_process_sol()

    def post_process_sol(self):
        ordered_indices = np.nonzero(self.raw_solution)
        ordered_indices = list(zip(ordered_indices[0].tolist(),
                                   ordered_indices[1].tolist()))
        ordered_indices.sort(key=lambda x: x[1])

        self.solution.solution_path_ids = [
            self.problem.waypt_ids[node_id[0]] for node_id in ordered_indices]
        self.solution.solution_path_coords = [
            self.problem.waypt_coords[node_id[0]] for node_id in
            ordered_indices]
