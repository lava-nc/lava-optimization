# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
from lava.lib.optimization.apps.vrp.problems import VRP

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


class VRPConfig(SolverConfig):
    """Solver configuration for VRP solver.

    Parameters
    ----------
    core_solver : str
        Core algorithm that solves a given VRP. Possible values are
        'vrpy-cpu', 'lava-qubo-cpu', or 'lava-qubo-loihi2'"
    """

    core_solver: str = "vrpy-cpu"


class VRPSolver:
    """Solver for vehicle routing problems.
    """
    def __init__(self, vrp: VRP):
        self.problem = vrp

    def solve(self, scfg: VRPConfig = VRPConfig()) -> ty.Dict[int, ty.List[
     int]]:
        if scfg.core_solver == "vrpy-cpu":
            # 1. Prepare problem for VRPy
            graph_to_solve = self.problem.problem_graph
            # 2. Call VRPy.solve
            from vrpy import VehicleRoutingProblem
            vrpy_sol = VehicleRoutingProblem(graph_to_solve)
            vrpy_sol.solve()
            # 3. Post process the solution
            routes = vrpy_sol.best_routes
            # 4. Return the list of Node IDs
            return routes
        elif "lava-qubo" in scfg.core_solver:
            # 1. Generate Q matrix for clustering
            # 2. Call Lava QUBO solvers
            # 3. Post process the clustering solution
            # 4. In a loop, generate Q matrices for TSPs
            # 5. Call parallel instances of Lava QUBO solvers
            raise NotImplementedError
        else:
            raise ValueError("Incorrect core solver. Should be one of "
                             "'vrpy-cpu', 'lava-qubo-cpu', or "
                             "'lava-qubo-loihi2'")
