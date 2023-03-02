# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
from abc import abstractmethod

import numpy as np
import networkx as ntx
from pprint import pprint
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
        def _prepare_graph_for_vrpy(g):
            demands = [1 if g.nodes[n]["Type"] == "Node" else 0 for n in
                       g.nodes]
            ntx.set_node_attributes(g, dict(zip(g.nodes, demands)),
                                    name="demand")
            # Add Source and Sink nodes with Type as "Dummy" and same
            # Coordinates attribute.
            g.add_node("Source")
            g.add_node("Sink")
            ntx.set_node_attributes(g, {"Source": (0, 0), "Sink": (0, 0)},
                                    name="Coordinates")
            ntx.set_node_attributes(g, {"Source": "Dummy", "Sink": "Dummy"},
                                    name="Type")
            # Remove outward edges from "Sink", add inward edges to "Sink",
            # and assign costs to the new edges.
            for n in g.nodes:
                if g.nodes[n]["Type"] != "Dummy":
                    cost = 0 if g.nodes[n]["Type"] == "Vehicle" else 2**16
                    g.add_edge("Source", n, cost=cost)
                    g.add_edge(n, "Sink", cost=cost)
            return g
        if scfg.core_solver == "vrpy-cpu":
            # Assume that call vehicles start from the same depot location,
            # specifically for VRPy. Other solvers might not suffer from this
            # restriction.

            # 1. Prepare problem for VRPy
            graph_to_solve = self.problem.problem_graph
            graph_to_solve = _prepare_graph_for_vrpy(graph_to_solve)

            # 2. Call VRPy.solve
            from vrpy import VehicleRoutingProblem
            vrpy_sol = VehicleRoutingProblem(
                graph_to_solve,
                load_capacity=
                self.problem.num_nodes * self.problem.num_vehicles,
                num_vehicles=self.problem.num_vehicles,
                use_all_vehicles=True
            )
            vrpy_sol.solve(max_iter=1000)

            # 3. Post process the solution
            routes = vrpy_sol.best_routes
            for route in routes.values():
                route.remove("Source")
                route.remove("Sink")
            print(f"Best value: {vrpy_sol.best_value}\t Best route "
                  f"costs: {vrpy_sol.best_routes_cost}")

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
