# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import enum
import numpy as np
import networkx as ntx
from pprint import pprint
from dataclasses import dataclass
try:
    from vrpy import VehicleRoutingProblem
except ImportError:
    class VehicleRoutingProblem:

        vrpy_not_installed: bool = True

        def __init__(self, graph):
            self.graph = graph

from lava.lib.optimization.problems.problems import QUBO
from lava.lib.optimization.solvers.generic.solver import OptimizationSolver, \
    SolverReport
from lava.lib.optimization.apps.vrp.problems import VRP
from lava.lib.optimization.apps.vrp.utils.q_matrix_generator import \
    ProblemType, QMatrixVRP

import typing as ty
from typing import Tuple, Dict, List
import numpy.typing as npty

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


class CoreSolver(enum.IntEnum):
    VRPY_CPU = 0
    LAVA_QUBO = 1


@dataclass
class VRPConfig(SolverConfig):
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

    core_solver: CoreSolver = CoreSolver.VRPY_CPU
    do_distance_sparsification: bool = False
    sparsification_algo: str = "cutoff"
    max_dist_cutoff_fraction: float = 1.0
    profile_q_mat_gen_clust: bool = False
    profile_q_mat_gen_tsp: bool = False
    only_gen_q_mats: bool = False
    only_cluster: bool = False


class VRPSolver:
    """Solver for vehicle routing problems.
    """
    def __init__(self, vrp: VRP):
        self.problem = vrp
        self.clust_solver = None
        self.clust_profiler = None
        self.tsp_profiler_list = []
        self.dist_sparsity = 0.
        self.dist_proxy_sparsity = 0.
        self.q_gen_time_clust = 0.
        self.q_gen_times_tsp = []
        self.clust_q_shape = None
        self.tsp_q_shapes = []

    def solve(self, scfg: VRPConfig = VRPConfig()) -> \
            Tuple[npty.NDArray, Dict[int, List[int]]]:
        """
        Solve a VRP using a given solver configuration.

        Parameters
        ----------
        scfg (VRPConfig) : Configuration parameters.

        Returns
        -------
        (numpy.ndarray) : An array of route-costs if the solver uses VRPy
        backend to solve a VRP. If QUBO solver backend is used,
        it is the binary clustering solution from the clustering stage of
        the QUBO solver.

        routes (Dict[int, List[int]]) : A dictionary whose keys are vehicle
        IDs and values are lists of the corresponding waypoint IDs to be
        visited.
        """
        def _prepare_graph_for_vrpy(g):
            demands = [1 for n in g.nodes]
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
            cost_src_to_veh = 0
            cost_src_to_nod = 2 ** 16
            cost_veh_to_snk = 2 ** 16
            cost_nod_to_snk = 0
            for n in g.nodes:
                if g.nodes[n]["Type"] == "Vehicle":
                    g.add_edge("Source", n, cost=cost_src_to_veh)
                    g.add_edge(n, "Sink", cost=cost_veh_to_snk)
                if g.nodes[n]["Type"] == "Node":
                    g.add_edge("Source", n, cost=cost_src_to_nod)
                    g.add_edge(n, "Sink", cost=cost_nod_to_snk)
            return g
        if scfg.core_solver == CoreSolver.VRPY_CPU:
            vrpy_is_installed = not hasattr(VehicleRoutingProblem,
                                            "vrpy_not_installed")
            if not vrpy_is_installed:
                raise ImportError("VRPy is not installed.")
            # 1. Prepare problem for VRPy
            graph_to_solve = self.problem.problem_graph.copy()
            graph_to_solve = _prepare_graph_for_vrpy(graph_to_solve)

            # 2. Call VRPy.solve
            load_cap = self.problem.num_nodes * self.problem.num_vehicles
            vrpy_sol = \
                VehicleRoutingProblem(graph_to_solve,
                                      load_capacity=load_cap,
                                      num_vehicles=self.problem.num_vehicles,
                                      use_all_vehicles=True)
            vrpy_sol.solve(max_iter=1000)

            # 3. Post process the solution
            best_routes = vrpy_sol.best_routes
            routes = {}
            for rt_id, route in best_routes.items():
                route.remove("Source")
                route.remove("Sink")
                routes.update({route[0]: route[1:]})
            if scfg.probe_cost:
                print(f"Best value: {vrpy_sol.best_value}\t Best route "
                      f"costs: {vrpy_sol.best_routes_cost}")

            # 4. Return the list of Node IDs
            return np.array(list(vrpy_sol.best_routes_cost.items())), routes
        elif scfg.core_solver == CoreSolver.LAVA_QUBO:
            # 1. Generate Q matrix for clustering
            node_list_for_clustering = self.problem.vehicle_init_coords + \
                self.problem.node_coords
            # number of binary variables = total_num_nodes * num_clusters
            mat_size = len(node_list_for_clustering) * self.problem.num_vehicles
            q_clust_obj = QMatrixVRP(
                node_list_for_clustering,
                num_vehicles=self.problem.num_vehicles,
                problem_type=ProblemType.CLUSTER,
                mat_size_for_random=mat_size,
                lamda_dist=1,
                lamda_wypts=100,
                lamda_vhcles=100,
                lamda_cnstrt=1,
                fixed_pt=True,
                fixed_pt_range=(-128, 127),
                clust_dist_sparse_params={
                    "do_sparse": scfg.do_distance_sparsification,
                    "algo": scfg.sparsification_algo,
                    "max_dist_cutoff_fraction": scfg.max_dist_cutoff_fraction},
                profile_mat_gen=scfg.profile_q_mat_gen_clust)
            Q_clust = q_clust_obj.matrix.astype(int)
            self.dist_sparsity = q_clust_obj.dist_sparsity
            self.dist_proxy_sparsity = q_clust_obj.dist_proxy_sparsity
            if scfg.profile_q_mat_gen_clust:
                self.q_gen_time_clust = q_clust_obj.time_clust_mat
                self.clust_q_shape = Q_clust.shape
            # 2. Call Lava QUBO solvers
            if not scfg.only_gen_q_mats:
                prob = QUBO(q=Q_clust)
                self.clust_solver = OptimizationSolver(problem=prob)
                init_value = np.random.randint(0, 2, size=(mat_size,))
                scfg.hyperparameters.update({
                    'refract': np.random.randint(1, int(np.sqrt(mat_size)) + 4,
                                                 size=(mat_size,)),
                    'init_value': init_value,
                    'temperature': 1
                })
                report: SolverReport = self.clust_solver.solve(config=scfg)
                if report.profiler:
                    self.clust_profiler = report.profiler
                    pprint(f"Clustering execution"
                           f" took {np.sum(report.profiler.execution_time)}s")
                # 3. Post process the clustering solution
                clustering_solution: npty.NDArray = \
                    report.best_state.reshape((self.problem.num_vehicles,
                                               len(node_list_for_clustering))).T
            else:
                clustering_solution = np.zeros(shape=(
                    self.problem.num_nodes + self.problem.num_vehicles,
                    self.problem.num_vehicles))
                clustering_solution[0:self.problem.num_vehicles,
                                    0:self.problem.num_vehicles] = \
                    np.eye(self.problem.num_vehicles)
                clustering_solution[self.problem.num_vehicles:, :] = \
                    np.random.randint(0, 2, size=(
                        self.problem.num_nodes, self.problem.num_vehicles))
            # 4. In a loop, generate Q matrices for TSPs
            tsp_routes = {}
            if not scfg.only_cluster:
                for j, col in enumerate(clustering_solution.T):
                    # number of binary variables = num nodes * num steps
                    matsize = np.count_nonzero(col[self.problem.num_vehicles:])**2
                    node_idxs = np.nonzero(col)
                    vehicle_id_this_cluster = node_idxs[0][
                        node_idxs[0] < self.problem.num_vehicles] + 1
                    node_idxs = node_idxs[0][
                        node_idxs[0] >= self.problem.num_vehicles]
                    nodes_to_pass = np.array(node_list_for_clustering)[node_idxs, :]
                    nodes_to_pass = [tuple(node) for node in nodes_to_pass.tolist()]
                    q_vrp_obj = QMatrixVRP(
                        nodes_to_pass,
                        num_vehicles=1,
                        problem_type=ProblemType.TSP,
                        mat_size_for_random=matsize,
                        lamda_dist=1,
                        lamda_wypts=1,
                        lamda_vhcles=1,
                        lamda_cnstrt=100,
                        fixed_pt=True,
                        fixed_pt_range=(-128, 127),
                        profile_mat_gen=scfg.profile_q_mat_gen_tsp
                    )
                    Q_VRP = q_vrp_obj.matrix.astype(int)
                    if scfg.profile_q_mat_gen_tsp:
                        self.q_gen_times_tsp.append(q_vrp_obj.time_tsp_mat)
                        self.tsp_q_shapes.append(Q_VRP.shape)
                    if not scfg.only_gen_q_mats:
                        tsp = QUBO(q=Q_VRP)
                        tsp_solver = OptimizationSolver(problem=tsp)
                        scfg.hyperparameters.update({
                            'refract': np.random.randint(0, int(np.sqrt(matsize)),
                                                         size=(matsize,)),
                            'init_value': np.random.randint(0, 2, size=(matsize,))
                        })
                        report: SolverReport = tsp_solver.solve(config=scfg)
                        if report.profiler:
                            self.tsp_profiler_list.append(report.profiler)
                            pprint(f"TSP {j} execution took"
                                   f" {np.sum(report.profiler.execution_time)}s")
                        solution: npty.NDArray = \
                            report.best_state.reshape((len(nodes_to_pass),
                                                       len(nodes_to_pass))).T
                        node_idxs_2 = np.nonzero(solution)
                        node_idxs_2 = list(zip(node_idxs_2[0].tolist(),
                                               node_idxs_2[1].tolist()))
                        node_idxs_2.sort(key=lambda x: x[1])

                        route = {
                            vehicle_id_this_cluster.item(): [(node_idxs[node_id[
                                0]] + 1).tolist() for node_id in node_idxs_2]}
                        tsp_routes.update(route)
            return clustering_solution, tsp_routes
        else:
            raise ValueError("Incorrect core solver specified or VRPy is not "
                             "installed.")
