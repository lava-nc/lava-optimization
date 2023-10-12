# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


import numpy as np
from pprint import pprint
from dataclasses import dataclass

from lava.lib.optimization.problems.problems import QUBO
from lava.lib.optimization.solvers.generic.solver import OptimizationSolver, \
    SolverReport
from lava.lib.optimization.apps.clustering.problems import ClusteringProblem
from lava.lib.optimization.apps.clustering.utils.q_matrix_generator import \
    QMatrixClust

import typing as ty
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


@dataclass
class ClusteringConfig(SolverConfig):
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

    do_distance_sparsification: bool = False
    sparsification_algo: str = "cutoff"
    max_dist_cutoff_fraction: float = 1.0
    profile_q_mat_gen: bool = False
    only_gen_q_mat: bool = False


@dataclass
class ClusteringSolution:
    """Clustering solution holds two dictionaries:
        - `clustering_id_map` holds a map from cluster center ID to a list
        of point IDs
        - `clustering_coords_map` holds a map from the cluster center
        coordinates to the point coordinates
    """
    clustering_id_map: dict = None
    clustering_coords_map: dict = None


class ClusteringSolver:
    """Solver for clustering problems, given cluster centers.
    """
    def __init__(self, clp: ClusteringProblem):
        self.problem = clp
        self._solver = None
        self._profiler = None
        self.dist_sparsity = 0.
        self.dist_proxy_sparsity = 0.
        self.q_gen_time = 0.
        self.q_shape = None
        self.raw_solution = None
        self.solution = ClusteringSolution()

    @property
    def solver(self):
        return self._solver

    @property
    def profiler(self):
        return self._profiler

    def solve(self, scfg: ClusteringConfig = ClusteringConfig()):
        """
        Solve a clustering problem using a given solver configuration.

        Parameters
        ----------
        scfg (ClusteringConfig) : Configuration parameters.

        Notes
        -----
        The solver object also stores profiling data as its attributes.
        """
        # 1. Generate Q matrix for clustering
        node_list_for_clustering = self.problem.center_coords + \
            self.problem.point_coords
        # number of binary variables = total_num_nodes * num_clusters
        mat_size = len(node_list_for_clustering) * self.problem.num_clusters
        q_mat_obj = QMatrixClust(
            node_list_for_clustering,
            num_clusters=self.problem.num_clusters,
            lambda_dist=1,
            lambda_points=100,
            lambda_centers=100,
            fixed_pt=True,
            fixed_pt_range=(-128, 127),
            clust_dist_sparse_params={
                "do_sparse": scfg.do_distance_sparsification,
                "algo": scfg.sparsification_algo,
                "max_dist_cutoff_fraction": scfg.max_dist_cutoff_fraction},
            profile_mat_gen=scfg.profile_q_mat_gen)
        q_mat = q_mat_obj.matrix.astype(int)
        self.dist_sparsity = q_mat_obj.dist_sparsity
        self.dist_proxy_sparsity = q_mat_obj.dist_proxy_sparsity
        if scfg.profile_q_mat_gen:
            self.q_gen_time = q_mat_obj.time_to_gen_mat
            self.q_shape = q_mat.shape
        # 2. Call Lava QUBO solvers
        if not scfg.only_gen_q_mat:
            prob = QUBO(q=q_mat)
            self._solver = OptimizationSolver(problem=prob)
            hparams = {
                'neuron_model': 'nebm-sa-refract',
                'refract': 10,
                'refract_scaling': 6,
                'init_state': np.random.randint(0, 2, size=(mat_size,)),
                'min_temperature': 1,
                'max_temperature': 5,
                'steps_per_temperature': 200
            }
            if not scfg.hyperparameters:
                scfg.hyperparameters.update(hparams)
            report: SolverReport = self._solver.solve(config=scfg)
            if report.profiler:
                self._profiler = report.profiler
                pprint(f"Clustering execution"
                       f" took {np.sum(report.profiler.execution_time)}s")
            # 3. Post process the clustering solution
            self.raw_solution: npty.NDArray = \
                report.best_state.reshape((self.problem.num_clusters,
                                           len(node_list_for_clustering))).T
        else:
            self.raw_solution = -1 * np.ones((self.problem.num_clusters,
                                              len(node_list_for_clustering))).T

        self.post_process_sol()

    def post_process_sol(self):
        """
        Post-process the clustering solution returned by `solve()`.

        The clustering solution returned by the `solve` method is a 2-D
        binary numpy array, wherein the columns correspond to clusters and
        rows correspond to points or cluster centers. entry (i, j) is 1 if
        point/cluster center 'i' belongs to cluster 'j'.
        """

        coord_list = (self.problem.center_coords + self.problem.point_coords)
        id_map = {}
        coord_map = {}
        for j, col in enumerate(self.raw_solution.T):
            node_idxs = np.nonzero(col)
            # ID of "this" cluster is the only nonzero row in this column
            # from row 0 to row 'num_clusters' - 1
            this_cluster_id = \
                (node_idxs[0][node_idxs[0] < self.problem.num_clusters] + 1)
            if len(this_cluster_id) != 1:
                raise ValueError(f"More than one cluster center found in "
                                 f"{j}th cluster. Clustering might not have "
                                 f"converged to a valid solution.")
            node_idxs = node_idxs[0][node_idxs[0] >= self.problem.num_clusters]
            id_map.update({this_cluster_id.item(): (node_idxs + 1).tolist()})

            this_center_coords = np.array(coord_list)[this_cluster_id - 1, :]
            point_coords_this_cluster = np.array(coord_list)[node_idxs, :]
            point_coords_this_cluster = \
                [tuple(point) for point in point_coords_this_cluster.tolist()]
            coord_map.update({
                tuple(this_center_coords.flatten()): point_coords_this_cluster})

        self.solution.clustering_id_map = id_map
        self.solution.clustering_coords_map = coord_map
