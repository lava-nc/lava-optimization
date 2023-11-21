import os
import unittest

import numpy as np

from lava.lib.optimization.apps.clustering.problems import ClusteringProblem
from lava.lib.optimization.apps.clustering.solver import (ClusteringConfig,
                                                          ClusteringSolver)


def get_bool_env_setting(env_var: str):
    """Get an environment varible and return
    True if the variable is set to 1 else return
    false
    """
    env_test_setting = os.environ.get(env_var)
    test_setting = False
    if env_test_setting == "1":
        test_setting = True
    return test_setting


run_loihi_tests: bool = get_bool_env_setting("RUN_LOIHI_TESTS")
run_lib_tests: bool = get_bool_env_setting("RUN_LIB_TESTS")
skip_reason = "Either Loihi or Lib or both tests are disabled."


class TestClusteringSolver(unittest.TestCase):
    def setUp(self) -> None:
        all_coords = [(1, 1), (23, 23), (2, 1), (3, 2), (1, 3),
                      (4, 2), (22, 21), (20, 23), (21, 21), (24, 25)]
        self.center_coords = all_coords[:2]
        self.point_coords = all_coords[2:]
        self.edges = [(1, 3), (2, 9), (3, 5), (3, 4), (4, 6), (6, 5),
                      (9, 7), (8, 10), (7, 10), (10, 8), (7, 9)]
        self.clustering_prob_instance = ClusteringProblem(
            point_coords=self.point_coords, center_coords=self.center_coords)

    def test_init(self):
        solver = ClusteringSolver(clp=self.clustering_prob_instance)
        self.assertIsInstance(solver, ClusteringSolver)

    @unittest.skipUnless(run_loihi_tests and run_lib_tests, skip_reason)
    def test_solve_lava_qubo(self):
        solver = ClusteringSolver(clp=self.clustering_prob_instance)
        scfg = ClusteringConfig(backend="Loihi2",
                                hyperparameters={},
                                target_cost=-1000000,
                                timeout=1000,
                                probe_time=False,
                                log_level=40)
        np.random.seed(0)
        solver.solve(scfg=scfg)
        gt_id_map = {1: [3, 4, 5, 6], 2: [7, 8, 9, 10]}
        self.assertSetEqual(set(gt_id_map.keys()),
                            set(solver.solution.clustering_id_map.keys()))
        for cluster_id, cluster in gt_id_map.items():
            self.assertSetEqual(
                set(cluster),
                set(solver.solution.clustering_id_map[cluster_id]))
        gt_coord_map = {self.center_coords[0]: self.point_coords[:4],
                        self.center_coords[1]: self.point_coords[4:]}
        self.assertSetEqual(set(gt_coord_map.keys()),
                            set(solver.solution.clustering_coords_map.keys()))
        for center_coords, points in gt_coord_map.items():
            self.assertSetEqual(
                set(points),
                set(solver.solution.clustering_coords_map[center_coords]))


if __name__ == '__main__':
    unittest.main()
