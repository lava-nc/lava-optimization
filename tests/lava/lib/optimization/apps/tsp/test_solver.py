import os
import pprint
import unittest

import numpy as np
import networkx as ntx
from lava.lib.optimization.apps.tsp.problems import TravellingSalesmanProblem
from lava.lib.optimization.apps.tsp.solver import TSPConfig, TSPSolver


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


class TestTSPSolver(unittest.TestCase):
    def setUp(self) -> None:
        all_coords = [(1, 1), (2, 1), (16, 1), (16, 15), (2, 15)]
        self.center_coords = all_coords[0]
        self.point_coords = all_coords[1:]
        self.tsp_instance = TravellingSalesmanProblem(
            waypt_coords=self.point_coords, starting_pt=self.center_coords)

    def test_init(self):
        solver = TSPSolver(tsp=self.tsp_instance)
        self.assertIsInstance(solver, TSPSolver)

    @unittest.skipUnless(run_loihi_tests and run_lib_tests, skip_reason)
    def test_solve_lava_qubo(self):
        solver = TSPSolver(tsp=self.tsp_instance)
        scfg = TSPConfig(backend="Loihi2",
                         hyperparameters={},
                         target_cost=-1000000,
                         timeout=1000,
                         probe_time=False,
                         log_level=40)
        np.random.seed(0)
        solver.solve(scfg=scfg)
        gt_indices = np.array([2, 3, 4, 5])
        spidx = solver.solution.solution_path_ids
        self.assertEqual(gt_indices.size, len(spidx))

        gt_graph = ntx.Graph([(2, 3), (3, 4), (4, 5), (5, 2)])
        sol_graph = ntx.Graph([(spidx[0], spidx[1]),
                               (spidx[1], spidx[2]),
                               (spidx[2], spidx[3]),
                               (spidx[3], spidx[0])])

        self.assertTrue(ntx.utils.graphs_equal(gt_graph, sol_graph))


if __name__ == '__main__':
    unittest.main()
