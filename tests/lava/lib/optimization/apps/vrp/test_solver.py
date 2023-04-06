import os
import unittest

import numpy as np

from lava.lib.optimization.apps.vrp.problems import VRP
from lava.lib.optimization.apps.vrp.solver import VRPConfig, VRPSolver, \
    CoreSolver


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


class testVRPSolver(unittest.TestCase):
    def setUp(self) -> None:
        all_coords = [(1, 1), (23, 23), (2, 1), (3, 2), (1, 3),
                      (4, 2), (22, 21), (20, 23), (21, 21), (24, 25)]
        self.vehicle_coords = all_coords[:2]
        self.node_coords = all_coords[2:]
        self.edges = [(1, 3), (2, 9), (3, 5), (3, 4), (4, 6), (6, 5),
                      (9, 7), (8, 10), (7, 10), (10, 8), (7, 9)]
        self.vrp_instance_natural_edges = VRP(
            node_coords=self.node_coords, vehicle_coords=self.vehicle_coords)
        self.vrp_instance_user_edges = VRP(
            node_coords=self.node_coords, vehicle_coords=self.vehicle_coords,
            edges=self.edges)

    def test_init(self):
        solver = VRPSolver(vrp=self.vrp_instance_natural_edges)
        self.assertIsInstance(solver, VRPSolver)

    def test_solve_vrpy(self):
        solver = VRPSolver(vrp=self.vrp_instance_natural_edges)
        costs, result = solver.solve()
        gt = {1: [3, 4, 5, 6], 2: [7, 8, 9, 10]}
        self.assertSetEqual(set(gt.keys()), set(result.keys()))
        for vehicle_id, route in gt.items():
            self.assertSetEqual(set(route), set(result[vehicle_id]))

    @unittest.skipUnless(run_loihi_tests, "")
    def test_solve_lava_qubo(self):
        solver = VRPSolver(vrp=self.vrp_instance_natural_edges)
        scfg = VRPConfig(backend="Loihi2",
                         core_solver=CoreSolver.LAVA_QUBO,
                         hyperparameters={},
                         target_cost=-1000000,
                         timeout=10000,
                         probe_time=False,
                         log_level=40)
        np.random.seed(42313)
        clusters, routes = solver.solve(scfg=scfg)
        gt = {1: [3, 4, 5, 6], 2: [7, 8, 9, 10]}
        self.assertSetEqual(set(gt.keys()), set(routes.keys()))
        for vehicle_id, route in gt.items():
            self.assertSetEqual(set(route), set(routes[vehicle_id]))


if __name__ == '__main__':
    unittest.main()
