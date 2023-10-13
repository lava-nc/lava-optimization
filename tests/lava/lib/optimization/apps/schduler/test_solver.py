# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import pprint
import unittest
import os

import numpy as np

from lava.lib.optimization.apps.scheduler.problems import (
    SchedulingProblem, SatelliteScheduleProblem)
from lava.lib.optimization.apps.scheduler.solver import (Scheduler,
                                                         SatelliteScheduler)


def get_bool_env_setting(env_var: str):
    """Get an environment variable and return True if the variable is set to
    1 else return false.
    """
    env_test_setting = os.environ.get(env_var)
    test_setting = False
    if env_test_setting == "1":
        test_setting = True
    return test_setting


run_loihi_tests: bool = get_bool_env_setting("RUN_LOIHI_TESTS")
run_lib_tests: bool = get_bool_env_setting("RUN_LIB_TESTS")
skip_reason = "Either Loihi tests or Lib tests or both are not enabled."


class TestScheduler(unittest.TestCase):
    def setUp(self) -> None:
        self.sp = SchedulingProblem(num_agents=3, num_tasks=3)
        self.sp.generate(seed=42)
        self.scheduler = Scheduler(sp=self.sp, qubo_weights=(4, 20))

    def test_init(self):
        self.assertIsInstance(self.scheduler, Scheduler)  # add assertion here

    @unittest.skipUnless(run_lib_tests and run_loihi_tests, skip_reason)
    def test_netx_solver(self):
        self.scheduler.solve_with_netx()
        gt_sol = np.array([[0., 0., 0., 0.],
                           [5., 1., 2., 2.],
                           [7., 2., 1., 1.]])
        self.assertTrue(np.all(self.scheduler.netx_solution == gt_sol))

    @unittest.skipUnless(run_lib_tests and run_loihi_tests, skip_reason)
    def test_lava_solver(self):
        self.scheduler.solve_with_lava_qubo()
        gt_possible_node_ids = [[0, 4, 8], [0, 5, 7],
                                [1, 3, 8], [1, 5, 6],
                                [2, 3, 7], [2, 4, 6]]
        self.assertTrue(self.scheduler.lava_solution[:, 0].tolist() in
                        gt_possible_node_ids)


class TestSatelliteScheduler(unittest.TestCase):
    def setUp(self) -> None:
        requests = np.array(
            [[0.02058449, 0.96990985], [0.05808361, 0.86617615],
             [0.15601864, 0.15599452], [0.18182497, 0.18340451],
             [0.29214465, 0.36636184], [0.30424224, 0.52475643],
             [0.37454012, 0.95071431], [0.43194502, 0.29122914],
             [0.60111501, 0.70807258], [0.61185289, 0.13949386],
             [0.73199394, 0.59865848], [0.83244264, 0.21233911]]
        )
        self.ssp = SatelliteScheduleProblem(num_satellites=3,
                                            num_requests=12,
                                            requests=requests)
        self.ssp.generate(seed=42)
        self.sat_scheduler = SatelliteScheduler(ssp=self.ssp,
                                                qubo_weights=(4, 20))
        self.gt_sol = np.array([[0., 1., 0.30424224, 0.52475643],
                                [1., 1., 0.73199394, 0.59865848],
                                [2., 2., 0.02058449, 0.96990985],
                                [3., 2., 0.37454012, 0.95071431]])

    def test_init(self):
        self.assertIsInstance(self.sat_scheduler, SatelliteScheduler)

    @unittest.skipUnless(run_lib_tests and run_loihi_tests, skip_reason)
    def test_netx_solver(self):
        self.sat_scheduler.solve_with_netx()
        self.assertTrue(np.all(self.sat_scheduler.netx_solution == self.gt_sol))

    @unittest.skipUnless(run_lib_tests and run_loihi_tests, skip_reason)
    def test_lava_solver(self):
        self.sat_scheduler.solve_with_lava_qubo()
        self.assertTrue(np.all(self.sat_scheduler.lava_solution == self.gt_sol))


if __name__ == '__main__':
    unittest.main()
