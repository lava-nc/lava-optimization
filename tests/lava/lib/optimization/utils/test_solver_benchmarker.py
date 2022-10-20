# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

from lava.lib.optimization.utils.solver_benchmarker import ProtytypeOfSolverBenchmarker


class TestSolverBenchmarker(unittest.TestCase):
    # 1) test instantiation.
    # 2) test get_power_measurement_cfg is callable and accepts num_steps:
    # 3) test get_time_measurement_cfg is callable and accepts num_steps:
    # 4) test creation of pre_run_fxs, post_run_fxs for power
    # 5) test contents of pre_run_fxs, post_run_fxs (can join with prev)
    #    for power
    # 6) test creation of pre_run_fxs, post_run_fxs for time
    # 7) test contents of pre_run_fxs, post_run_fxs (can join with prev) for
    #    time
    # 8) test measured_power property exists and can be accessed
    # 9) test return of measured_power property (can join with prev)
    # 10) test measured_time property exists and can be accessed
    # 11) test return of measured_time property (can join with prev)
    
    def setUp(self) -> None:
        self.benchmarker = ProtytypeOfSolverBenchmarker()

    def test_create_obj(self):
        self.assertIsInstance(self.benchmarker, ProtytypeOfSolverBenchmarker)

    def test_power_measurement_cfg(self):
        num_steps = 5
        power_cfg = self.benchmarker.get_power_measurement_cfg(num_steps=num_steps)
        self.assertIsInstance(power_cfg[0], list)
        self.assertIsInstance(power_cfg[1], list)

    def test_time_measurement_cfg(self):
        num_steps = 5
        time_cfg = self.benchmarker.get_time_measurement_cfg(num_steps=num_steps)
        self.assertIsInstance(time_cfg[0], list)
        self.assertIsInstance(time_cfg[1], list)
        
    

if __name__ == '__main__':
    unittest.main()
