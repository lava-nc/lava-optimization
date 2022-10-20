# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import typing as ty

import numpy as np

from lava.lib.optimization.utils.solver_benchmarker import ProtytypeOfSolverBenchmarker


class TestSolverBenchmarker(unittest.TestCase):
    """
    Unit test suite for SolverBenchmarker.
    """

    def setUp(self) -> None:
        self.benchmarker = ProtytypeOfSolverBenchmarker()

    def test_create_obj(self):
        """    # 1) test instantiation."""
        self.assertIsInstance(self.benchmarker, ProtytypeOfSolverBenchmarker)

    def test_power_measurement_cfg(self):
        """    # 2) test get_power_measurement_cfg is callable and accepts num_steps:"""
        num_steps = 5
        self.power_cfg = self.benchmarker.get_power_measurement_cfg(
            num_steps=num_steps)
        self.assertIsNotNone(self.power_cfg)

    def test_time_measurement_cfg(self):
        """    # 3) test get_time_measurement_cfg is callable and accepts num_steps:"""
        num_steps = 5
        self.time_cfg = self.benchmarker.get_time_measurement_cfg(
            num_steps=num_steps)
        self.assertIsNotNone(self.time_cfg)

    def test_power_cfg(self):
        """    # 4) test creation of pre_run_fxs, post_run_fxs for power
               # 5) test contents of pre_run_fxs, post_run_fxs (can join with prev) for power"""
        self.assertIsInstance(self.power_cfg[0][0], ty.Callable)

    def test_time_cfg(self):
        """    # 6) test creation of pre_run_fxs, post_run_fxs for time
               # 7) test contents of pre_run_fxs, post_run_fxs (can join with prev) for time"""
        self.assertIsInstance(self.time_cfg[0][0], ty.Callable)

    def test_measured_power_property(self):
        """    # 8) test measured_power property exists and can be accessed
               # 9) test return of measured_power property (can join with prev)"""
        measured_power = self.benchmarker.measured_power
        self.assertIsInstance(measured_power, np.ndarray)

    def test_measured_time_property(self):
        """    # 10) test measured_time property exists and can be accessed
               # 11) test return of measured_time property (can join with prev)"""
        measured_time = self.benchmarker.measured_time
        self.assertIsInstance(measured_time, np.ndarray)


if __name__ == '__main__':
    unittest.main()
