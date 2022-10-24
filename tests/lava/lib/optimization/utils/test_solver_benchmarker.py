# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import typing as ty

import numpy as np

from lava.lib.optimization.utils.solver_benchmarker import SolverBenchmarker


class TestSolverBenchmarker(unittest.TestCase):
    """
    Unit test suite for SolverBenchmarker.
    """

    def setUp(self) -> None:
        self.benchmarker = SolverBenchmarker()

    def test_create_obj(self):
        self.assertIsInstance(self.benchmarker, SolverBenchmarker)

    def test_power_measurement_cfg(self):
        num_steps = 5
        power_cfg = self.benchmarker.get_power_measurement_cfg(
            num_steps=num_steps
        )
        self.assertIsNotNone(power_cfg)

    def test_time_measurement_cfg(self):
        num_steps = 5
        time_cfg = self.benchmarker.get_time_measurement_cfg(
            num_steps=num_steps
        )
        self.assertIsNotNone(time_cfg)

    def test_power_cfg(self):
        num_steps = 5
        power_cfg = self.benchmarker.get_power_measurement_cfg(
            num_steps=num_steps
        )
        self.assertIsInstance(power_cfg[0][0], ty.Callable)

    def test_time_cfg(self):
        num_steps = 5
        time_cfg = self.benchmarker.get_time_measurement_cfg(
            num_steps=num_steps
        )
        self.assertIsInstance(time_cfg[0][0], ty.Callable)

    def test_measured_power_property(self):
        num_steps = 5
        power_cfg = self.benchmarker.get_power_measurement_cfg(
            num_steps=num_steps
        )
        measured_power = self.benchmarker.measured_power
        self.assertIsInstance(measured_power, np.ndarray)

    def test_measured_time_property(self):
        num_steps = 5
        time_cfg = self.benchmarker.get_time_measurement_cfg(
            num_steps=num_steps
        )
        measured_time = self.benchmarker.measured_time
        self.assertIsInstance(measured_time, np.ndarray)


if __name__ == "__main__":
    unittest.main()
