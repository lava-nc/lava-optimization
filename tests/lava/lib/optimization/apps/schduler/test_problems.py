# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
from lava.lib.optimization.apps.scheduler.problems import (
    SchedulingProblem, SatelliteScheduleProblem)


class TestSchedulingProblem(unittest.TestCase):

    def setUp(self) -> None:
        self.sp = SchedulingProblem(num_agents=3, num_tasks=3)

    def test_init(self):
        self.assertIsInstance(self.sp, SchedulingProblem)

    def test_generate(self):
        self.sp.generate(42)
        print(f"{dict(self.sp.graph.nodes)}")
        

class TestSatelliteSchedulingProblem(unittest.TestCase):
    def test_init(self):
        pass


if __name__ == '__main__':
    unittest.main()
