# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import pprint
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
        nodeids = list(self.sp.graph.nodes.keys())
        nodedicts = list(self.sp.graph.nodes.values())
        self.assertListEqual(list(range(9)), nodeids)
        for j in range(3):
            for k in range(3):
                self.assertTupleEqual((nodedicts[3 * j + k]['agent_id'],
                                       nodedicts[3 * j + k]['task_id']),
                                      (j, k))


class TestSatelliteSchedulingProblem(unittest.TestCase):

    def setUp(self) -> None:
        self.ssp = SatelliteScheduleProblem(num_satellites=8,
                                            num_requests=48,
                                            view_height=0.125)

    def test_init(self):
        self.assertIsInstance(self.ssp, SatelliteScheduleProblem)




if __name__ == '__main__':
    unittest.main()
