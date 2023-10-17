# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import pprint
import unittest

import numpy as np

from lava.lib.optimization.apps.scheduler.problems import (
    SchedulingProblem, SatelliteScheduleProblem)


class TestSchedulingProblem(unittest.TestCase):

    def setUp(self) -> None:
        self.sp = SchedulingProblem(num_agents=3, num_tasks=3)

    def test_init(self):
        self.assertIsInstance(self.sp, SchedulingProblem)

    def test_generate(self):
        self.sp.generate(seed=42)
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
                                            requests=requests,
                                            view_height=0.5,
                                            seed=42)

    def test_init(self):
        self.assertIsInstance(self.ssp, SatelliteScheduleProblem)

    def test_generate(self):
        self.ssp.generate(seed=42)
        gt_graph_dict = {0: {'agent_attr': (0.25, 0.5),
                             'agent_id': 1,
                             'task_attr': [0.30424224, 0.52475643],
                             'task_id': 5},
                         1: {'agent_attr': (0.25, 0.5),
                             'agent_id': 1,
                             'task_attr': [0.73199394, 0.59865848],
                             'task_id': 10},
                         2: {'agent_attr': (0.25, 1.0),
                             'agent_id': 2,
                             'task_attr': [0.02058449, 0.96990985],
                             'task_id': 0},
                         3: {'agent_attr': (0.25, 1.0),
                             'agent_id': 2,
                             'task_attr': [0.37454012, 0.95071431],
                             'task_id': 6}}
        self.assertDictEqual(gt_graph_dict, dict(self.ssp.graph.nodes))


if __name__ == '__main__':
    unittest.main()
