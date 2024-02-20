# Code to test Process interface of Breadth First Search based graph search

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

# Tests Hierarchical process models in BFS model

import unittest
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
import numpy as np

from lava.lib.optimization.solvers.graphs.bfs.process import (
    BreadthFirstSearch,
)
from lava.lib.optimization.solvers.graphs.bfs.models import (
    BreadthFirstSearchModel,
)


connection_config = []


class TestBreadthFirstSearchModel(unittest.TestCase):
    """Test the models chosen in Breadth First Search
    Breadth First Search
    """

    def test_initalization_of_process(self):
        pass

    def test_connectivity_matrices(self):
        np.random.seed(5)
        dim = 5
        # Generate random symmetric matrix with 1s and 0s
        rand_mat = np.random.randint(2, size=(dim, dim))
        sym_mat = np.triu(rand_mat) + np.triu(rand_mat).T
        sym_mat = np.where(sym_mat != 0, 1, 0)

        # Specify start and target nodes
        status = np.array([0, 128, 0, 64, 0])

        process = BreadthFirstSearch(
            connection_config,
            adjacency_matrix=sym_mat,
            node_description=status,
        )

        model = BreadthFirstSearchModel(process)
        # process.run(condition=RunSteps(num_steps=1), run_cfg=Loihi2SimCfg())


if __name__ == "__main__":
    unittest.main()
