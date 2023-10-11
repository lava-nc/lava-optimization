import unittest
import numpy as np
from lava.lib.optimization.apps.tsp.problems import TravellingSalesmanProblem
from lava.lib.optimization.utils.generators.clustering_tsp_vrp import (
    GaussianSampledTSP)


class testTravellingSalesmanProblem(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(2)
        self.gtsp = GaussianSampledTSP(num_starting_pts=1,
                                       num_dest_nodes=5,
                                       domain=[(0, 0), (25, 25)],
                                       variance=3)
        self.tsp = TravellingSalesmanProblem(
            waypt_coords=self.gtsp.dest_coords,
            starting_pt=self.gtsp.starting_coords[0]
        )

    def test_init(self):
        self.assertIsInstance(self.tsp, TravellingSalesmanProblem)

    def test_properties(self):
        self.assertEqual(self.tsp.num_waypts, 5)
        self.assertListEqual(self.tsp.waypt_ids, list(range(2, 7)))


if __name__ == '__main__':
    unittest.main()
