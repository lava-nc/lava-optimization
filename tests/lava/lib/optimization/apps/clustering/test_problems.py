import unittest
import numpy as np
from lava.lib.optimization.apps.clustering.problems import ClusteringProblem
from lava.lib.optimization.utils.generators.clustering_tsp_vrp import (
    GaussianSampledClusteringProblem)


class TestClusteringProblem(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(2)
        self.gcp = GaussianSampledClusteringProblem(num_clusters=4,
                                                    num_points=10,
                                                    domain=[(0, 0), (25, 25)],
                                                    variance=3)
        self.cp = ClusteringProblem(point_coords=self.gcp.point_coords,
                                    center_coords=self.gcp.center_coords)

    def test_init(self):
        self.assertIsInstance(self.cp, ClusteringProblem)

    def test_properties(self):
        self.assertEqual(self.cp.num_clusters, 4)
        self.assertEqual(self.cp.num_points, 10)
        self.assertListEqual(self.cp.cluster_ids, list(range(1, 5)))
        self.assertListEqual(self.cp.point_ids, list(range(5, 15)))


if __name__ == '__main__':
    unittest.main()
