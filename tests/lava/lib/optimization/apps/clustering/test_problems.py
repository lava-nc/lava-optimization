import unittest
import numpy as np
from lava.lib.optimization.apps.clustering.problems import ClusteringProblem


class testClusteringProblem(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)
        self.point_coords = [(np.random.randint(0, 26),
                             np.random.randint(0, 26)) for _ in range(5)]
        self.center_coords = [(12, 13)] * 3
        self.edges = [(4, 5), (4, 6), (6, 8), (8, 7), (7, 6), (8, 5)]
        self.clust_prob_instance_natural_edges = ClusteringProblem(
            point_coords=self.point_coords, center_coords=self.center_coords)
        self.clust_prob_instance_user_edges = ClusteringProblem(
            point_coords=self.point_coords, center_coords=self.center_coords,
            edges=self.edges)

    def test_init(self):
        self.assertIsInstance(self.clust_prob_instance_user_edges,
                              ClusteringProblem)
        self.assertIsInstance(self.clust_prob_instance_natural_edges,
                              ClusteringProblem)

    def test_generate_problem_graph(self):
        ne_edges_gold = [(4, 5), (4, 6), (4, 7), (4, 8), (5, 4), (5, 6),
                         (5, 7), (5, 8), (6, 4), (6, 5), (6, 7), (6, 8),
                         (7, 4), (7, 5), (7, 6), (7, 8), (8, 4), (8, 5),
                         (8, 6), (8, 7), (1, 4), (1, 5), (1, 6), (1, 7),
                         (1, 8), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8),
                         (3, 4), (3, 5), (3, 6), (3, 7), (3, 8)]
        ne_edges_gold.sort(key=lambda x: x[0])

        ue_edges_gold = [(4, 5), (4, 6), (6, 8), (7, 6), (8, 7), (8, 5),
                         (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (2, 4),
                         (2, 5), (2, 6), (2, 7), (2, 8), (3, 4), (3, 5),
                         (3, 6), (3, 7), (3, 8)]
        ue_edges_gold.sort(key=lambda x: x[0])

        pg_ne = self.clust_prob_instance_natural_edges.problem_graph
        pg_ue = self.clust_prob_instance_user_edges.problem_graph
        ne_edges_graph = list(pg_ne.edges)
        ne_edges_graph.sort(key=lambda x: x[0])
        ue_edges_graph = list(pg_ue.edges)
        ue_edges_graph.sort(key=lambda x: x[0])

        self.assertListEqual(ne_edges_gold, ne_edges_graph)
        self.assertListEqual(ue_edges_gold, ue_edges_graph)


if __name__ == '__main__':
    unittest.main()
