import unittest
import numpy as np
from lava.lib.optimization.apps.vrp.problems import VRP

from pprint import pprint


class testVRP(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)
        self.node_coords = [(np.random.randint(0, 26),
                             np.random.randint(0, 26)) for _ in range(5)]
        self.vehicle_coords = [(12, 13)] * 3
        self.edges = [(4, 5), (4, 6), (6, 8), (8, 7), (7, 6), (8, 5)]
        self.vrp_instance_natural_edges = VRP(
            node_coords=self.node_coords, vehicle_coords=self.vehicle_coords)
        self.vrp_instance_user_edges = VRP(
            node_coords=self.node_coords, vehicle_coords=self.vehicle_coords,
            edges=self.edges)

    def test_init(self):
        self.assertIsInstance(self.vrp_instance_natural_edges, VRP)
        self.assertIsInstance(self.vrp_instance_user_edges, VRP)

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

        pg_ne = self.vrp_instance_natural_edges.problem_graph
        pg_ue = self.vrp_instance_user_edges.problem_graph
        ne_edges_graph = list(pg_ne.edges)
        ne_edges_graph.sort(key=lambda x: x[0])
        ue_edges_graph = list(pg_ue.edges)
        ue_edges_graph.sort(key=lambda x: x[0])

        self.assertListEqual(ne_edges_gold, ne_edges_graph)
        self.assertListEqual(ue_edges_gold, ue_edges_graph)


if __name__ == '__main__':
    unittest.main()
