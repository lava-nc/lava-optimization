# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.lib.optimization.utils.generators.clustering_tsp_vrp import (
    AbstractProblem, AbstractClusteringProblem, AbstractTSP, AbstractVRP,
    AbstractUniformProblem, AbstractGaussianProblem,
    UniformlySampledClusteringProblem, GaussianSampledClusteringProblem)


class TestAbstractProblem(unittest.TestCase):

    def test_abstract_problem_init(self):
        ap = AbstractProblem()
        self.assertIsInstance(ap, AbstractProblem)

    def test_abstract_problem_properties(self):
        dom = np.array([[-5, -5], [5, 5]])
        ap = AbstractProblem(num_anchors=5, num_nodes=20, domain=dom)
        self.assertEqual(ap.num_anchors, 5)
        self.assertEqual(ap.num_nodes, 20)
        self.assertEqual(ap.num_pt_per_clust, 4)  # 20 // 5 = 4
        self.assertTrue(np.all(ap.domain == dom))
        self.assertListEqual(ap.domain_ll.tolist(), [-5, -5])
        self.assertListEqual(ap.domain_ur.tolist(), [5, 5])
        self.assertIsNone(ap.anchor_coords)
        self.assertIsNone(ap.node_coords)

    def test_abstract_problem_setters(self):
        ap = AbstractProblem()
        ap.num_anchors = 4
        ap.num_nodes = 36
        ap.domain = [(0, 0), (20, 20)]
        self.assertEqual(ap.num_anchors, 4)
        self.assertEqual(ap.num_nodes, 36)
        self.assertTrue(np.all(ap.domain == np.array([(0, 0), (20, 20)])))


class TestAbstractDerivedProblems(unittest.TestCase):

    def test_abstract_clustering_problem(self):
        acp = AbstractClusteringProblem(num_clusters=5,
                                        num_points=20)
        self.assertEqual(acp.num_clusters, 5)
        self.assertEqual(acp.num_anchors, 5)
        self.assertEqual(acp.num_points, 20)
        self.assertEqual(acp.num_nodes, 20)
        self.assertIsNone(acp.center_coords)
        self.assertIsNone(acp.point_coords)

    def test_abstract_tsp(self):
        atsp = AbstractTSP(num_starting_pts=5,
                           num_dest_nodes=20)
        self.assertEqual(atsp.num_starting_pts, 5)
        self.assertEqual(atsp.num_anchors, 5)
        self.assertEqual(atsp.num_dest_nodes, 20)
        self.assertEqual(atsp.num_nodes, 20)
        self.assertIsNone(atsp.starting_coords)
        self.assertIsNone(atsp.dest_coords)

    def test_abstract_vrp(self):
        avrp = AbstractVRP(num_vehicles=5,
                           num_waypoints=20)
        self.assertEqual(avrp.num_vehicles, 5)
        self.assertEqual(avrp.num_anchors, 5)
        self.assertEqual(avrp.num_waypoints, 20)
        self.assertEqual(avrp.num_nodes, 20)
        self.assertIsNone(avrp.vehicle_coords)
        self.assertIsNone(avrp.waypoint_coords)

    def test_abstract_uniform_problem(self):
        np.random.seed(2)
        aup = AbstractUniformProblem(num_anchors=4,
                                     num_nodes=10,
                                     domain=[(-5, -5), (5, 5)])
        gt_anchor_coords = np.array([[3, 3], [1, -3], [3, 2], [-3, -4]])
        gt_node_coords = np.array([[0, -1], [-1, 0], [2, -2], [1, -1],
                                   [-2, 2], [1, -4], [-2, 0], [3, -1],
                                   [1, -2], [4, -3]])
        self.assertTrue(np.all(aup.anchor_coords == gt_anchor_coords))
        self.assertTrue(np.all(aup.node_coords == gt_node_coords))

    def test_abstract_gaussian_problem(self):
        np.random.seed(2)
        agp = AbstractGaussianProblem(num_anchors=4,
                                      num_nodes=7,
                                      domain=[(-3, -3), (3, 3)])
        gt_anchor_coords = np.array([[-3, 2], [-3, 0], [-1, 0], [-3, -1]])
        gt_node_coords = np.array([[-4, 2], [-2, -1], [-4, 0], [-2, 0],
                                   [-2, 0], [-1, -1], [0, 0]])
        self.assertTrue(np.all(agp.anchor_coords == gt_anchor_coords))
        self.assertTrue(np.all(agp.node_coords == gt_node_coords))


class TestClusteringProblems(unittest.TestCase):

    def test_uniform_clustering(self):
        np.random.seed(2)
        ucp = UniformlySampledClusteringProblem(num_clusters=4,
                                                num_points=10,
                                                domain=[(0, 0), (25, 25)])
        gt_center_coords = np.array([[8, 15], [13, 8], [22, 11], [18, 11]])
        gt_point_coords = np.array([[8, 7], [2, 17], [11, 21], [15, 20],
                                    [20, 5], [7, 3], [6, 4], [10, 11],
                                    [19, 7], [6, 10]])
        self.assertTrue(np.all(ucp.center_coords == gt_center_coords))
        self.assertTrue(np.all(ucp.point_coords == gt_point_coords))

    def test_gaussian_clustering(self):
        np.random.seed(2)
        gcp = GaussianSampledClusteringProblem(num_clusters=4,
                                               num_points=10,
                                               domain=[(0, 0), (25, 25)],
                                               variance=3)
        gt_center_coords = np.array([[8, 15], [13, 8], [22, 11], [18, 11]])
        gt_point_coords = np.array([[4, 13], [3, 17], [10, 10], [9, 8],
                                    [8, 8], [23, 12], [25, 10], [18, 8],
                                    [17, 8], [13, 12]])
        self.assertTrue(np.all(gcp.center_coords == gt_center_coords))
        self.assertTrue(np.all(gcp.point_coords == gt_point_coords))


if __name__ == '__main__':
    unittest.main()
