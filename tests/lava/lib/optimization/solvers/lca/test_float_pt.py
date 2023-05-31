#  Copyright (C) 2023 Battelle Memorial Institute
#  SPDX-License-Identifier: BSD-2-Clause
#  See: https://spdx.org/licenses/

import unittest

import numpy as np
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.io import sink

from lava.lib.optimization.solvers.lca.process import LCA1Layer, LCA2Layer


class TestLCAFloat(unittest.TestCase):
    def test_identity_matrix(self):
        """
        Tests the sparse code is equal to the input when using the identity
        matrix as a dictionary.
        """
        weights = np.eye(5)
        input_val = np.array([0.99203318, 0.53107722, 0.84873413,
                              0.15441692, 0.60863695])
        threshold = 0.001
        lca = LCA2Layer(weights=weights, input_vec=input_val,
                        threshold=threshold, spike_height=0)
        v1_output = sink.RingBuffer(shape=(5,), buffer=1)
        res_output = sink.RingBuffer(shape=(5,), buffer=1)
        lca.v1.connect(v1_output.a_in)
        lca.res.connect(res_output.a_in)

        run_config = Loihi1SimCfg(select_tag='floating_pt',
                                  select_sub_proc_model=True)

        v1_output.run(condition=RunSteps(num_steps=1000), run_cfg=run_config)

        self.assertTrue(
            np.allclose(input_val, v1_output.data.get()[:, 0], atol=threshold),
            f"Expected: {input_val} Actual: {v1_output.data.get()[:, 0]}")

        v1_output.stop()

    def test_negative_residual(self):
        """
        Test that the input can be negative
        """
        weights = np.eye(5)
        input_val = np.array([0.48474121, 1.384066, -0.074854,
                              -0.03878497, 0.50936179])
        threshold = 0.001
        lca = LCA2Layer(weights=weights, input_vec=input_val,
                        threshold=threshold, spike_height=0)
        v1_output = sink.RingBuffer(shape=(5,), buffer=1)
        res_output = sink.RingBuffer(shape=(5,), buffer=1)
        lca.v1.connect(v1_output.a_in)
        lca.res.connect(res_output.a_in)

        run_config = Loihi1SimCfg(select_tag='floating_pt',
                                  select_sub_proc_model=True)

        v1_output.run(condition=RunSteps(num_steps=1000), run_cfg=run_config)

        self.assertTrue(
            np.allclose(input_val, v1_output.data.get()[:, 0], atol=threshold),
            f"Expected: {input_val} Actual: {v1_output.data.get()[:, 0]}")
        v1_output.stop()

    def test_competition_2_layer(self):
        """
        Test that V1 neurons exhibit competition, where the first element of the
        dictionary inhibits the second from being used.
        """
        weights = np.array([[0, np.sqrt(1 / 2), np.sqrt(1 / 2)],
                            [np.sqrt(1 / 3), np.sqrt(1 / 3), np.sqrt(1 / 3)]])
        input_val = np.array([0, 1, 1])
        expected = np.array([1 / np.sqrt(1 / 2), 0])

        threshold = 0.001
        lca = LCA2Layer(weights=weights, input_vec=input_val,
                        threshold=threshold, spike_height=0)
        v1_output = sink.RingBuffer(shape=(2,), buffer=1)
        res_output = sink.RingBuffer(shape=(3,), buffer=1)
        lca.v1.connect(v1_output.a_in)
        lca.res.connect(res_output.a_in)

        run_config = Loihi1SimCfg(select_tag='floating_pt',
                                  select_sub_proc_model=True)

        v1_output.run(condition=RunSteps(num_steps=1000), run_cfg=run_config)

        self.assertTrue(
            np.allclose(expected, v1_output.data.get()[:, 0], atol=threshold),
            f"Expected: {expected} Actual: {v1_output.data.get()[:, 0]}")

        v1_output.stop()

    def test_excitation_2_layer(self):
        """
        Test that V1 neurons exhibit excitation, where the first element of the
        dictionary excites the second into being used.
        """
        weights = np.array([[-np.sqrt(1 / 3), np.sqrt(1 / 3), np.sqrt(1 / 3)],
                            [1, 0, 0]])
        input_val = np.array([0, 1, 1])
        expected = np.array([1 / np.sqrt(1 / 3), 1])

        threshold = 0.001
        lca = LCA2Layer(weights=weights, input_vec=input_val,
                        threshold=threshold, spike_height=0)
        v1_output = sink.RingBuffer(shape=(2,), buffer=1)
        res_output = sink.RingBuffer(shape=(3,), buffer=1)
        lca.v1.connect(v1_output.a_in)
        lca.res.connect(res_output.a_in)

        run_config = Loihi1SimCfg(select_tag='floating_pt',
                                  select_sub_proc_model=True)

        v1_output.run(condition=RunSteps(num_steps=1000), run_cfg=run_config)

        self.assertTrue(
            np.allclose(expected, v1_output.data.get()[:, 0], atol=threshold,
                        rtol=1e-2),
            f"Expected: {expected} Actual: {v1_output.data.get()[:, 0]}")

        v1_output.stop()

    def test_competition_1_layer(self):
        """
        Test that V1 neurons exhibit competition, where the first element of the
        dictionary inhibits the second from being used.
        """
        weights = np.array([[0, np.sqrt(1 / 2), np.sqrt(1 / 2)],
                            [np.sqrt(1 / 3), np.sqrt(1 / 3), np.sqrt(1 / 3)]])
        w = (np.einsum('bj,ij->bi', -weights, weights) + np.eye(
            weights.shape[0])) * 0.1
        input_val = np.array([0, 1, 1])
        bias = (input_val @ weights.T * 0.1)
        expected = np.array([1 / np.sqrt(1 / 2), 0])

        threshold = 0.001
        lca = LCA1Layer(weights=w, bias=bias, threshold=threshold)
        v1_output = sink.RingBuffer(shape=(2,), buffer=1)
        lca.v1.connect(v1_output.a_in)

        run_config = Loihi1SimCfg(select_tag='floating_pt',
                                  select_sub_proc_model=True)

        v1_output.run(condition=RunSteps(num_steps=1000), run_cfg=run_config)

        self.assertTrue(
            np.allclose(expected, v1_output.data.get()[:, 0], atol=threshold),
            f"Expected: {expected} Actual: {v1_output.data.get()[:, 0]}")

        v1_output.stop()

    def test_excitation_1_layer(self):
        """
        Test that V1 neurons exhibit excitation, where the first element of the
        dictionary excites the second into being used.
        """
        weights = np.array([[-np.sqrt(1 / 3), np.sqrt(1 / 3), np.sqrt(1 / 3)],
                            [1, 0, 0]])
        w = (np.einsum('bj,ij->bi', -weights, weights) + np.eye(
            weights.shape[0])) * 0.1
        input_val = np.array([0, 1, 1])
        bias = (input_val @ weights.T * 0.1)
        expected = np.array([1 / np.sqrt(1 / 3), 1])

        threshold = 0.001
        lca = LCA1Layer(weights=w, bias=bias, threshold=threshold)
        v1_output = sink.RingBuffer(shape=(2,), buffer=1)
        lca.v1.connect(v1_output.a_in)

        run_config = Loihi1SimCfg(select_tag='floating_pt',
                                  select_sub_proc_model=True)

        v1_output.run(condition=RunSteps(num_steps=1000), run_cfg=run_config)

        self.assertTrue(
            np.allclose(expected, v1_output.data.get()[:, 0], atol=threshold,
                        rtol=1e-2),
            f"Expected: {expected} Actual: {v1_output.data.get()[:, 0]}")

        v1_output.stop()


if __name__ == "__main__":
    unittest.main()
