# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

from lava.lib.optimization.solvers.generic.nebm.process import NEBM
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol


def boltzmann(delta_e, temperature):
    """ Return the probability that each unit should be ON from a boltzmann
    distribution with the given temperature. Note that the boltzmann
    distribution is undefined for temperature equal to zero, in this case
    return probabilities corresponding to greedy energy descent."""
    if temperature == 0:
        return delta_e < 0
    return 1 / (1 + np.exp(delta_e / temperature))


@implements(proc=NEBM, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class NEBMPyModel(PyLoihiProcessModel):
    """
    Fixed point implementation of Boltzmann neuron for solving QUBO problems.
    """
    a_in = LavaPyType(PyInPort.VEC_DENSE, int, precision=24)
    s_sig_out = LavaPyType(PyOutPort.VEC_DENSE, int, precision=24)
    s_wta_out = LavaPyType(PyOutPort.VEC_DENSE, int, precision=24)
    state: np.ndarray = LavaPyType(np.ndarray, int, precision=24)
    spk_hist: np.ndarray = LavaPyType(np.ndarray, int, precision=8)
    temperature: np.ndarray = LavaPyType(np.ndarray, int, precision=8)
    refract: np.ndarray = LavaPyType(np.ndarray, int, precision=8)
    refract_counter: np.ndarray = LavaPyType(np.ndarray, int, precision=8)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.a_in_data = np.zeros(proc_params['shape'])

    def _update_buffers(self):
        # Populate the buffer for local computation
        spk_hist_buffer = self.spk_hist.copy()
        spk_hist_buffer &= 3
        self.spk_hist <<= 2
        self.spk_hist &= 0xFF  # AND with 0xFF retains 8 LSBs
        return spk_hist_buffer

    def _generate_sigma_spikes(self, spk_hist_buffer):
        s_sig = np.zeros_like(self.state)
        # If we have fired in the previous time-step, we send out the local
        # cost now, i.e., when spk_hist_buffer == 1
        sig_spk_idx = np.where(spk_hist_buffer == 1)
        # Send the local cost
        s_sig[sig_spk_idx] = self.state[sig_spk_idx]
        return s_sig

    def _generate_wta_spikes(self, spk_hist_buffer):
        self.state += self.a_in_data
        wta_spk_idx_prev = (spk_hist_buffer == 1)
        # HACK: Need to enable temperature < 1, because temperature = 1 is
        # already too hot for small QUBOs.
        prob = boltzmann(self.state, self.temperature[0] / 4.0)
        rand = np.random.rand(*prob.shape)
        wta_spk_idx = rand < prob
        refractory = self.refract_counter > 0
        wta_spk_idx[refractory] = wta_spk_idx_prev[refractory]
        # Add spikes to history
        self.spk_hist[wta_spk_idx] |= 1
        # Generate output spikes for feedback
        s_wta = np.zeros_like(self.state)
        pos_spk = np.logical_and(wta_spk_idx, np.logical_not(wta_spk_idx_prev))
        neg_spk = np.logical_and(wta_spk_idx_prev, np.logical_not(wta_spk_idx))
        s_wta[pos_spk] = +1
        s_wta[neg_spk] = -1
        # Update the refractory periods
        self.refract_counter = np.maximum(
            self.refract_counter - 1,
            np.multiply(self.refract, (s_wta != 0)))
        return s_wta

    def run_spk(self) -> None:
        self.a_in_data = self.a_in.recv().astype(int)
        spk_hist_buffer = self._update_buffers()
        s_wta = self._generate_wta_spikes(spk_hist_buffer)
        s_sig = self._generate_sigma_spikes(spk_hist_buffer)
        self.s_wta_out.send(s_wta)
        self.s_sig_out.send(s_sig)
