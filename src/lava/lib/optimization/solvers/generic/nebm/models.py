import numpy as np

from lava.lib.optimization.solvers.generic.nebm.process import NEBM
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol


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

    # The local cost
    state: np.ndarray = LavaPyType(np.ndarray, int, precision=24)
    spk_hist: np.ndarray = LavaPyType(np.ndarray, int, precision=8)

    temperature: np.ndarray = LavaPyType(np.ndarray, int, precision=8)
    refract: np.ndarray = LavaPyType(np.ndarray, int, precision=8)

    refract_counter: np.ndarray = LavaPyType(np.ndarray, int, precision=8)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.a_in_data = np.zeros(proc_params['shape'])

    def _prng(self):
        """Pseudo-random number generator
        """

        # ToDo: Choosing a 16-bit signed random integer. For bit-accuracy,
        #   need to replace it with Loihi-conformant LFSR function
        prand = np.zeros(shape=self.state.shape)
        if prand.size > 0:
            prand = \
                np.random.randint(0, (2 ** 16) - 1, size=prand.size)

        return prand

    def _update_buffers(self):
        # !! Side effect: Changes self.beta !!

        # Populate the buffer for local computation
        spk_hist_buffer = self.spk_hist.copy()
        spk_hist_buffer &= 3
        self.spk_hist <<= 2
        self.spk_hist &= 0xFF  # AND with 0xFF retains 8 LSBs

        return spk_hist_buffer

    def _gen_sig_spks(self, spk_hist_buffer):
        s_sig = np.zeros_like(self.state)
        # If we have fired in the previous time-step, we send out the local
        # cost now, i.e., when spk_hist_buffer == 1
        sig_spk_idx = np.where(spk_hist_buffer == 1)
        # Send the local cost
        s_sig[sig_spk_idx] = self.state[sig_spk_idx]

        return s_sig

    def _gen_wta_spks(self, spk_hist_buffer):
        lfsr = self._prng()

        self.state += self.a_in_data
        # Note: this should not happen; otherwise, cost is too high/low!
        np.clip(self.state, a_min=-(2 ** 23), a_max=2 ** 23 - 1,
                out=self.state)

        # WTA spikes from previous time step
        wta_spk_idx_prev = (spk_hist_buffer == 1)

        # New WTA spikes
        # Spike always if this would decrease the energy
        wta_spk_idx = self.state < 0
        # WTA spike indices when threshold is exceeded
        wta_spk_idx = np.logical_or(
            wta_spk_idx,
            (2 ** 16 - 1) * self.temperature >= np.multiply(
                lfsr, (2 * self.temperature + self.state)))

        # Neurons can only switch states outside their refractory period
        wta_spk_idx = np.array([wta_spk_idx[ii] if self.refract_counter[ii] <= 0
                                else wta_spk_idx_prev[ii]
                                for ii in range(wta_spk_idx.shape[0])])

        # Spiking neuron voltages go in refractory (if neg_tau_ref < 0)
        # self.state[wta_spk_idx] = 0
        self.spk_hist[wta_spk_idx] |= 1

        s_wta = np.zeros_like(self.state)
        # Two kinds of spikes.
        # Switched off to on -> +1
        s_wta[np.logical_and(wta_spk_idx, np.logical_not(wta_spk_idx_prev))] \
            = +1
        # Switched on to off -> -1
        s_wta[np.logical_and(wta_spk_idx_prev, np.logical_not(wta_spk_idx))] \
            = -1

        self.refract_counter = np.maximum(
            self.refract_counter - 1,
            np.multiply(self.refract, (s_wta != 0)))

        # s_wta[wta_spk_idx] = 1

        return s_wta

    def run_spk(self) -> None:
        # Receive synaptic input
        self.a_in_data = self.a_in.recv().astype(int)

        # !! Side effect: Changes self.beta !!
        spk_hist_buffer = self._update_buffers()

        # Generate WTA spikes
        s_wta = self._gen_wta_spks(spk_hist_buffer)

        # Generate Sigma spikes
        s_sig = self._gen_sig_spks(spk_hist_buffer)
        # Send out spikes
        self.s_sig_out.send(s_sig)
        self.s_wta_out.send(s_wta)
