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

    state: np.ndarray = LavaPyType(np.ndarray, int, precision=24)
    spk_hist: np.ndarray = LavaPyType(np.ndarray, int, precision=8)

    temperature: np.ndarray = LavaPyType(np.ndarray, int, precision=8)
    refract: np.ndarray = LavaPyType(np.ndarray, int, precision=8)

    refract_counter: np.ndarray = LavaPyType(np.ndarray, int, precision=8)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.a_in_data = np.zeros(proc_params['shape'])

    def _get_random(self):
        # ToDo: Choosing a 16-bit signed random integer. For bit-accuracy,
        # need to replace it with Loihi-conformant LFSR function
        prand = np.zeros(shape=self.state.shape)
        if prand.size > 0:
            prand = np.random.randint(0, (2 ** 16) - 1, size=prand.size)
        return prand

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
        lfsr = self._get_random()
        self.state += self.a_in_data
        # Note: this should not happen; otherwise, cost is too high/low!
        np.clip(self.state, a_min=-(2 ** 23), a_max=2 ** 23 - 1, out=self.state)
        # Get spikes from previous time step
        wta_spk_idx_prev = (spk_hist_buffer == 1)
        # Generate new spikes when spiking will decrease the energy (grad descent)
        # or randomly in proportion to the temperature, but only when not refractory
        wta_spk_idx = self.state < 0
        rng_spk = lfsr < self.temperature - self.state
        wta_spk_idx = np.logical_or(wta_spk_idx, rng_spk)
        refractory = self.refract_counter > 0
        wta_spk_idx[refractory] = wta_spk_idx_prev[refractory]
        # Add spikes to history
        self.spk_hist[wta_spk_idx] |= 1
        # Generate output spikes for feedback
        s_wta = np.zeros_like(self.state)
        s_wta[np.logical_and(wta_spk_idx, np.logical_not(wta_spk_idx_prev))] = +1
        s_wta[np.logical_and(wta_spk_idx_prev, np.logical_not(wta_spk_idx))] = -1
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
