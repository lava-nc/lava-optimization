# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty
import numpy.typing as npty

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.lib.optimization.solvers.generic.scif.process import CspScif, \
    QuboScif


class PyModelAbstractScifFixed(PyLoihiProcessModel):
    """Abstract fixed point implementation of Stochastic Constraint
    Integrate and Fire (SCIF) neuron for solving QUBO and CSP problems.
    """

    a_in = LavaPyType(PyInPort.VEC_DENSE, int, precision=8)
    s_sig_out = LavaPyType(PyOutPort.VEC_DENSE, int, precision=8)
    s_wta_out = LavaPyType(PyOutPort.VEC_DENSE, int, precision=8)

    cnstr_intg: np.ndarray = LavaPyType(np.ndarray, int, precision=24)
    state: np.ndarray = LavaPyType(np.ndarray, int, precision=24)
    spk_hist: np.ndarray = LavaPyType(np.ndarray, int, precision=8)

    step_size: np.ndarray = LavaPyType(np.ndarray, int, precision=24)
    theta: np.ndarray = LavaPyType(np.ndarray, int, precision=24)
    noise_ampl: np.ndarray = LavaPyType(np.ndarray, int, precision=5)
    noise_prec: np.ndarray = LavaPyType(np.ndarray, int, precision=5)
    sustained_on_tau: np.ndarray = LavaPyType(np.ndarray, int, precision=24)

    def __init__(self, proc_params):
        """
        Notes
        -----
        **Noise model**:
        1. An unsigned 24-bit random number is chosen, say `R`
        2. The bit-precision of `noise_amplitude` is inferred from its value.
           Say, it is `ap`.
        3. The number `R` is truncated to (`noise_precision` - `ap`) bits and
           multiplied by `noise_amplitude`:
           ```
           S = noise_amplitude * [R << (24 - noise_precision + ap)] >> (
           24 - noise_precision + ap)
           ```
        4. The number `S` is used in the SCIF dynamics.
        """
        super(PyModelAbstractScifFixed, self).__init__(proc_params)
        self.a_in_data = np.zeros(proc_params["shape"])

    @staticmethod
    def _get_precision(vn_in: npty.NDArray) -> ty.Union[npty.NDArray, int]:
        """Get the minimum number of bits needed to represent every element
        of `vn`.
        """
        vn = vn_in.copy()
        if np.any(vn < 0):
            AssertionError(
                "_get_precision() is implemented only for "
                "non-negative integers"
            )
        prec = np.zeros_like(vn)
        while np.any(vn):
            prec[vn != 0] += 1
            vn >>= 1

        return prec.item() if prec.size == 1 else prec

    def _prng(self):
        """Pseudo-random number generator."""

        # ToDo: Choosing a 24-bit unsigned random integer. For bit-accuracy,
        #   need to replace it with Loihi-conformant LFSR function
        prand = np.random.randint(0, 2**24 - 1, size=self.state.size)
        ampl_prec = self._get_precision(self.noise_ampl)
        prand = self.noise_ampl * (
            ((prand << (24 - self.noise_prec + ampl_prec)) & 0xFFFFFF)
            >> (24 - self.noise_prec + ampl_prec)
        )
        # AND with 0xFFFFFF -> retains 24 LSBs
        return prand

    def _update_buffers(self):
        """Update spiking history from previous time-step."""
        # !! Side effect: Changes self.spk_hist !!

        # Populate the buffer for local computation
        spk_hist_buffer = self.spk_hist.copy()
        spk_hist_buffer &= 3
        self.spk_hist <<= 2
        self.spk_hist &= 0xFF  # AND with 0xFF retains 8 LSBs

        return spk_hist_buffer

    # This method is overloaded for CSP and QUBO
    def _get_local_validity_conflict(self, spk_hist_status):
        """Method for checking local conflict of a neuron and its incoming
        connections.

        Irrelevant for QUBO. Overloaded for CSP in a meaningful manner.

        Parameters
        ----------
        spk_hist_status : np.ndarray
            Status of spiking history of the previous time-step

        Returns
        -------
        local_validity : np.ndarray
            Boolean array with True corresponding to validity
        local_conflict : np.ndarray
            Boolean array with True corresponding to a conflict
        """
        return np.ones_like(self.cnstr_intg).astype(bool), np.zeros_like(
            self.cnstr_intg
        ).astype(bool)

    # This method is overloaded for CSP and QUBO
    def _gen_sig_spks(self, spk_hist_status, local_validity):
        """Method to generate 'sigma' spikes on the output axon that
        communicates local cost in the case of QUBO and termination of
        activity in the case of CSP.

        The method is overloaded for QUBO and CSP separately.

        Parameters
        ----------
        spk_hist_status : np.ndarray
            Spike history from previous time-step.
        local_validity : np.ndarray
            Boolean array with True corresponding to local validity
        Returns
        -------
        s_sig : np.ndarray
            Sigma spikes
        """
        return np.zeros_like(self.state)

    def _integration_dynamics(self, intg_idx):
        """Dynamics of integration of neural state.

        Parameters
        ----------
        intg_idx : np.ndarray
            Indices of neurons that need integration. Excludes the neurons
            in their refractory state.

        Returns
        -------
        wta_enter_on_state : np.ndarray
            Indices of neurons that have crossed threshold and are entering
            the 'on' state (i.e., refractory state, see refractory dynamics).
        """
        state_to_intg = self.state[intg_idx]  # voltages to be integrated
        cnstr_to_intg = self.cnstr_intg[intg_idx]  # currents to be integrated
        step_size_to_intg = self.step_size[intg_idx]  # bias to be integrated

        lfsr = self._prng()
        lfsr_to_intg = lfsr[intg_idx]

        state_to_intg = (
            state_to_intg + lfsr_to_intg + cnstr_to_intg + step_size_to_intg
        )
        np.clip(state_to_intg, a_min=0, a_max=2**24 - 1, out=state_to_intg)

        # Assign all temporary states to state Vars
        self.state[intg_idx] = state_to_intg

        # WTA spike indices when threshold is exceeded
        wta_enter_on_state = np.where(self.state >= self.theta)

        # Spiking neuron voltages go in refractory (sustained_on_tau < 0)
        self.state[wta_enter_on_state] = self.sustained_on_tau
        self.spk_hist[wta_enter_on_state] |= 1

        return wta_enter_on_state

    def _refractory_dynamics(self, rfct_idx, local_conflict):
        """Dynamics of refractory state of neurons.

        Here, 'refractory' means a neuron keeps spiking as long as it is in
        this state, ignoring all synaptic input. This is in contrast with
        biological neurons, which are silent when they are in a refractory
        period.

        Parameters
        ----------
        rfct_idx : np.ndarray
            Indices of neurons in refractory state.
        local_conflict

        Returns
        -------
        wta_enter_off_state : np.ndarray
            Indices of neurons that will enter 'off' state in the next
            time-step.
        """
        # Split/fork state variables
        state_in_rfct = self.state[rfct_idx]  # voltages in refractory
        spk_hist_in_rfct = self.spk_hist[rfct_idx]

        # Refractory dynamics
        state_in_rfct += 1  # voltage increments by 1 every step
        spk_hist_in_rfct |= 3

        # The following proxy avoids picking up states that have gone to zero
        # through their "natural" dynamics, and only picks up indices of
        # those states that have become zero by coming out of sustained ON
        # period.
        state_proxy_for_rfct = np.ones_like(self.state)
        state_proxy_for_rfct[rfct_idx] = state_in_rfct
        spk_hist_proxy = np.zeros_like(self.spk_hist)
        spk_hist_proxy[rfct_idx] = spk_hist_in_rfct

        # Second set of unsatisfied WTA indices based on refractory
        wta_enter_off_state = np.where(
            np.logical_or(
                state_proxy_for_rfct == 0,
                np.logical_and(spk_hist_proxy & 1, local_conflict),
            )
        )

        # Assign all temporary states to state Vars
        self.state[rfct_idx] = state_in_rfct
        self.spk_hist[rfct_idx] = spk_hist_in_rfct

        # Reset voltage of unsatisfied WTA in refractory
        self.state[wta_enter_off_state] = 0
        self.spk_hist[wta_enter_off_state] &= 0xFE
        # AND with 0xFE is same as AND with 2 (=0b10)

        return wta_enter_off_state

    def _gen_wta_spks(self, spk_hist_status, local_conflict):
        """Generate spikes on 'WTA' output axons.

        Parameters
        ----------
        spk_hist_status : np.ndarray
            Spiking history from previous time-step
        local_conflict : np.ndarray
            Boolean array with True corresponding to a local conflict
        Returns
        -------
        s_wta : np.ndarray
            WTA spikes
        """
        # Indices of WTA neurons signifying unsatisfied constraints, based on
        # buffered history from previous timestep
        # indices of neurons to be integrated:
        intg_idx = np.where(self.state >= 0)
        # indices of neurons in refractory:
        rfct_idx = np.where(self.state < 0)

        # Indices of WTA neurons that will spike and enter refractory
        wta_enter_on_state = self._integration_dynamics(intg_idx)

        wta_enter_off_state = np.where(
            np.logical_and(spk_hist_status & 1, local_conflict)
        )
        # Indices of WTA neurons coming out of refractory or those signifying
        # unsatisfied constraints
        wta_enter_off_state_2 = (
            self._refractory_dynamics(rfct_idx, local_conflict)
            if self.sustained_on_tau != 0
            else (np.array([], dtype=np.int32),)
        )

        s_wta = np.zeros_like(self.state)
        s_wta[wta_enter_on_state] = 1
        s_wta[wta_enter_off_state] = -1
        s_wta[wta_enter_off_state_2] = -1

        return s_wta

    def run_spk(self) -> None:
        # Receive synaptic input
        self.a_in_data = self.a_in.recv()

        # Add the incoming activation and saturate to min-max limits
        np.clip(
            self.cnstr_intg + self.a_in_data,
            a_min=-(2**23),
            a_max=2**23 - 1,
            out=self.cnstr_intg,
        )

        # !! Side effect: Changes self.beta !!
        # State history status:
        # 1 -> entering ON state, 2 -> entering OFF state, 3 -> refractory
        spk_hist_status = self._update_buffers()

        local_validity, local_conflict = self._get_local_validity_conflict(
            spk_hist_status
        )

        # Generate Sigma spikes
        s_sig = self._gen_sig_spks(spk_hist_status, local_validity)

        # Generate WTA spikes
        s_wta = self._gen_wta_spks(spk_hist_status, local_conflict)

        # Send out spikes
        self.s_sig_out.send(s_sig)
        self.s_wta_out.send(s_wta)


@implements(proc=CspScif, protocol=LoihiProtocol)
@requires(CPU)
@tag("fixed_pt")
class PyModelCspScifFixed(PyModelAbstractScifFixed):
    """Concrete implementation of Stochastic Constraint Integrate and
    Fire (SCIF) neuron for solving CSP problems.

    Derives from `PyModelAbstractScifFixed`.
    """

    def __init__(self, proc_params):
        super(PyModelCspScifFixed, self).__init__(proc_params)
        self.a_in_data = np.zeros(proc_params["shape"])

    def _get_local_validity_conflict(self, spk_hist_status):
        local_validity = self.cnstr_intg == 0
        local_conflict = self.cnstr_intg < 0

        return local_validity, local_conflict

    def _gen_sig_spks(self, spk_hist_status, local_validity):
        s_sig = np.zeros_like(self.state)
        # Gather spike and unsatisfied indices for summation axons
        sig_unsat_idx = np.where(spk_hist_status == 2)
        sig_spk_idx = np.where(
            np.logical_and(spk_hist_status == 1, local_validity)
        )

        # Assign sigma spikes (+/- 1)
        s_sig[sig_unsat_idx] = -1
        s_sig[sig_spk_idx] = 1

        return s_sig


@implements(proc=QuboScif, protocol=LoihiProtocol)
@requires(CPU)
@tag("fixed_pt")
class PyModelQuboScifFixed(PyModelAbstractScifFixed):
    """Concrete implementation of Stochastic Constraint Integrate and
    Fire (SCIF) neuron for solving QUBO problems.

    Derives from `PyModelAbstractScifFixed`.
    """

    cost_diagonal: np.ndarray = LavaPyType(np.ndarray, int, precision=24)

    def __init__(self, proc_params):
        super(PyModelQuboScifFixed, self).__init__(proc_params)
        self.a_in_data = np.zeros(proc_params["shape"])

    def _gen_sig_spks(self, spk_hist_status, local_validity):
        s_sig = np.zeros_like(self.state)

        sig_spk_idx = np.where(
            np.logical_and(spk_hist_status & 1, local_validity)
        )
        # Compute the local cost
        s_sig[sig_spk_idx] = (
            self.cnstr_intg[sig_spk_idx] + self.cost_diagonal[sig_spk_idx]
        )

        return s_sig


@implements(proc=QuboScif, protocol=LoihiProtocol)
@requires(CPU)
@tag("fixed_pt")
class PyModelQuboScifRefracFixed(PyLoihiProcessModel):
    """***Deprecated*** Concrete implementation of Stochastic Constraint
    Integrate and Fire (SCIF) neuron for solving QUBO problems.
    """
    a_in = LavaPyType(PyInPort.VEC_DENSE, int, precision=8)
    s_sig_out = LavaPyType(PyOutPort.VEC_DENSE, int, precision=8)
    s_wta_out = LavaPyType(PyOutPort.VEC_DENSE, int, precision=8)

    state: np.ndarray = LavaPyType(np.ndarray, int, precision=24)
    spk_hist: np.ndarray = LavaPyType(np.ndarray, int, precision=8)

    step_size: np.ndarray = LavaPyType(np.ndarray, int, precision=24)
    theta: np.ndarray = LavaPyType(np.ndarray, int, precision=24)
    noise_ampl: np.ndarray = LavaPyType(np.ndarray, int, precision=1)
    sustained_on_tau: np.ndarray = LavaPyType(np.ndarray, int, precision=24)
    sustained_off_tau: np.ndarray = LavaPyType(np.ndarray, int, precision=24)

    cost_diagonal: np.ndarray = LavaPyType(np.ndarray, int, precision=24)
    noise_shift: np.ndarray = LavaPyType(np.ndarray, int, precision=24)

    def __init__(self, proc_params):
        super(PyModelQuboScifRefracFixed, self).__init__(proc_params)
        self.a_in_data = np.zeros(proc_params["shape"])

    def _prng(self):
        """Pseudo-random number generator"""

        # ToDo: Choosing a 16-bit signed random integer. For bit-accuracy,
        #   need to replace it with Loihi-conformant LFSR function
        prand = np.zeros(shape=self.state.shape)
        if prand.size > 0:
            rand_nums = \
                np.random.randint(0, (2 ** 16) - 1, size=prand.size)
            # Assign random numbers only to neurons, for which noise is enabled
            prand = np.right_shift(
                (rand_nums * self.noise_ampl).astype(int), self.noise_shift
            )

        return prand

    def _update_buffers(self):
        # !! Side effect: Changes self.beta !!

        # Populate the buffer for local computation
        spk_hist_buffer = self.spk_hist.copy()
        spk_hist_buffer &= 1
        self.spk_hist <<= 1
        # The following ensures that we flush all history beyond 3 timesteps
        # The number '3' comes from the fact that it takes 3 timesteps to
        # read out solutions after a minimum cost is detected (due to
        # downstream connected process-chain)
        self.spk_hist &= 7

        return spk_hist_buffer

    def _integration_dynamics(self, intg_idx):
        lfsr = self._prng()
        a_in_to_intg = self.a_in_data[intg_idx]
        state_to_intg = self.state[intg_idx]  # voltages to be integrated
        cost_diag_intg = self.cost_diagonal[intg_idx]
        lfsr_to_intg = lfsr[intg_idx]

        state_to_intg += lfsr_to_intg + cost_diag_intg + a_in_to_intg
        np.clip(state_to_intg, a_min=-(2 ** 23), a_max=2 ** 23 - 1,
                out=state_to_intg)
        self.state[intg_idx] = state_to_intg

        # WTA spike indices when threshold is exceeded
        wta_spk_idx = np.where(self.state >= self.theta)  # Exceeds threshold
        # Spiking neuron voltages go in refractory (if sustained_on_tau < 0)
        self.state[wta_spk_idx] = self.sustained_on_tau + self.sustained_off_tau
        self.spk_hist[wta_spk_idx] |= 1

        return wta_spk_idx

    def _refractory_dynamics(self, rfct_idx):
        # Split/fork state variables u, v, beta
        state_in_rfct = self.state[rfct_idx]  # voltages in refractory

        # Refractory dynamics
        state_in_rfct += 1  # voltage increments by 1 every step
        # Assign all temporary states to state Vars
        self.state[rfct_idx] = state_in_rfct
        # Neurons with sustained_on + sustained_off <= state < sustained_off
        # need to keep on spiking, those with sustained_off <= state < 0 need
        # to be silent
        wta_keep_on_idx = np.where(
            np.logical_and(
                self.sustained_on_tau + self.sustained_off_tau <= self.state,
                self.state < self.sustained_off_tau,
            )
        )
        wta_keep_off_idx = np.where(
            np.logical_and(self.sustained_off_tau <= self.state, self.state < 0)
        )
        # Need to update history for neurons which will keep on spiking
        self.spk_hist[wta_keep_on_idx] |= 1

        return wta_keep_on_idx, wta_keep_off_idx

    def _gen_sig_spks(self, spk_hist_buffer):
        s_sig = np.zeros_like(self.state)
        # If we have fired in the previous time-step, we send out the local
        # cost now, i.e., when spk_hist_buffer == 1
        sig_spk_idx = np.where(spk_hist_buffer == 1)
        # Compute the local cost
        s_sig[sig_spk_idx] = (
            self.cost_diagonal[sig_spk_idx] + self.a_in_data[sig_spk_idx]
        )
        return s_sig

    def _gen_wta_spks(self):

        # indices of neurons to be integrated:
        intg_idx = np.where(self.state >= 0)
        # indices of neurons in refractory:
        rfct_idx = np.where(self.state < 0)

        # Indices of WTA neurons that will spike and enter refractory
        wta_spk_idx = self._integration_dynamics(intg_idx)

        # Indices of WTA neurons coming out of refractory or those signifying
        # unsatisfied constraints
        wta_keep_on_idx, wta_keep_off_idx = (
            self._refractory_dynamics(rfct_idx)
            if self.sustained_on_tau != 0 or self.sustained_off_tau != 0
            else (np.array([], dtype=np.int32), np.array([], dtype=np.int32))
        )

        s_wta = np.zeros_like(self.state)
        s_wta[wta_spk_idx] = 1
        s_wta[wta_keep_on_idx] = 1
        s_wta[wta_keep_off_idx] = 0

        return s_wta

    def run_spk(self) -> None:

        # Receive synaptic input
        self.a_in_data = self.a_in.recv().astype(int)

        # !! Side effect: Changes self.beta !!
        spk_hist_buffer = self._update_buffers()

        # Generate Sigma spikes
        s_sig = self._gen_sig_spks(spk_hist_buffer)

        # Generate WTA spikes
        s_wta = self._gen_wta_spks()

        # Send out spikes
        self.s_sig_out.send(s_sig)
        self.s_wta_out.send(s_wta)
