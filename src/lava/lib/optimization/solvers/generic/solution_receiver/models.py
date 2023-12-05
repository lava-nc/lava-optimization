# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import numpy as np
from lava.lib.optimization.solvers.generic.monitoring_processes\
    .solution_readout.process import SolutionReadout
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import (
    PyLoihiProcessModel,
    PyAsyncProcessModel
)
from lava.magma.core.model.py.ports import PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol

from lava.lib.optimization.solvers.generic.solution_receiver.process import \
    (
    SolutionReceiver, SpikeIntegrator
)
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.sparse.process import Sparse
from lava.utils.weightutils import SignMode
from lava.proc import embedded_io as eio

@implements(SolutionReceiver, protocol=AsyncProtocol)
@requires(CPU)
class SolutionReceiverPyModel(PyAsyncProcessModel):
    """CPU model for the SolutionReadout process.
    The process receives two types of messages, an updated cost and the
    state of
    the solver network representing the current candidate solution to an
    OptimizationProblem. Additionally, a target cost can be defined by the
    user, once this cost is reached by the solver network, this process
    will request the runtime service to pause execution.
    """

    best_state: np.ndarray = LavaPyType(np.ndarray, np.int8, 32)
    best_timestep: np.ndarray = LavaPyType(np.ndarray, np.int32, 32)
    best_cost: np.ndarray = LavaPyType(np.ndarray, np.int32, 32)
    num_message_bits: np.ndarray = LavaPyType(np.ndarray, np.int8, 32)

    state_in: PyInPort = LavaPyType(
        PyInPort.VEC_DENSE, np.int32, precision=32
    )
    cost_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32,
                                              precision=32)
    timestep_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32,
                                   precision=32)

    def run_async(self):
        num_message_bits = self.num_message_bits[0]

        buffer_cost = 0
        buffer_timestep = 0
        while buffer_cost == 0:
            buffer_cost = self.cost_in.recv()
            buffer_timestep = self.timestep_in.recv()

        # CProcModel currently has integer overflow
        if buffer_cost > 0:
            buffer_cost -= 2**num_message_bits

        self.best_cost = buffer_cost
        self.best_timestep = buffer_timestep

        compressed_states = np.zeros(self.state_in.shape)
        while not np.any(compressed_states):
            compressed_states = self.state_in.recv()

        buffer = self._decompress_state(compressed_states, num_message_bits).copy()

        self.best_state[:] = buffer[0]

        self._req_pause = True

    @staticmethod
    def _decompress_state(compressed_states, num_message_bits):
        """Add info!"""
        boolean_array = (compressed_states[:, None] & (
                1 << np.arange(num_message_bits - 1, -1, -1))) != 0
        # reshape into a 1D array
        boolean_array.reshape(-1)
        return boolean_array.astype(np.int8)

"""
def test_code():

    # Assuming you have a 32-bit integer numpy array
    original_array = np.array([4294967295, 2147483647, 0, 8983218],
                              dtype=np.uint32)

    # Use bitwise AND operation to convert each integer to a boolean array
    boolean_array = (original_array[:, None] & (1 << np.arange(31, -1, -1))) != 0

    # Display the result
    print(boolean_array)
"""

@implements(proc=SolutionReadout, protocol=LoihiProtocol)
@requires(CPU)
class SolutionReadoutModel(AbstractSubProcessModel):
    """Model for the SolutionReadout process.

    """

    def __init__(self, proc):
        num_message_bits = proc.proc_params.get("num_message_bits")

        # Define the dense input layer
        num_variables = np.prod(proc.proc_params.get("shape"))
        num_spike_integrators = np.ceil(num_variables / num_message_bits).astype(int)

        weights = self._get_input_weights(num_vars=num_variables,
                                          num_spike_int=num_spike_integrators,
                                          num_spike_integrators=num_message_bits)

        self.synapses_in = Sparse(weights=weights,
                                  sign_mode=SignMode.EXCITATORY,
                                  num_weight_bits=1,
                                  num_message_bits=num_message_bits)

        self.spike_integrators = SpikeIntegrator(shape=(num_spike_integrators,))

        self.out_adapter_cost_integrator = eio.spike.NxToPyAdapter(
            shape=(1,),
            num_message_bits=num_message_bits)
        self.out_adapter_best_state = eio.spike.NxToPyAdapter(
            shape=(num_spike_integrators,),
            num_message_bits=num_message_bits)

        self.solution_receiver = SolutionReceiver(
            shape=(1,),
            best_cost_init = self.best_cost.get(),
            best_state_init = self.best_state.get(),
            best_timestep_init = self.best_timestep.get())

        # Connect the parent InPort to the InPort of the child-Process.
        proc.in_ports.states_in.connect(self.synapses_in.s_in)
        proc.in_ports.cost_integrator_in.connect(
            self.out_adapter_cost_integrator.inp)

        # Connect intermediate ports
        self.synapses_in.connect(self.spike_integrators.state_in)
        self.spike_integrators.state_out.connect(
            self.out_adapter_best_state.inp)
        self.out_adapter_best_state.out.connect(self.solution_receiver.state_in)

        self.out_adapter_cost_integrator.out.connect(
            self.solution_receiver.cost_integrator_in)

        # Create aliases for variables
        proc.vars.best_state.alias(self.solution_receiver.best_state)
        proc.vars.best_timestep.alias(self.solution_receiver.best_timestep)
        proc.vars.best_cost.alias(self.solution_receiver.best_cost)

    @staticmethod
    def _get_input_weights(num_vars, num_spike_int, num_vars_per_int):
        """To be verified. Deprecated due to efficiency"""
        weights = np.zeros((num_spike_int, num_vars), dtype=np.int8)
        for spike_integrator in range(num_spike_int - 1):
            variable_start = num_vars_per_int*spike_integrator
            weights[spike_integrator, variable_start:variable_start + num_vars_per_int] = 1

        # The last spike integrator might be connected by less than
        # num_vars_per_int neurons
        # This happens when mod(num_variables, num_vars_per_int) != 0
        weights[-1, num_vars_per_int*(spike_integrator + 1): -1] = 1

        return weights

    @staticmethod
    def _get_input_weights_index(num_vars, num_spike_int, num_vars_per_int):
        """To be verified"""
        weights = np.zeros((num_spike_int, num_vars), dtype=np.int8)

        # Compute the indices for setting the values to 1
        indices = np.arange(0, num_vars_per_int * (num_spike_int - 1), num_vars_per_int)

        # Set the values to 1 using array indexing
        weights[:num_spike_int-1, indices:indices + num_vars_per_int] = 1

        # Set the values for the last spike integrator
        weights[-1, num_vars_per_int * (num_spike_int - 1):num_vars] = 1

        return weights
