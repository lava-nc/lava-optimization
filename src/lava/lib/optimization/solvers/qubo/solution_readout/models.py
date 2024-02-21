# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import numpy as np
import itertools
import typing as ty
import numpy.typing as npty

from lava.lib.optimization.solvers.qubo.solution_readout.process import (
    SolutionReadoutEthernet
)
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import (
    PyAsyncProcessModel
)
from lava.magma.core.model.py.ports import PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol

from lava.lib.optimization.solvers.qubo.solution_readout.process import (
    SolutionReceiver, SpikeIntegrator
)
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.sparse.process import Sparse
from scipy.sparse import csr_matrix
from abc import ABC, abstractmethod


@implements(SolutionReceiver, protocol=AsyncProtocol)
@requires(CPU)
class SolutionReceiverAbstractPyModel(PyAsyncProcessModel, ABC):
    """CPU model for the SolutionReadout process.
    This is the abstract class.
    The process receives two types of messages, an updated cost and the
    state of
    the solver network representing the current candidate solution to an
    OptimizationProblem. Additionally, a target cost can be defined by the
    user, once this cost is reached by the solver network, this process
    will request the runtime service to pause execution.
    """

    variables_1bit: np.ndarray = LavaPyType(np.ndarray, np.uint8, 1)
    variables_32bit: np.ndarray = LavaPyType(np.ndarray, np.int32, 32)
    num_message_bits: np.ndarray = LavaPyType(np.ndarray, np.int8, 32)
    timeout: np.ndarray = LavaPyType(np.ndarray, np.int32, 32)
    results_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, 32)

    @abstractmethod
    def run_async(self):
        pass

    @staticmethod
    def _decompress_state(compressed_states,
                          num_message_bits,
                          variables_1bit_num,
                          variables_32bit_num):
        """Receives the output of a recv from SolutionReadout, and extracts
        32bit and 1bit variables!"""

        variables_32bit = compressed_states[:variables_32bit_num].astype(
            np.int32)

        variables_1bit = (compressed_states[variables_32bit_num:, None] & (
            1 << np.arange(0, num_message_bits))) != 0

        # reshape into a 1D array
        variables_1bit.reshape(-1)
        # If n_vars is not a multiple of num_message_bits, then last entries
        # must be cut off
        variables_1bit = variables_1bit.astype(
            np.int8).flatten()[:variables_1bit_num]

        return variables_32bit, variables_1bit


@implements(SolutionReceiver, protocol=AsyncProtocol)
@requires(CPU)
class SolutionReceiverQUBOPyModel(SolutionReceiverAbstractPyModel):
    """CPU model for the SolutionReadout process.
    This model is specific for the QUBO Solver.

    See docstring of parent class for more information
    """

    def run_async(self):

        # Get required user input
        num_message_bits = self.num_message_bits[0]
        variables_1bit_num = self.variables_1bit.shape[0]
        variables_32bit_num = self.variables_32bit.shape[0]
        timeout = self.timeout[0]

        # Set default values, required only if the Process will be restarted
        self.variables_32bit[1] = 1
        self.variables_32bit[0] = 0
        self.variables_1bit[:] = 0

        # Iterating for timeout - 1 because an additional step is used to
        # recv the state
        while True:
            results_buffer = self.results_in.recv()

            if self._check_if_input(results_buffer):
                break

        results_buffer, _ = self._decompress_state(
            compressed_states=results_buffer,
            num_message_bits=num_message_bits,
            variables_1bit_num=variables_1bit_num,
            variables_32bit_num=variables_32bit_num)
        self.variables_32bit = results_buffer

        # best states are returned with a delay of 1 timestep
        results_buffer = self.results_in.recv()
        _, results_buffer = self._decompress_state(
            compressed_states=results_buffer,
            num_message_bits=num_message_bits,
            variables_1bit_num=variables_1bit_num,
            variables_32bit_num=variables_32bit_num)
        self.variables_1bit = results_buffer

        print("==============================================================")
        print("Solution found!")
        print(f"{self.variables_32bit=}")
        print(f"{self.variables_1bit=}")
        print("==============================================================")

    @staticmethod
    def _check_if_input(results_buffer) -> bool:
        """For QUBO, we know that the readout starts as soon as the 2nd output
        (best_timestep) is > 0."""

        return results_buffer[1] > 0

    @staticmethod
    def postprocess_variables_32bit(
        variables_32bit,
        timeout,
    ) -> ty.Tuple[int, int]:
        best_cost = variables_32bit[0]
        best_timestep = variables_32bit[1]
        best_timestep = timeout - best_timestep - 3
        return best_cost, best_timestep


@implements(proc=SolutionReadoutEthernet, protocol=LoihiProtocol)
@requires(CPU)
class SolutionReadoutEthernetModel(AbstractSubProcessModel):
    """Model for the SolutionReadout process."""

    def __init__(self, proc):
        num_message_bits = proc.proc_params.get("num_message_bits")

        timeout = proc.proc_params.get("timeout")

        variables_1bit_num = proc.variables_1bit.shape[0]
        variables_32bit_num = proc.variables_32bit.shape[0]
        num_spike_integrators = proc.proc_params.get("num_spike_integrators")

        connection_config = proc.proc_params.get("connection_config")

        self.spike_integrators = SpikeIntegrator(shape=(num_spike_integrators,))

        # Connect the 1bit binary neurons

        weights_variables_1bit_0_in = self._get_input_weights(
            variables_1bit_num=variables_1bit_num,
            variables_32bit_num=variables_32bit_num,
            num_spike_int=num_spike_integrators,
            num_1bit_vars_per_int=num_message_bits,
            weight_exp=0
        )
        self.synapses_variables_1bit_0_in = Sparse(
            weights=weights_variables_1bit_0_in,
            num_weight_bits=8,
            num_message_bits=num_message_bits,
            weight_exp=0,
        )

        proc.in_ports.variables_1bit_in.connect(
            self.synapses_variables_1bit_0_in.s_in)
        self.synapses_variables_1bit_0_in.a_out.connect(
            self.spike_integrators.a_in)

        if variables_1bit_num > 8:
            weights_variables_1bit_1_in = self._get_input_weights(
                variables_1bit_num=variables_1bit_num,
                variables_32bit_num=variables_32bit_num,
                num_spike_int=num_spike_integrators,
                num_1bit_vars_per_int=num_message_bits,
                weight_exp=8
            )
            self.synapses_variables_1bit_1_in = Sparse(
                weights=weights_variables_1bit_1_in,
                num_weight_bits=8,
                num_message_bits=num_message_bits,
                weight_exp=8,
            )

            proc.in_ports.variables_1bit_in.connect(
                self.synapses_variables_1bit_1_in.s_in)
            self.synapses_variables_1bit_1_in.a_out.connect(
                self.spike_integrators.a_in)

        if variables_1bit_num > 16:
            weights_variables_1bit_2_in = self._get_input_weights(
                variables_1bit_num=variables_1bit_num,
                variables_32bit_num=variables_32bit_num,
                num_spike_int=num_spike_integrators,
                num_1bit_vars_per_int=num_message_bits,
                weight_exp=16
            )
            self.synapses_variables_1bit_2_in = Sparse(
                weights=weights_variables_1bit_2_in,
                num_weight_bits=8,
                num_message_bits=num_message_bits,
                weight_exp=16,
            )

            proc.in_ports.variables_1bit_in.connect(
                self.synapses_variables_1bit_2_in.s_in)
            self.synapses_variables_1bit_2_in.a_out.connect(
                self.spike_integrators.a_in)

        if variables_1bit_num > 24:
            weights_variables_1bit_3_in = self._get_input_weights(
                variables_1bit_num=variables_1bit_num,
                variables_32bit_num=variables_32bit_num,
                num_spike_int=num_spike_integrators,
                num_1bit_vars_per_int=num_message_bits,
                weight_exp=24
            )
            self.synapses_variables_1bit_3_in = Sparse(
                weights=weights_variables_1bit_3_in,
                num_weight_bits=8,
                num_message_bits=num_message_bits,
                weight_exp=24,
            )
            proc.in_ports.variables_1bit_in.connect(
                self.synapses_variables_1bit_3_in.s_in)
            self.synapses_variables_1bit_3_in.a_out.connect(
                self.spike_integrators.a_in)

        # Connect the 32bit InPorts, one by one
        for ii in range(variables_32bit_num):
            # Create the synapses for InPort ii as self.
            synapses_in = Sparse(
                weights=self._get_32bit_in_weights(
                    num_spike_int=num_spike_integrators,
                    var_index=ii),
                num_weight_bits=8,
                num_message_bits=32,)
            setattr(self, f"synapses_variables_32bit_{ii}_in", synapses_in)

            getattr(proc.in_ports,
                    f"variables_32bit_{ii}_in").connect(synapses_in.s_in)
            synapses_in.a_out.connect(self.spike_integrators.a_in)

        # Define and connect the SolutionReceiver
        self.solution_receiver = SolutionReceiver(
            shape=(1,),
            timeout=timeout,
            variables_1bit_num=variables_1bit_num,
            variables_1bit_init=proc.variables_1bit.get(),
            variables_32bit_num=variables_32bit_num,
            variables_32bit_init=proc.variables_32bit.get(),
            num_spike_integrators=num_spike_integrators,
            num_message_bits=num_message_bits,
        )

        self.spike_integrators.s_out.connect(
            self.solution_receiver.results_in, connection_config)

        # Create aliases for variables
        proc.vars.variables_1bit.alias(self.solution_receiver.variables_1bit)
        proc.vars.variables_32bit.alias(self.solution_receiver.variables_32bit)
        proc.vars.timeout.alias(self.solution_receiver.timeout)

    @staticmethod
    def _get_input_weights(variables_1bit_num,
                           variables_32bit_num,
                           num_spike_int,
                           num_1bit_vars_per_int,
                           weight_exp) -> csr_matrix:
        """Builds weight matrices from 1bit variable neurons to
        SpikeIntegrators. For this, num_spike_int binary neurons are bundled
        and converge onto 1 SpikeIntegrator. For efficiency reasons, this
        function may get vectorized in the future."""

        weights = np.zeros((num_spike_int, variables_1bit_num), dtype=np.uint8)

        # The first SpikeIntegrators receive 32bit variables
        for spike_integrator_id in range(variables_32bit_num,
                                         num_spike_int - 1):
            variable_start = num_1bit_vars_per_int * (
                spike_integrator_id - variables_32bit_num) + weight_exp
            weights[spike_integrator_id,
                    variable_start:variable_start + 8] = np.power(2,
                                                                  np.arange(8))
        # The last spike integrator might be connected by less than
        # num_1bit_vars_per_int neurons
        # This happens when mod(num_variables, num_1bit_vars_per_int) != 0
        variable_start = num_1bit_vars_per_int * (
            num_spike_int - variables_32bit_num - 1) + weight_exp
        weights[-1, variable_start:] = np.power(2, np.arange(weights.shape[1]
                                                             - variable_start))

        return csr_matrix(weights)

    @staticmethod
    def _get_32bit_in_weights(num_spike_int: int, var_index: int) -> csr_matrix:

        data = [1]
        row = [var_index]
        col = [0]

        return csr_matrix((data, (row, col)),
                          shape=(num_spike_int, 1),
                          dtype=np.int8)
