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
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol

from lava.lib.optimization.solvers.generic.solution_receiver.process import \
    SolutionReceiver


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

    state_in: PyInPort = LavaPyType(
        PyInPort.VEC_DENSE, np.int32, precision=32
    )
    cost_integrator_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32,
                                              precision=32)

    def run_async(self):
        self.best_cost = self.cost_integrator_in.recv()

        self.timestep = self.cost_integrator_in.recv()

        compressed_states = self.state_in.recv()

        self.best_state = self._decompress_state(compressed_states)

    @staticmethod
    def _decompress_state(self, compressed_states):
        """Add info!"""

        boolean_array = (compressed_states[:, None] & (
                1 << np.arange(31, -1, -1))) != 0

        # reshape into a 1D array
        boolean_array.reshape(-1)

        return boolean_array.astype(np.int8)


def test_code():

    # Assuming you have a 32-bit integer numpy array
    original_array = np.array([4294967295, 2147483647, 0, 8983218],
                              dtype=np.uint32)

    # Use bitwise AND operation to convert each integer to a boolean array
    boolean_array = (original_array[:, None] & (1 << np.arange(31, -1, -1))) != 0

    # Display the result
    print(boolean_array)