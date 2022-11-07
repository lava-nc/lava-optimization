# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import numpy as np
from lava.lib.optimization.solvers.generic.monitoring_processes \
    .solution_readout.process import SolutionReadout
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyAsyncProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol


@implements(SolutionReadout, protocol=AsyncProtocol)
@requires(CPU)
class SolutionReadoutPyModel(PyAsyncProcessModel):
    """CPU model for the SolutionReadout process.

    The process receives two types of messages, an updated cost and the state of
    the solver network representing the current candidate solution to an
    OptimizationProblem. Additionally, a target cost can be defined by the
    user, once this cost is reached by the solver network, this process
    will request the runtime service to pause execution.
    """
    solution: np.ndarray = LavaPyType(np.ndarray, np.int32, 32)
    solution_step: np.ndarray = LavaPyType(np.ndarray, np.int32, 32)
    read_solution: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32,
                                         precision=32)
    cost_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=32)
    req_stop_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32,
                                       precision=32)
    target_cost: int = LavaPyType(int, np.int32, 32)
    acknowledgement: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32,
                                            precision=32)
    min_cost: int = LavaPyType(int, np.int32, 32)

    def run_async(self):
        """Execute spiking phase, integrate input, update dynamics and send
        messages out."""
        raw_cost = self.cost_in.recv()
        self.acknowledgement.send(np.asarray([1]))
        req_stop = self.req_stop_in.recv()
        self.acknowledgement.send(np.asarray([1]))
        cost = [0]
        if raw_cost[0]:
            # The following casts cost as a signed 24-bit value (8 = 32 - 24)
            cost = (raw_cost.astype(np.int32) << 8) >> 8
        if cost[0]:
            raw_solution = self.read_solution.recv()
            # ToDo: The following way of reading out solutions only works
            #  when the solutions are binary, i.e., QUBO problems. It relies
            #  on the assumption that `solution' is the spiking history of the
            #  neurons solving a problem and picks the spiking history from 3
            #  timesteps ago, when the minimum cost was actually achieved.
            self.solution[:] = (raw_solution.astype(np.int8) >> 2) & 1
            self.min_cost = cost[0]
            if req_stop[0] != 0:
                self.solution_step = req_stop[0]
                print(f"Host: received a better solution: "
                      f"{self.solution} at "
                      f"step"
                      f" {self.solution_step}")
        if req_stop[0] == 0:
            self._req_pause = True

        # Post guard
        if self.min_cost is not None:
            if self.min_cost <= self.target_cost:
                print("Host: LMT notified network reached target cost:",
                      self.min_cost)
                # Post mgmt
                print("Host: stopping simulation at step:", self.solution_step)
                self._req_pause = True
