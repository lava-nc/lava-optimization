# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import numpy as np
from lava.lib.optimization.solvers.generic.monitoring_processes \
    .solution_readout.process import SolutionReadout
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort, PyRefPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol


@implements(SolutionReadout, protocol=LoihiProtocol)
@requires(CPU)
class SolutionReadoutPyModel(PyLoihiProcessModel):
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

    def run_spk(self):
        raw_cost = self.cost_in.recv()

        if raw_cost[0]:
            req_stop = self.req_stop_in.recv()
            # The following casts cost as a signed 24-bit value (8 = 32 - 24)
            cost = (raw_cost.astype(np.int32) << 8) >> 8
            raw_solution = self.read_solution.recv()
            self.solution[:] = (raw_solution.astype(np.int8) >> 2) & 1
            self.min_cost = cost[0]
            self.solution_step = abs(req_stop[0])
            print(f"Host: received a better solution at step {req_stop[0]}"
                  f" with cost {cost[0]}: {self.solution}")

            if self.min_cost is not None and self.min_cost <= self.target_cost:
                print(f"Host: network reached target cost {self.target_cost}.")

            if req_stop[0] > 0:
                self._req_pause = True
