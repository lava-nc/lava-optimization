# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import numpy as np
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort, PyRefPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.lib.optimization.solvers.generic.read_gate.process import ReadGate


@implements(ReadGate, protocol=LoihiProtocol)
@requires(CPU)
class ReadGatePyModel(PyLoihiProcessModel):
    """CPU model for the ReadGate process.

    The model verifies if better payload (cost) has been notified by the
    downstream processes, if so, it reads those processes state and sends out to
    the upstream process the new payload (cost) and the network state.
    """
    target_cost: int = LavaPyType(int, np.int32, 32)
    # best_solution: int = LavaPyType(int, np.int32, 32)
    cost_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32,
                                   precision=32)
    cost_out: PyOutPort = LavaPyType(
        PyOutPort.VEC_DENSE, np.int32, precision=32
    )
    solution_out: PyOutPort = LavaPyType(
        PyOutPort.VEC_DENSE, np.int32, precision=32
    )
    send_pause_request: PyOutPort = LavaPyType(
        PyOutPort.VEC_DENSE, np.int32, precision=32
    )
    solution_reader = LavaPyType(PyRefPort.VEC_DENSE, np.int32, precision=32)
    min_cost: int = None
    solution: np.ndarray = None

    def post_guard(self):
        """Decide whether to run post management phase."""
        return True if self.min_cost else False

    def run_spk(self):
        """Execute spiking phase, integrate input, update dynamics and
        send messages out."""
        cost = self.cost_in.recv()
        if cost[0]:
            self.min_cost = cost[0]
            self.cost_out.send(np.array([0]))
            # print(f"RDGT: Found a solution with cost: {self.min_cost} at step "
            #       f"{self.time_step}")
        elif self.solution is not None:
            timestep = - np.array([self.time_step])
            if self.min_cost <= self.target_cost:
                self._req_pause = True
            self.cost_out.send(np.array([self.min_cost]))
            self.send_pause_request.send(timestep)
            self.solution_out.send(self.solution)
            self.solution = None
            self.min_cost = None
        else:
            self.cost_out.send(np.array([0]))

    def run_post_mgmt(self):
        """Execute post management phase."""
        self.solution = self.solution_reader.read()
