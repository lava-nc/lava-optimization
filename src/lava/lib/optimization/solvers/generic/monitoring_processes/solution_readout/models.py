# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import numpy as np
from lava.lib.optimization.solvers.generic.monitoring_processes \
    .solution_readout.process import SolutionReadout
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyRefPort
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
    read_solution: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32,
                                         precision=32)
    cost_in: PyRefPort = LavaPyType(PyInPort.VEC_DENSE, np.int32,
                                    precision=32)
    target_cost: int = LavaPyType(int, np.int32, 32)
    min_cost: int = None

    def post_guard(self):
        """Decide whether to run post management phase."""
        if self.min_cost is not None:
            if self.min_cost <= self.target_cost:
                print("Host: LMT notified network reached target cost:",
                      self.min_cost)
                return True
        return False

    def run_spk(self):
        """Execute spiking phase, integrate input, update dynamics and send
        messages out."""
        cost = self.cost_in.recv()
        print(f"Host: cost: {cost}, at step {self.time_step}")
        if cost[0]:
            self.solution[:] = self.read_solution.recv()
            self.min_cost = cost[0]
            print(f"Host: received a better solution: {self.solution} at step"
                  f" {self.time_step}")

    def run_post_mgmt(self):
        """Execute post management phase."""
        print("Host: stopping simulation at step:", self.time_step)
        self._req_pause = True
