# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import numpy as np
from lava.lib.optimization.solvers.generic.monitoring_processes\
    .solution_readout.process import SolutionReadout
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol


@implements(SolutionReadout, protocol=LoihiProtocol)
@requires(CPU)
class SolutionReadoutPyModel(PyLoihiProcessModel):
    """CPU model for the SolutionReadout process.
    The process receives two types of messages, an updated cost and the
    state of
    the solver network representing the current candidate solution to an
    OptimizationProblem. Additionally, a target cost can be defined by the
    user, once this cost is reached by the solver network, this process
    will request the runtime service to pause execution.
    """

    solution: np.ndarray = LavaPyType(np.ndarray, np.int32, 32)
    solution_step: np.ndarray = LavaPyType(np.ndarray, np.int32, 32)
    time_steps_per_algorithmic_step: np.ndarray = LavaPyType(np.ndarray,
                                                             np.int32, 32)
    read_solution: PyInPort = LavaPyType(
        PyInPort.VEC_DENSE, np.int32, precision=32
    )
    cost_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=32)
    timestep_in: PyInPort = LavaPyType(
        PyInPort.VEC_DENSE, np.int32, precision=32
    )
    target_cost: int = LavaPyType(int, np.int32, 32)
    min_cost: np.ndarray = LavaPyType(np.ndarray, np.int32, 32)
    stop = False

    def run_spk(self):
        if self.stop:
            return
        raw_cost, min_cost_id = self.cost_in.recv()
        if raw_cost != 0:
            timestep, raw_solution = self._receive_data()
            cost = self.decode_cost(raw_cost)
            self.solution_step = abs(timestep)
            self.solution[:] = self.decode_solution(
                raw_solution=raw_solution,
                time_steps_per_algorithmic_step=self.
                time_steps_per_algorithmic_step)
            self.min_cost[:] = np.asarray([cost[0], min_cost_id])
            if cost[0] < 0:
                self._printout_new_solution(cost, min_cost_id, timestep)
            self._printout_if_converged()
            self._stop_if_requested(timestep, min_cost_id)

    def _receive_data(self):
        timestep = self.timestep_in.recv()[0]
        raw_solution = self.read_solution.recv()
        return timestep, raw_solution

    @staticmethod
    def decode_cost(raw_cost) -> np.ndarray:
        return np.array([raw_cost]).astype(np.int32)

    @staticmethod
    def decode_solution(raw_solution,
                        time_steps_per_algorithmic_step) -> np.ndarray:
        if time_steps_per_algorithmic_step == 1:
            raw_solution &= 0x1F  # AND with 0x1F (=0b11111) retains 5 LSBs
            # The binary solution was attained 2 steps ago. Shift down by 4.
            return raw_solution.astype(np.int8) >> 4
        elif time_steps_per_algorithmic_step == 2:
            raw_solution &= 0x7  # AND with 0x7 (=0b111) retains 3 LSBs
            # The binary solution was attained 1 step ago. Shift down by 4.
            return raw_solution.astype(np.int8) >> 2
        else:
            raise ValueError(f"The number of time steps that a single "
                             f"algorithmic step requires must be either 1 or "
                             f"2 but is {time_steps_per_algorithmic_step}.")

    def _printout_new_solution(self, cost, min_cost_id, timestep):
        self.log.info(
            f"Host: better solution found by network {min_cost_id} at "
            f"step {abs(timestep) - 2} "
            f"with cost {cost[0]}: {self.solution}"
        )

    def _printout_if_converged(self):
        if (
                self.min_cost[0] is not None
                and self.min_cost[0] <= self.target_cost
        ):
            self.log.info(
                f"Host: network reached target cost {self.target_cost}.")

    def _stop_if_requested(self, timestep, min_cost_id):
        if (timestep > 0 or timestep == -1) and min_cost_id != -1:
            self.stop = True
