# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.lib.optimization.solvers.generic.cost_integrator.process import (
    CostIntegrator,
)


@implements(proc=CostIntegrator, protocol=LoihiProtocol)
@requires(CPU)
class CostIntegratorModel(PyLoihiProcessModel):
    """CPU model for the CostIntegrator process.

    The process adds up local cost components from downstream units comming as
    spike payload. It has a cost_min variable which keeps track of the best
    cost seen so far, if the new cost is better, the minimum cost is updated
    and send as an output spike to an upstream process.
    Note that cost_min is divided into the first and last three bits.
    cost_min = cost_min_first << 24 + cost_min_last
    """

    cost_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    cost_out_last: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)
    cost_out_first: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)
    cost_min_last: np.ndarray = LavaPyType(np.ndarray, int, 24)
    cost_min_first: np.ndarray = LavaPyType(np.ndarray, int, 8)
    cost_last: np.ndarray = LavaPyType(np.ndarray, int, 24)
    cost_first: np.ndarray = LavaPyType(np.ndarray, int, 8)

    def run_spk(self):
        """Execute spiking phase, integrate input, update dynamics and send
        messages out."""
        cost = self.cost_in.recv()
        # Clip cost to 32bit
        np.clip(cost, -2 ** 31, 2 ** 31 - 1, out=cost)
        # Distribute into components
        cost = cost.astype(int)
        cost_last = cost & (2**24 - 1) # maintain last 24 bit
        cost_first = (cost & ((2**8 - 1) << 24)) >> 24 # maintain first 8 bit
        cost_first = np.array(cost_first).astype(np.int8) # signed int8
        if cost < (self.cost_min_first << 24) + self.cost_min_first:
            self.cost_min_first[:] = cost_first
            self.cost_min_last[:] = cost_last
            self.cost_out_last.send(cost_last)
            self.cost_out_first.send(cost_first)
        else:
            self.cost_out_last.send(np.asarray([0]))
            self.cost_out_first.send(np.asarray([0]))
        self.cost_last[:] = cost_last
        self.cost_first[:] = cost_first
