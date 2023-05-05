#  Copyright (C) 2023 Battelle Memorial Institute
#  SPDX-License-Identifier: BSD-2-Clause
#  See: https://spdx.org/licenses/

import numpy as np
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.lib.optimization.solvers.lca.v1_neuron.process import V1Neuron


@implements(proc=V1Neuron, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyV1NeuronFloat(PyLoihiProcessModel):
    # This model might spike too frequently. Implement an accumulator if so.
    vth: float = LavaPyType(float, float)
    tau: float = LavaPyType(float, float)
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    v: np.ndarray = LavaPyType(np.ndarray, float)
    bias: np.ndarray = LavaPyType(np.ndarray, float)

    def run_spk(self):
        # Soft-threshold activation
        activation = np.maximum(np.abs(self.v) - self.vth, 0) * np.sign(self.v)
        bias = activation * self.tau if self.proc_params['two_layer'] else \
            self.bias
        self.v = self.v * (1 - self.tau) + self.a_in.recv() + bias
        self.s_out.send(activation)


@implements(proc=V1Neuron, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PyV1NeuronFixed(PyLoihiProcessModel):
    vth: int = LavaPyType(int, int)
    tau: int = LavaPyType(int, int)
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)
    v: np.ndarray = LavaPyType(np.ndarray, int)
    bias: np.ndarray = LavaPyType(np.ndarray, int)

    def run_spk(self):
        # Soft-threshold activation
        activation = np.maximum(np.abs(self.v) - self.vth, 0) * np.sign(self.v)
        bias = np.right_shift(activation * self.tau, 24) \
            if self.proc_params['two_layer'] else self.bias
        self.v = np.right_shift(self.v * (2 ** 24 - self.tau), 24) \
            + self.a_in.recv() + bias
        self.s_out.send(activation)
