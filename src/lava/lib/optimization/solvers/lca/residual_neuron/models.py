#  Copyright (C) 2023 Battelle Memorial Institute
#  SPDX-License-Identifier: BSD-2-Clause
#  See: https://spdx.org/licenses/

from numbers import Number
import numpy as np
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.lib.optimization.solvers.lca.residual_neuron.process import \
    ResidualNeuron


class AbstractPyResidual(PyLoihiProcessModel):
    spike_height: Number
    a_in: PyInPort
    s_out: PyOutPort
    v: np.ndarray
    bias: np.ndarray

    def run_spk(self):
        self.v += self.a_in.recv() + self.bias
        activation = np.abs(self.v) > self.spike_height
        self.s_out.send(np.where(activation, self.v, 0))
        self.v[activation] = 0


@implements(proc=ResidualNeuron, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyResidualFloat(AbstractPyResidual):
    spike_height: float = LavaPyType(float, float)
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    v: np.ndarray = LavaPyType(np.ndarray, float)
    bias: np.ndarray = LavaPyType(np.ndarray, float)


@implements(proc=ResidualNeuron, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PyResidualFixed(AbstractPyResidual):
    spike_height: int = LavaPyType(int, int)
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)
    v: np.ndarray = LavaPyType(np.ndarray, int)
    bias: np.ndarray = LavaPyType(np.ndarray, int)
