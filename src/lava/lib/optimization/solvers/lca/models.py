#  Copyright (C) 2023 Battelle Memorial Institute
#  SPDX-License-Identifier: BSD-2-Clause
#  See: https://spdx.org/licenses/

import numpy as np

from lava.magma.core.model.sub.model import AbstractSubProcessModel

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.decorator import implements, tag
from lava.proc.dense.process import Dense

from lava.lib.optimization.solvers.lca.process import LCA1Layer, LCA2Layer
from lava.lib.optimization.solvers.lca.v1_neuron.process import V1Neuron
from lava.lib.optimization.solvers.lca.residual_neuron.process import \
    ResidualNeuron


@implements(proc=LCA2Layer, protocol=LoihiProtocol)
class LCA2LayerModel(AbstractSubProcessModel):
    def __init__(self, proc: LCA2Layer):
        threshold = proc.threshold.get()
        T = proc.tau.get()
        T_exp = proc.tau_exp.get()
        weights = proc.weights.get()
        weights_exp = proc.weights_exp.get()
        input_val = proc.input.get()
        spike_height = proc.spike_height.get()

        self.v1 = V1Neuron(shape=(weights.shape[0],), tau=T, tau_exp=T_exp,
                           vth=threshold, two_layer=True)
        # weight_exp shifted 8 bits for the weights, 6 for the v1 output.
        self.weights_T = Dense(weights=-weights.T, num_message_bits=24,
                               weight_exp=weights_exp)

        self.res = ResidualNeuron(shape=(weights.shape[1],),
                                  spike_height=spike_height, bias=input_val)

        self.weights = Dense(weights=(weights * T), num_message_bits=24,
                             weight_exp=weights_exp + T_exp)

        self.weights.a_out.connect(self.v1.a_in)
        self.res.s_out.connect(self.weights.s_in)

        self.weights_T.a_out.connect(self.res.a_in)
        self.v1.s_out.connect(self.weights_T.s_in)

        # Expose output and voltage
        self.v1.s_out.connect(proc.out_ports.v1)
        self.res.s_out.connect(proc.out_ports.res)
        proc.vars.voltage.alias(self.v1.vars.v)
        proc.vars.input.alias(self.res.bias)


@implements(proc=LCA1Layer, protocol=LoihiProtocol)
class LCA1LayerModel(AbstractSubProcessModel):
    def __init__(self, proc: LCA1Layer):
        threshold = proc.threshold.get()
        T = proc.tau.get()
        T_exp = proc.tau_exp.get()

        weights = proc.weights.get()
        weights_exp = proc.weights_exp.get()
        bias = proc.bias.get()

        self.v1 = V1Neuron(shape=(weights.shape[0],), tau=T, tau_exp=T_exp,
                           vth=threshold, bias=bias, two_layer=False)
        # weight_exp shifted 8 bits for the weights, 6 for the v1 output.
        self.weights = Dense(weights=weights, num_message_bits=24,
                             weight_exp=weights_exp)

        self.weights.a_out.connect(self.v1.a_in)
        self.v1.s_out.connect(self.weights.s_in)

        # Expose output and voltage
        self.v1.s_out.connect(proc.out_ports.v1)
        proc.vars.voltage.alias(self.v1.vars.v)
        proc.vars.bias.alias(self.v1.bias)
