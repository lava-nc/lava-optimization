# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from processes import *


@implements(proc=constraintDirections, protocol=LoihiProtocol)
@requires(CPU)
class PyDenseModel(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    weights: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=8)

    def run_spk(self):
        s_in = self.s_in.recv()
        # matrix multiplication
        a_out = self.weights@s_in
        self.a_out.send(a_out)
        self.a_out.flush()

@implements(proc=constraintNeurons, protocol=LoihiProtocol)
@requires(CPU)
class PyDenseModel(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    threshold: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)

    def run_spk(self):
        s_in = self.s_in.recv()
        # constraint violation check
        a_out = (s_in-self.threshold)*(s_in < self.threshold)
        self.a_out.send(a_out)
        self.a_out.flush()

@implements(proc=quadraticConnectivity, protocol=LoihiProtocol)
@requires(CPU)
class PyDenseModel(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    weights: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=8)

    def run_spk(self):
        s_in = self.s_in.recv()
        #change this line
        a_out = self.weights@s_in
        self.a_out.send(a_out)
        self.a_out.flush()

@implements(proc=solutionNeurons, protocol=LoihiProtocol)
@requires(CPU)
class PyDenseModel(PyLoihiProcessModel):
    s_in_qc: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    a_out_qc: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, 
                                     precision=24)
    s_in_cn: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    a_out_cn: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, 
                                     precision=24)
    s_in_alpha: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, 
                                      precision=24)
    s_in_beta: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    a_out_cc: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, 
                                     precision=24)
    qp_neuron_state: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    grad_bias: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)

    def run_spk(self):
        s_in_qc = self.s_in_qc.recv()
        s_in_cn = self.s_in_cn.recv()
        s_in_alpha = self.s_in_alpha.recv()
        s_in_beta = self.s_in_beta.recv()
        self.qp_neuron_state = -s_in_alpha*(s_in_qc+self.grad_bias) \
                                -s_in_beta*s_in_cn
        a_out_cn = self.qp_neuron_state
        a_out_qc = self.qp_neuron_state
        a_out_cc = self.qp_neuron_state

        self.a_out_qc.send(a_out_qc)
        self.a_out_cn.send(a_out_cn)
        self.a_out_cc.send(a_out_cc)
        self.a_out_qc.flush()
        self.a_out_cn.flush()
        self.a_out_cc.flush()

@implements(proc=constraintNormals, protocol=LoihiProtocol)
@requires(CPU)
class PyDenseModel(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    weights: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=8)

    def run_spk(self):
        s_in = self.s_in.recv()
        # matrix multiplication
        a_out = self.weights@s_in
        self.a_out.send(a_out)
        self.a_out.flush()

@implements(proc=learningConstantAlpha, protocol=LoihiProtocol)
@requires(CPU)
class PyDenseModel(PyLoihiProcessModel):
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    alpha: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    decay_counter: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    alpha_decay_schedule: np.ndarray = LavaPyType(np.ndarray, np.int32, 
                                                  precision=24)
    def run_spk(self):
        self.decay_counter +=1
        if (self.decay_counter == self.alpha_decay_schedule):
            self.alpha = np.right_shift(self.alpha, 1)
            
            self.decay_counter = np.zeros(self.decay_counter.shape)
        a_out = self.alpha
        self.a_out.send(a_out)
        self.a_out.flush()

@implements(proc=learningConstantBeta, protocol=LoihiProtocol)
@requires(CPU)
class PyDenseModel(PyLoihiProcessModel):
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    beta: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    growth_counter: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    beta_growth_schedule: np.ndarray = LavaPyType(np.ndarray, np.int32, 
                                                  precision=24)
    def run_spk(self):
        self.growth_counter +=1
        if (self.growth_counter == self.beta_growth_schedule):
            self.beta = np.left_shift(self.beta, 1)
            # TODO: guard against shift overflows
            self.growth_counter = np.zeros(self.growth_counter.shape)
        a_out = self.beta
        self.a_out.send(a_out)
        self.a_out.flush()