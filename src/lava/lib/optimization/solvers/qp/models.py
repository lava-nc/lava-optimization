# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

"""
Implement behaviors (models) of the processes defined in processes.py
For further documentation please refer to processes.py
"""
import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.lib.optimization.solvers.qp.processes import (
    ConstraintDirections,
    ConstraintCheck,
    ConstraintNeurons,
    ConstraintNormals,
    QuadraticConnectivity,
    SolutionNeurons,
    GradientDynamics,
)


@implements(proc=ConstraintDirections, protocol=LoihiProtocol)
@requires(CPU)
class PyCDModel(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    weights: np.ndarray = LavaPyType(np.ndarray, float)

    def run_spk(self):
        s_in = self.s_in.recv()
        # process behavior: matrix multiplication
        a_out = self.weights @ s_in
        self.a_out.send(a_out)


@implements(proc=ConstraintNeurons, protocol=LoihiProtocol)
@requires(CPU)
class PyCNeuModel(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)
    thresholds: np.ndarray = LavaPyType(np.ndarray, np.float64)

    def run_spk(self):
        s_in = self.s_in.recv()
        # process behavior: constraint violation check
        a_out = (s_in - self.thresholds) * (s_in > self.thresholds)
        self.a_out.send(a_out)


@implements(proc=QuadraticConnectivity, protocol=LoihiProtocol)
@requires(CPU)
class PyQCModel(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)
    weights: np.ndarray = LavaPyType(np.ndarray, np.float64)

    def run_spk(self):
        s_in = self.s_in.recv()
        # process behavior: matrix multiplication
        a_out = self.weights @ s_in
        self.a_out.send(a_out)


@implements(proc=SolutionNeurons, protocol=LoihiProtocol)
@requires(CPU)
class PySNModel(PyLoihiProcessModel):
    s_in_qc: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    a_out_qc: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)
    s_in_cn: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    a_out_cc: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)
    qp_neuron_state: np.ndarray = LavaPyType(np.ndarray, np.float64)
    grad_bias: np.ndarray = LavaPyType(np.ndarray, np.float64)
    alpha: np.ndarray = LavaPyType(np.ndarray, np.float64)
    beta: np.ndarray = LavaPyType(np.ndarray, np.float64)
    alpha_decay_schedule: int = LavaPyType(int, np.int32)
    beta_growth_schedule: int = LavaPyType(int, np.int32)
    decay_counter: int = LavaPyType(int, np.int32)
    growth_counter: int = LavaPyType(int, np.int32)

    def run_spk(self):
        a_out = self.qp_neuron_state
        self.a_out_cc.send(a_out)
        self.a_out_qc.send(a_out)

        s_in_qc = self.s_in_qc.recv()
        s_in_cn = self.s_in_cn.recv()

        self.decay_counter += 1
        if self.decay_counter == self.alpha_decay_schedule:
            # TODO: guard against shift overflows in fixed-point
            self.alpha = np.right_shift(self.alpha, 1)
            self.decay_counter = np.zeros(self.decay_counter.shape)

        self.growth_counter += 1
        if self.growth_counter == self.beta_growth_schedule:
            self.beta = np.left_shift(self.beta, 1)
            # TODO: guard against shift overflows in fixed-point
            self.growth_counter = np.zeros(self.growth_counter.shape)

        # process behavior: gradient update
        self.qp_neuron_state += (
            -self.alpha * (s_in_qc + self.grad_bias) - self.beta * s_in_cn
        )


@implements(proc=ConstraintNormals, protocol=LoihiProtocol)
@requires(CPU)
class PyCNorModel(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)
    weights: np.ndarray = LavaPyType(np.ndarray, np.float64)

    def run_spk(self):
        s_in = self.s_in.recv()
        # process behavior: matrix multiplication
        a_out = self.weights @ s_in
        self.a_out.send(a_out)


@implements(proc=ConstraintCheck, protocol=LoihiProtocol)
class SubCCModel(AbstractSubProcessModel):
    """Implement constraintCheckProcess behavior via sub Processes."""

    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    constraint_matrix: np.ndarray = LavaPyType(np.ndarray, np.float64)
    constraint_bias: np.ndarray = LavaPyType(np.ndarray, np.float64)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)

    def __init__(self, proc):
        """Builds sub Process structure of the Process."""
        constraint_matrix = proc.init_args.get("constraint_matrix", 0)
        constraint_bias = proc.init_args.get("constraint_bias", 0)

        # Initialize subprocesses
        self.constraintDirections = ConstraintDirections(
            shape=constraint_matrix.shape,
            constraint_directions=constraint_matrix,
        )
        self.constraintNeurons = ConstraintNeurons(
            shape=constraint_bias.shape, thresholds=constraint_bias
        )

        # connect subprocesses to obtain required process behavior
        proc.in_ports.s_in.connect(self.constraintDirections.in_ports.s_in)
        self.constraintDirections.out_ports.a_out.connect(
            self.constraintNeurons.in_ports.s_in
        )
        self.constraintNeurons.out_ports.a_out.connect(proc.out_ports.a_out)

        # alias process variables to subprocess variables
        proc.vars.constraint_matrix.alias(
            self.constraintDirections.vars.weights
        )
        proc.vars.constraint_bias.alias(self.constraintNeurons.vars.thresholds)


@implements(proc=GradientDynamics, protocol=LoihiProtocol)
class SubGDModel(AbstractSubProcessModel):
    """Implement gradientDynamics Process behavior via sub Processes."""

    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    hessian: np.ndarray = LavaPyType(np.ndarray, np.float64)
    constraint_matrix_T: np.ndarray = LavaPyType(
        np.ndarray,
        np.float64,
    )
    grad_bias: np.ndarray = LavaPyType(np.ndarray, np.float64)
    qp_neuron_state: np.ndarray = LavaPyType(np.ndarray, np.float64)
    alpha: np.ndarray = LavaPyType(np.ndarray, np.float64)
    beta: np.ndarray = LavaPyType(np.ndarray, np.float64)
    alpha_decay_schedule: int = LavaPyType(int, np.int32)
    beta_growth_schedule: int = LavaPyType(int, np.int32)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)

    def __init__(self, proc):
        """Builds sub Process structure of the Process."""
        hessian = proc.init_args.get("hessian", 0)
        shape_hess = hessian.shape
        shape_sol = (shape_hess[0], 1)
        constraint_matrix_T = proc.init_args.get("constraint_matrix_T", 0)
        shape_constraint_matrix_T = constraint_matrix_T.shape
        grad_bias = proc.init_args.get("grad_bias", np.zeros(shape_sol))
        qp_neuron_i = proc.init_args.get(
            "qp_neurons_init", np.zeros(shape_sol)
        )
        alpha = proc.init_args.get("alpha", np.ones(shape_sol))
        beta = proc.init_args.get("beta", np.ones(shape_sol))
        a_d = proc.init_args.get("alpha_decay_schedule", 10000)
        b_g = proc.init_args.get("beta_decay_schedule", 10000)

        # Initialize subprocesses
        self.qC = QuadraticConnectivity(shape=shape_hess, hessian=hessian)
        self.sN = SolutionNeurons(
            shape=shape_sol,
            qp_neurons_init=qp_neuron_i,
            grad_bias=grad_bias,
            alpha=alpha,
            beta=beta,
            alpha_decay_schedule=a_d,
            beta_growth_schedule=b_g,
        )
        self.cN = ConstraintNormals(
            shape=shape_constraint_matrix_T,
            constraint_normals=constraint_matrix_T,
        )

        # connect subprocesses to obtain required process behavior
        proc.in_ports.s_in.connect(self.cN.in_ports.s_in)
        self.cN.out_ports.a_out.connect(self.sN.in_ports.s_in_cn)
        self.sN.out_ports.a_out_qc.connect(self.qC.in_ports.s_in)
        self.qC.out_ports.a_out.connect(self.sN.in_ports.s_in_qc)
        self.sN.out_ports.a_out_cc.connect(proc.out_ports.a_out)

        # alias process variables to subprocess variables
        proc.vars.hessian.alias(self.qC.vars.weights)
        proc.vars.constraint_matrix_T.alias(self.cN.vars.weights)
        proc.vars.grad_bias.alias(self.sN.vars.grad_bias)
        proc.vars.qp_neuron_state.alias(self.sN.vars.qp_neuron_state)
        proc.vars.alpha.alias(self.sN.vars.alpha)
        proc.vars.beta.alias(self.sN.vars.beta)
        proc.vars.alpha_decay_schedule.alias(self.sN.vars.alpha_decay_schedule)
        proc.vars.beta_growth_schedule.alias(self.sN.vars.beta_growth_schedule)
