# Copyright (C) 2021-2022 Intel Corporation
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
from lava.lib.optimization.solvers.generic.qp.processes import (
    ConstraintCheck,
    ConstraintNeurons,
    SolutionNeurons,
    GradientDynamics,
    ProjectedGradientNeuronsPIPGeq,
    ProportionalIntegralNeuronsPIPGeq,
    SigmaNeurons,
    DeltaNeurons,
    QPDense,
)


@implements(proc=QPDense, protocol=LoihiProtocol)
@requires(CPU)
class PyQPDenseModel(PyLoihiProcessModel):
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
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)
    thresholds: np.ndarray = LavaPyType(np.ndarray, np.float64)

    def run_spk(self):
        a_in = self.a_in.recv()
        # process behavior: constraint violation check
        s_out = (a_in - self.thresholds) * (a_in > self.thresholds)
        self.s_out.send(s_out)


@implements(proc=SolutionNeurons, protocol=LoihiProtocol)
@requires(CPU)
class PySNModel(PyLoihiProcessModel):
    a_in_qc: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    s_out_qc: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)
    a_in_cn: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    s_out_cc: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)
    qp_neuron_state: np.ndarray = LavaPyType(np.ndarray, np.float64)
    grad_bias: np.ndarray = LavaPyType(np.ndarray, np.float64)
    alpha: np.ndarray = LavaPyType(np.ndarray, np.float64)
    beta: np.ndarray = LavaPyType(np.ndarray, np.float64)
    alpha_decay_schedule: int = LavaPyType(int, np.int32)
    beta_growth_schedule: int = LavaPyType(int, np.int32)
    decay_counter: int = LavaPyType(int, np.int32)
    growth_counter: int = LavaPyType(int, np.int32)

    def run_spk(self):
        s_out = self.qp_neuron_state
        self.s_out_cc.send(s_out)
        self.s_out_qc.send(s_out)

        a_in_qc = self.a_in_qc.recv()
        a_in_cn = self.a_in_cn.recv()

        self.decay_counter += 1
        if self.decay_counter == self.alpha_decay_schedule:
            # TODO: guard against shift overflows in fixed-point
            self.alpha = self.alpha / 2
            self.decay_counter = 0

        self.growth_counter += 1
        if self.growth_counter == self.beta_growth_schedule:
            self.beta = self.beta * 2
            # TODO: guard against shift overflows in fixed-point
            self.growth_counter = 0

        # process behavior: gradient update
        self.qp_neuron_state += (
            -self.alpha * (a_in_qc + self.grad_bias) - self.beta * a_in_cn
        )


@implements(proc=ConstraintCheck, protocol=LoihiProtocol)
class SubCCModel(AbstractSubProcessModel):
    """Implement constraintCheckProcess behavior via sub Processes."""

    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    constraint_matrix: np.ndarray = LavaPyType(np.ndarray, np.float64)
    constraint_bias: np.ndarray = LavaPyType(np.ndarray, np.float64)
    x_internal: np.ndarray = LavaPyType(np.ndarray, np.float64)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)

    def __init__(self, proc):
        """Builds sub Process structure of the Process."""

        constraint_matrix = proc.proc_params["con_mat"]
        constraint_bias = proc.proc_params["con_bias"]
        x_int_init = proc.proc_params["x_init"]
        sparse = proc.proc_params["sparsity"]

        # Initialize subprocesses
        self.constraintDirections = QPDense(
            shape=constraint_matrix.shape,
            weights=constraint_matrix,
        )
        self.constraintNeurons = ConstraintNeurons(
            shape=constraint_bias.shape, thresholds=constraint_bias
        )

        if sparse:
            print("[INFO]: Using additional Sigma layer")
            self.sigmaNeurons = SigmaNeurons(
                shape=(constraint_matrix.shape[1], 1), x_sig_init=x_int_init
            )

            # proc.vars.x_internal.alias(self.sigmaNeurons.vars.x_internal)
            # connect subprocesses to obtain required process behavior
            proc.in_ports.s_in.connect(self.sigmaNeurons.in_ports.s_in)
            self.sigmaNeurons.out_ports.s_out.connect(
                self.constraintDirections.in_ports.s_in
            )

        else:
            proc.in_ports.s_in.connect(self.constraintDirections.in_ports.s_in)

        # remaining procesess to connect irrespective of sparsity
        self.constraintDirections.out_ports.a_out.connect(
            self.constraintNeurons.in_ports.a_in
        )
        self.constraintNeurons.out_ports.s_out.connect(proc.out_ports.s_out)

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
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)

    def __init__(self, proc):
        """Builds sub Process structure of the Process."""
        hessian = proc.proc_params["hess"]
        shape_hess = hessian.shape
        shape_sol = (shape_hess[0], 1)
        constraint_matrix_T = proc.proc_params["con_mat_T"]
        shape_constraint_matrix_T = constraint_matrix_T.shape
        grad_bias = proc.proc_params["grd_bs"]

        qp_neuron_i = proc.proc_params["qp_neur_init"]
        sparse = proc.proc_params["sparsity"]
        model = proc.proc_params["mod"]
        theta = proc.proc_params["thet"]
        alpha = proc.proc_params["al"]
        beta = proc.proc_params["bet"]
        t_d = proc.proc_params["thet_dec"]
        a_d = proc.proc_params["al_dec"]
        b_g = proc.proc_params["bet_dec"]

        # Initialize subprocesses
        self.qC = QPDense(shape=shape_hess, weights=hessian)

        self.cN = QPDense(
            shape=shape_constraint_matrix_T,
            weights=constraint_matrix_T,
        )

        if sparse:
            if model == "SigDel":
                print("[INFO]: Using Sigma Delta Solution Neurons")
                self.sN = SolutionNeurons(
                    shape=shape_sol,
                    qp_neurons_init=qp_neuron_i,
                    grad_bias=grad_bias,
                    alpha=alpha,
                    beta=beta,
                    alpha_decay_schedule=a_d,
                    beta_growth_schedule=b_g,
                )

                self.sigmaNeurons = SigmaNeurons(
                    shape=shape_sol, x_sig_init=qp_neuron_i
                )

                self.deltaNeurons = DeltaNeurons(
                    shape=shape_sol,
                    x_del_init=qp_neuron_i,
                    theta=theta,
                    theta_decay_schedule=t_d,
                )

                proc.vars.theta.alias(self.deltaNeurons.vars.theta)
                proc.vars.theta_decay_schedule.alias(
                    self.deltaNeurons.vars.theta_decay_schedule
                )

                # connection processes and aliases
                self.sN.out_ports.s_out_qc.connect(
                    self.deltaNeurons.in_ports.s_in
                )
                self.deltaNeurons.out_ports.s_out.connect(
                    self.sigmaNeurons.in_ports.s_in
                )
                self.deltaNeurons.out_ports.s_out.connect(proc.out_ports.s_out)
                self.sigmaNeurons.out_ports.s_out.connect(
                    self.qC.in_ports.s_in
                )
            if model == "TLIF":
                raise ValueError("TLIF implementation coming soon!")
        else:
            print("[INFO]: Using Dense Solution Neurons")
            self.sN = SolutionNeurons(
                shape=shape_sol,
                qp_neurons_init=qp_neuron_i,
                grad_bias=grad_bias,
                alpha=alpha,
                beta=beta,
                alpha_decay_schedule=a_d,
                beta_growth_schedule=b_g,
            )
            self.sN.out_ports.s_out_qc.connect(self.qC.in_ports.s_in)
            self.sN.out_ports.s_out_cc.connect(proc.out_ports.s_out)

        # connect subprocesses to obtain required process behavior
        proc.in_ports.s_in.connect(self.cN.in_ports.s_in)
        self.cN.out_ports.a_out.connect(self.sN.in_ports.a_in_cn)
        self.qC.out_ports.a_out.connect(self.sN.in_ports.a_in_qc)

        # alias process variables to subprocess variables
        proc.vars.hessian.alias(self.qC.vars.weights)
        proc.vars.constraint_matrix_T.alias(self.cN.vars.weights)
        proc.vars.grad_bias.alias(self.sN.vars.grad_bias)
        proc.vars.qp_neuron_state.alias(self.sN.vars.qp_neuron_state)
        proc.vars.alpha.alias(self.sN.vars.alpha)
        proc.vars.beta.alias(self.sN.vars.beta)
        proc.vars.alpha_decay_schedule.alias(self.sN.vars.alpha_decay_schedule)
        proc.vars.beta_growth_schedule.alias(self.sN.vars.beta_growth_schedule)


@implements(proc=ProjectedGradientNeuronsPIPGeq, protocol=LoihiProtocol)
@requires(CPU)
class PyProjGradPIPGeqModel(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)
    qp_neuron_state: np.ndarray = LavaPyType(np.ndarray, np.float64)
    grad_bias: np.ndarray = LavaPyType(np.ndarray, np.float64)
    alpha: np.ndarray = LavaPyType(np.ndarray, np.float64)
    alpha_decay_schedule: int = LavaPyType(int, np.int32)

    # for bit-accurate implmentation
    da_exp: int = LavaPyType(int, int)
    grad_bias_man: np.ndarray = LavaPyType(np.ndarray, int)
    grad_bias_exp: int = LavaPyType(int, int)
    alpha_man: np.ndarray = LavaPyType(np.ndarray, int)
    alpha_exp: int = LavaPyType(int, int)
    decay_inter: int = LavaPyType(int, int)
    decay_index: int = LavaPyType(int, int)
    decay_factor: int = LavaPyType(int, int)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)
        self.lr_decay_type = self.proc_params["lr_dec_type"]
        (
            self.decay_index,
            self.decay_interval,
            self.decay_factor,
        ) = self.proc_params["alpha_dec_params"]
        self.alpha_decay_indices = self.proc_params["alpha_dec_list"]
        self.decay_counter = 0
        # variable to store incoming spike at odd timestep to emulate hardware
        # behavior
        self.connectivity_spike = 0

    def run_spk(self):
        self.decay_counter += 1
        a_in = self.a_in.recv()
        if self.decay_counter % 2 == 1:
            # spike for qp_neuron_state coming in from quadratic connectivity
            # in hardware is simulated here
            self.connectivity_spike = a_in
        if self.decay_counter % 2 == 0:
            if self.lr_decay_type == "schedule":
                if self.decay_counter == self.alpha_decay_schedule:
                    self.alpha = self.alpha / 2
                    self.decay_counter = 0

            if self.lr_decay_type == "indices":
                if self.decay_counter in self.alpha_decay_indices:
                    self.alpha = self.alpha / 2

            if self.lr_decay_type == "computed_schedule":
                if self.decay_counter == self.decay_index:
                    self.alpha = self.alpha / 2
                    self.decay_factor += 1
                    self.decay_index = (
                        self.decay_index
                        + self.decay_interval * self.decay_factor
                    )

            # process behavior: gradient update
            self.qp_neuron_state -= self.alpha * (
                a_in + self.grad_bias + self.connectivity_spike
            )
            self.s_out.send(self.qp_neuron_state)
        else:
            self.s_out.send(np.zeros(self.qp_neuron_state.shape))


@implements(proc=ProportionalIntegralNeuronsPIPGeq, protocol=LoihiProtocol)
@requires(CPU)
class PyPIneurPIPGeqModel(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)
    constraint_neuron_state: np.ndarray = LavaPyType(np.ndarray, np.float64)
    constraint_bias: np.ndarray = LavaPyType(np.ndarray, np.float64)
    beta: np.ndarray = LavaPyType(np.ndarray, np.float64)
    beta_growth_schedule: int = LavaPyType(int, np.int32)

    # for bit-accurate implmentation
    da_exp: int = LavaPyType(int, int)
    beta_exp: int = LavaPyType(int, int)
    con_bias_exp: int = LavaPyType(int, int)
    con_bias_man: np.ndarray = LavaPyType(np.ndarray, int)
    beta_man: np.ndarray = LavaPyType(np.ndarray, int)
    growth_inter: int = LavaPyType(int, int)
    growth_index: int = LavaPyType(int, int)
    growth_factor: int = LavaPyType(int, int)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)
        self.lr_growth_type = self.proc_params["lr_grw_type"]
        (
            self.growth_index,
            self.growth_factor,
        ) = self.proc_params["beta_grw_params"]
        self.beta_growth_indices = self.proc_params["beta_grw_list"]
        self.growth_counter = 0

    def run_spk(self):
        self.growth_counter += 1
        a_in = self.a_in.recv()

        if self.growth_counter % 2 == 1:
            if self.lr_growth_type == "schedule":
                if self.growth_counter == self.beta_growth_schedule:
                    self.beta = self.beta * 2
                    self.growth_counter = 0

            if self.lr_growth_type == "indices":
                if self.growth_counter in self.beta_growth_indices:
                    self.beta = self.beta * 2

            if self.lr_growth_type == "computed_schedule":
                if self.growth_counter == self.growth_index:
                    self.beta = self.beta * 2
                    self.growth_index = (
                        self.growth_index + 2 * self.growth_factor
                    )
                    self.growth_factor *= 2

            # process behavior:
            omega = self.beta * (
                a_in - self.constraint_bias
            )
            self.constraint_neuron_state += omega
            gamma = self.constraint_neuron_state + omega
            self.s_out.send(gamma)
        else:
            self.s_out.send(np.zeros(self.constraint_neuron_state.shape))


@implements(proc=SigmaNeurons, protocol=LoihiProtocol)
@requires(CPU)
class PySigNeurModel(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)
    x_internal: np.ndarray = LavaPyType(np.ndarray, np.float64)

    def run_spk(self):
        s_in = self.s_in.recv()
        self.x_internal += s_in
        s_out = self.x_internal
        self.s_out.send(s_out)


@implements(proc=DeltaNeurons, protocol=LoihiProtocol)
@requires(CPU)
class PyDelNeurModel(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)
    x_internal: np.ndarray = LavaPyType(np.ndarray, np.float64)
    theta: np.ndarray = LavaPyType(np.ndarray, np.float64)
    theta_decay_schedule: int = LavaPyType(int, np.int32)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)
        self.theta_decay_type = self.proc_params["theta_decay_type"]
        self.theta_decay_indices = self.proc_params["theta_decay_indices"]
        self.decay_counter = 0

    def run_spk(self):
        s_in = self.s_in.recv()
        delta_state = s_in - self.x_internal
        self.x_internal = s_in
        self.decay_counter += 1
        if self.theta_decay_type == "schedule":
            if self.decay_counter == self.theta_decay_schedule:
                # TODO: guard against shift overflows in fixed-point
                self.theta = self.theta / 2
                print(self.theta[0])
                self.decay_counter = 0
        if self.theta_decay_type == "indices":
            if self.decay_counter in self.theta_decay_indices:
                self.theta = self.theta / 2

        # IMP: Using x_internal below ensures sigma-delta behavior.
        # Additional sigma layer not required. Otherwise use self.delta would
        # lead to delta behavior only
        s_out = delta_state * (np.abs(delta_state) >= self.theta)
        self.s_out.send(s_out)
