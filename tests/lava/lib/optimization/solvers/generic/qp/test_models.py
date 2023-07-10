# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

# Behavioral tests for all the models in QP
import unittest
import numpy as np
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg

from lava.lib.optimization.solvers.generic.qp.models import (
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


class InSpikeSetProcess(AbstractProcess):
    def __init__(self, **kwargs):
        """Use to set value of input spike to a process

        Kwargs:
        ------
            in_shape : int tuple, optional
                Set a_out to custom value
            spike_in : 1-D array, optional
                Input spike  value to send
        """
        super().__init__(**kwargs)
        shape = kwargs.pop("in_shape", (1,))
        self.a_out = OutPort(shape=shape)
        self.spike_inp = Var(shape=shape, init=kwargs.pop("spike_in", 0))


class OutProbeProcess(AbstractProcess):
    def __init__(self, **kwargs):
        """Use to set read output spike from a process

        Kwargs:
        ------
            out_shape : int tuple, optional
                Set OutShape to custom value
        """
        super().__init__(**kwargs)
        shape = kwargs.pop("out_shape", (1,))
        self.s_in = InPort(shape=shape)
        self.spike_out = Var(shape=shape, init=np.zeros(shape))


@implements(proc=InSpikeSetProcess, protocol=LoihiProtocol)
@requires(CPU)
class PyISSModel(PyLoihiProcessModel):
    a_out: PyOutPort = LavaPyType(
        PyOutPort.VEC_DENSE, np.float64, precision=64
    )
    spike_inp: np.ndarray = LavaPyType(np.ndarray, np.float64, precision=64)

    def run_spk(self):
        a_out = self.spike_inp
        self.a_out.send(a_out)
        self.a_out.flush()


@implements(proc=OutProbeProcess, protocol=LoihiProtocol)
@requires(CPU)
class PyOPPModel(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64, precision=64)
    spike_out: np.ndarray = LavaPyType(np.ndarray, np.float64, precision=64)

    def run_spk(self):
        s_in = self.s_in.recv()
        self.spike_out = s_in


class TestModelsFloatingPoint(unittest.TestCase):
    """Tests of all model behaviors of the QP solver in floating point. Refer
    to QP models.py in qp/solver repo to understand behaviours.
    """

    def test_model_QPDense(self):
        """test behavior of constraint directions process
        (Matrix-vector multiplication)
        """
        weights = np.array([[2.5, 3.2, 6], [43, 3, 2]])
        process = QPDense(shape=weights.shape, weights=weights)

        input_spike = np.array([[1], [1], [1]])
        in_spike_process = InSpikeSetProcess(
            in_shape=input_spike.shape, spike_in=input_spike
        )
        out_spike_process = OutProbeProcess(out_shape=process.a_out.shape)

        in_spike_process.a_out.connect(process.s_in)
        process.a_out.connect(out_spike_process.s_in)

        in_spike_process.run(
            condition=RunSteps(num_steps=1),
            run_cfg=Loihi1SimCfg(),
        )
        var = out_spike_process.vars.spike_out.get()
        in_spike_process.stop()

        self.assertEqual(
            np.all(var == (weights @ input_spike)),
            True,
        )

    def test_model_constraint_neurons(self):
        """test behavior of constraint directions process
        (vector-vector addition)
        """
        inp_bias = np.array([[2, 4, 6]]).T
        process = ConstraintNeurons(shape=inp_bias.shape, thresholds=inp_bias)
        input_spike = np.array([[1], [5], [2]])
        in_spike_process = InSpikeSetProcess(
            in_shape=input_spike.shape, spike_in=input_spike
        )
        out_spike_process = OutProbeProcess(out_shape=process.s_out.shape)

        in_spike_process.a_out.connect(process.a_in)
        process.s_out.connect(out_spike_process.s_in)

        in_spike_process.run(
            condition=RunSteps(num_steps=1), run_cfg=Loihi1SimCfg()
        )
        self.assertEqual(
            np.all(
                out_spike_process.vars.spike_out.get()
                == ((input_spike - inp_bias) * (input_spike > inp_bias))
            ),
            True,
        )
        in_spike_process.stop()

    def test_model_solution_neurons(self):
        """test behavior of SolutionNeurons process
        -alpha*(input_spike_1 + p)- beta*input_spike_2
        """
        init_sol = np.array([[2, 4, 6, 4, 1]]).T
        p = np.array([[4, 3, 2, 1, 1]]).T
        alpha, beta, alpha_d, beta_g = 3, 2, 100, 100
        process = SolutionNeurons(
            shape=init_sol.shape,
            qp_neurons_init=init_sol,
            grad_bias=p,
            alpha=alpha,
            beta=beta,
            alpha_decay_schedule=alpha_d,
            beta_growth_schedule=beta_g,
        )
        input_spike_cn = np.array([[1], [5], [2], [2], [0]])
        input_spike_qc = np.array([[8], [2], [22], [21], [1]])
        in_spike_cn_process = InSpikeSetProcess(
            in_shape=input_spike_cn.shape, spike_in=input_spike_cn
        )
        in_spike_qc_process = InSpikeSetProcess(
            in_shape=input_spike_qc.shape, spike_in=input_spike_qc
        )
        out_spike_cc_process = OutProbeProcess(
            out_shape=process.s_out_cc.shape
        )
        out_spike_qc_process = OutProbeProcess(
            out_shape=process.s_out_qc.shape
        )

        in_spike_cn_process.a_out.connect(process.a_in_cn)
        in_spike_qc_process.a_out.connect(process.a_in_qc)
        process.s_out_cc.connect(out_spike_cc_process.s_in)
        process.s_out_qc.connect(out_spike_qc_process.s_in)

        # testing for two timesteps because of design of
        # solution neurons for recurrent connectivity. Nth
        # state available only at N+1th timestep
        in_spike_cn_process.run(
            condition=RunSteps(num_steps=2), run_cfg=Loihi1SimCfg()
        )
        self.assertEqual(
            np.all(
                out_spike_cc_process.vars.spike_out.get()
                == (
                    init_sol
                    - alpha * (input_spike_qc + p)
                    - beta * input_spike_cn
                )
            ),
            True,
        )
        in_spike_cn_process.stop()

    def test_model_constraint_check(self):
        """test behavior of ConstraintCheck process
        (Ax-k)*(Ax<k)
        """
        A = np.array([[2, 3, 6], [43, 3, 2]])

        k = np.array([[2, 4]]).T
        process = ConstraintCheck(constraint_matrix=A, constraint_bias=k)
        input_spike = np.array([[1], [2], [1]])
        in_spike_process = InSpikeSetProcess(
            in_shape=input_spike.shape, spike_in=input_spike
        )
        out_spike_process = OutProbeProcess(out_shape=process.s_out.shape)

        in_spike_process.a_out.connect(process.s_in)
        process.s_out.connect(out_spike_process.s_in)

        in_spike_process.run(
            condition=RunSteps(num_steps=1),
            run_cfg=Loihi1SimCfg(select_sub_proc_model=True),
        )
        self.assertEqual(
            np.all(
                out_spike_process.vars.spike_out.get()
                == ((A @ input_spike - k) * (A @ input_spike > k))
            ),
            True,
        )
        in_spike_process.stop()

    def test_sparse_model_constraint_check(self):
        """Test behavior of sparsified constraint check process. Contains on
        SigmaNeurons
        """
        A = np.array([[2, 3, 6], [43, 3, 2]])
        k = np.array([[2, 4]]).T
        x_init = np.array([[0.2, 0.4, 0.2]]).T
        process = ConstraintCheck(
            constraint_matrix=A,
            constraint_bias=k,
            sparse=True,
            x_int_init=x_init,
        )
        input_spike = np.array([[1], [2], [1]])
        in_spike_process = InSpikeSetProcess(
            in_shape=input_spike.shape, spike_in=input_spike
        )
        out_spike_process = OutProbeProcess(out_shape=process.s_out.shape)

        in_spike_process.a_out.connect(process.s_in)
        process.s_out.connect(out_spike_process.s_in)

        in_spike_process.run(
            condition=RunSteps(num_steps=1),
            run_cfg=Loihi1SimCfg(select_sub_proc_model=True),
        )
        val = out_spike_process.vars.spike_out.get()
        in_spike_process.stop()
        self.assertEqual(
            np.all(
                val
                == (
                    (A @ (input_spike + x_init) - k)
                    * (A @ (input_spike + x_init) > k)
                )
            ),
            True,
        )

    def test_model_gradient_dynamics(self):
        """test behavior of GradientDynamics process
        -alpha*(Q@x_init + p)- beta*A_T@graded_constraint_spike
        """
        Q = np.array([[2, 43.2, 2], [43, 3, 4], [2, 4, 1]])

        A_T = np.array([[2, 3.3, 6], [43, 3, 2]]).T

        init_sol = np.array([[2, 4, 6]]).T
        p = np.array([[4, 3, 2]]).T
        alpha, beta, alpha_d, beta_g = 3, 2, 100, 100
        process = GradientDynamics(
            hessian=Q,
            constraint_matrix_T=A_T,
            qp_neurons_init=init_sol,
            grad_bias=p,
            alpha=alpha,
            beta=beta,
            alpha_decay_schedule=alpha_d,
            beta_growth_schedule=beta_g,
        )

        input_spike = np.array([[1], [2]])
        in_spike_process = InSpikeSetProcess(
            in_shape=input_spike.shape, spike_in=input_spike
        )
        out_spike_process = OutProbeProcess(out_shape=process.s_out.shape)
        in_spike_process.a_out.connect(process.s_in)
        process.s_out.connect(out_spike_process.s_in)

        # testing for two timesteps because of design of
        # solution neurons for recurrent connectivity. Nth
        # state available only at N+1th timestep

        in_spike_process.run(
            condition=RunSteps(num_steps=2),
            run_cfg=Loihi1SimCfg(select_sub_proc_model=True),
        )
        self.assertEqual(
            np.all(
                out_spike_process.vars.spike_out.get()
                == (
                    init_sol
                    - alpha * (Q @ init_sol + p)
                    - beta * A_T @ input_spike
                )
            ),
            True,
        )
        in_spike_process.stop()

    def test_model_sparse_gradient_dynamics(self):
        """test gradient dynamics with delta and sigma sparsifiers"""
        Q = np.array([[2, 43.2, 2], [43, 3, 4], [2, 4, 1]])

        A_T = np.array([[2, 3.3, 6], [43, 3, 2]]).T

        init_sol = np.array([[2, 4, 6]]).T
        p = np.array([[4, 3, 2]]).T
        alpha, beta, alpha_d, beta_g = 3, 2, 100, 100
        theta = 0.2
        process = GradientDynamics(
            hessian=Q,
            constraint_matrix_T=A_T,
            qp_neurons_init=init_sol,
            sparse=True,
            model="SigDel",
            theta=theta,
            grad_bias=p,
            alpha=alpha,
            beta=beta,
            alpha_decay_schedule=alpha_d,
            beta_growth_schedule=beta_g,
        )

        input_spike = np.array([[1], [2]])
        in_spike_process = InSpikeSetProcess(
            in_shape=input_spike.shape, spike_in=input_spike
        )
        out_spike_process = OutProbeProcess(out_shape=process.s_out.shape)
        in_spike_process.a_out.connect(process.s_in)
        process.s_out.connect(out_spike_process.s_in)

        # testing for two timesteps because of design of
        # solution neurons for recurrent connectivity. Nth
        # state available only at N+1th timestep

        in_spike_process.run(
            condition=RunSteps(num_steps=2),
            run_cfg=Loihi1SimCfg(select_sub_proc_model=True),
        )

        val = out_spike_process.vars.spike_out.get()
        in_spike_process.stop()

        prev_val = init_sol

        curr_val = (
            init_sol - alpha * (Q @ init_sol + p) - beta * A_T @ input_spike
        )

        self.assertEqual(
            np.all(
                val
                == (
                    (curr_val - prev_val)
                    * (np.abs(curr_val - prev_val) > theta)
                )
            ),
            True,
        )

    def test_floating_pt_model_projected_gradient_neurons(self):
        """test behavior of neurons that perform projected gradient in pipgeq
        process.
        -alpha*(input_spike_1 + p)- beta*input_spike_2
        """
        init_sol = np.array([2, 4, 6, 4, 1]).T
        p = np.array([4, 3, 2, 1, 1]).T
        alpha, alpha_d = 3, 100
        process = ProjectedGradientNeuronsPIPGeq(
            shape=init_sol.shape,
            qp_neurons_init=init_sol,
            grad_bias=p,
            alpha=alpha,
            alpha_decay_schedule=alpha_d,
        )
        input_spike_cn = np.array([1, 5, 2, 2, 0])
        input_spike_qc = np.array([8, 2, 22, 21, 1])
        in_spike_cn_process = InSpikeSetProcess(
            in_shape=input_spike_cn.shape, spike_in=input_spike_cn
        )
        in_spike_qc_process = InSpikeSetProcess(
            in_shape=input_spike_qc.shape, spike_in=input_spike_qc
        )
        out_spike_cd_process = OutProbeProcess(out_shape=process.s_out.shape)

        in_spike_cn_process.a_out.connect(process.a_in)
        in_spike_qc_process.a_out.connect(process.a_in)
        process.s_out.connect(out_spike_cd_process.s_in)

        # Advancing one time-step does not change state since neuron works only
        # even time-steps
        in_spike_cn_process.run(
            condition=RunSteps(num_steps=1), run_cfg=Loihi1SimCfg()
        )
        self.assertEqual(
            np.all(
                out_spike_cd_process.vars.spike_out.get() == np.zeros(p.shape)
            ),
            True,
        )

        # Advancing two timesteps since the neuron fires only at even time-steps
        in_spike_cn_process.run(
            condition=RunSteps(num_steps=1), run_cfg=Loihi1SimCfg()
        )
        self.assertEqual(
            np.all(
                out_spike_cd_process.vars.spike_out.get()
                == (
                    init_sol
                    - alpha * (2 * (input_spike_qc + input_spike_cn) + p)
                )
            ),
            True,
        )
        in_spike_cn_process.stop()

    def test_floating_pt_model_proportional_integral_neurons(self):
        """test behavior of proportional integral neurons process which are a
        part of PIPGeq updates.
        Updates happen only at odd time-steps
        neuron_state =  neuron_state + beta*(input_spike - k)
        a_out = neuron_state + beta*(input_spike - k)
        """
        init_sol = np.array([2, 4, 6, 4, 1]).T
        p = np.array([4, 3, 2, 1, 1]).T
        beta, beta_g = 2, 100
        process = ProportionalIntegralNeuronsPIPGeq(
            shape=init_sol.shape,
            constraint_neurons_init=init_sol,
            thresholds=p,
            beta=beta,
            beta_growth_schedule=beta_g,
        )
        input_spike = np.array([1, 5, 2, 2, 0])
        in_spike_process = InSpikeSetProcess(
            in_shape=input_spike.shape, spike_in=input_spike
        )
        out_spike_process = OutProbeProcess(out_shape=process.s_out.shape)

        in_spike_process.a_out.connect(process.a_in)
        process.s_out.connect(out_spike_process.s_in)

        # Advancing one timestep since the neuron fires only at odd time-steps
        in_spike_process.run(
            condition=RunSteps(num_steps=1), run_cfg=Loihi1SimCfg()
        )
        state_var = process.vars.constraint_neuron_state.get()
        out_var = out_spike_process.vars.spike_out.get()
        self.assertEqual(
            np.all(state_var == (init_sol + beta * (input_spike - p))),
            True,
        )
        # Output spike is a non-zero vector at even time-step
        self.assertEqual(
            np.all(out_var == (init_sol + 2 * beta * (input_spike - p))),
            True,
        )
        # Even timestep does not change internal state
        in_spike_process.run(
            condition=RunSteps(num_steps=1), run_cfg=Loihi1SimCfg()
        )
        state_var = process.vars.constraint_neuron_state.get()
        out_var = out_spike_process.vars.spike_out.get()
        in_spike_process.stop()

        self.assertEqual(
            np.all(state_var == (init_sol + beta * (input_spike - p))),
            True,
        )
        # Output spike is a zero vector at even time-step
        self.assertEqual(
            np.all(out_var == np.zeros(p.shape)),
            True,
        )

    def test_model_sigma_neuron(self):
        """test behavior of sigma (accumulator) neuron process
        (vector-vector addition)
        """
        inp_bias = np.array([[2, 4, 6]]).T
        process = SigmaNeurons(shape=inp_bias.shape, x_sig_init=inp_bias)
        input_spike = np.array([[1], [5], [2]])
        in_spike_process = InSpikeSetProcess(
            in_shape=input_spike.shape, spike_in=input_spike
        )
        out_spike_process = OutProbeProcess(out_shape=process.s_out.shape)

        in_spike_process.a_out.connect(process.s_in)
        process.s_out.connect(out_spike_process.s_in)

        in_spike_process.run(
            condition=RunSteps(num_steps=2), run_cfg=Loihi1SimCfg()
        )
        val = out_spike_process.vars.spike_out.get()
        in_spike_process.stop()

        # testing accumulation operation
        self.assertEqual(np.all(val == (2 * input_spike + inp_bias)), True)

    def test_model_delta_neuron(self):
        """
        test behavior of delta (thresholded subractors) neuron process
        (vector-vector subraction and thresholding)
        """
        inp_bias = np.array([[2, 4, 6]]).T
        theta = 5
        process = DeltaNeurons(
            shape=inp_bias.shape, x_del_init=inp_bias, theta=theta
        )
        input_spike = np.array([[1], [5], [2]])
        in_spike_process = InSpikeSetProcess(
            in_shape=input_spike.shape, spike_in=input_spike
        )
        out_spike_process = OutProbeProcess(out_shape=process.s_out.shape)

        in_spike_process.a_out.connect(process.s_in)
        process.s_out.connect(out_spike_process.s_in)

        in_spike_process.run(
            condition=RunSteps(num_steps=1), run_cfg=Loihi1SimCfg()
        )
        val = out_spike_process.vars.spike_out.get()
        in_spike_process.stop()

        # testing delta operation
        delta_state = input_spike - inp_bias
        self.assertEqual(
            np.all(val == delta_state * (np.abs(delta_state) >= theta)), True
        )


if __name__ == "__main__":
    unittest.main()
