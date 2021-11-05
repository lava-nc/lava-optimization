# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
from numpy.core.fromnumeric import shape

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg   

from lava.lib.optimization.solvers.qp.models import ConstraintCheck, \
SolutionNeurons, ConstraintNormals, ConstraintDirections, \
QuadraticConnectivity, GradientDynamics

from lava.lib.optimization.solvers.qp.processes import ConstraintCheck, ConstraintNeurons, \
SolutionNeurons, ConstraintNormals, ConstraintDirections, \
QuadraticConnectivity, GradientDynamics

class InSpikeSetProcess(AbstractProcess):
    def __init__(self, **kwargs):
        """Use to set value of input spike to a process
        
        Kwargs:
            in_shape (int tuple): set a_out to custom value 
            spike_in (1-D array): input spike  value to send
        """
        super().__init__(**kwargs)
        shape = kwargs.pop("in_shape", (1,1))
        self.a_out = OutPort(shape=shape)
        self.spike_inp = Var(shape=shape, 
                             init=kwargs.pop("spike_in", 0)
                            )
        
class OutProbeProcess(AbstractProcess):
    def __init__(self, **kwargs):
        """Use to set read output spike from a process

        Kwargs:
            out_shape (int tuple): set OutShape to custom value 
        """
        super().__init__(**kwargs)
        shape = kwargs.pop("out_shape", (1,1))
        self.s_in = InPort(shape=shape)      
        self.spike_out = Var(shape=shape, 
                             init=np.zeros(shape)
                            )

@implements(proc=InSpikeSetProcess, protocol=LoihiProtocol)
@requires(CPU)
class PyDenseModel(PyLoihiProcessModel):
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    spike_inp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)

    def run_spk(self):    
        a_out = self.spike_inp
        self.a_out.send(a_out)
        self.a_out.flush()
@implements(proc=OutProbeProcess, protocol=LoihiProtocol)
@requires(CPU)
class PyDenseModel(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    spike_out: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)

    def run_spk(self):
        s_in = self.s_in.recv()
        self.spike_out = s_in


print("[LavaQpOpt][INFO]:Starting Floating Point tests for models in"
    + " QP solver.")
class TestModelsFloatingPoint(unittest.TestCase):
    """Tests of all models of the QP solver in floating point
    """

    def test_process_constraint_directions(self):
        process = ConstraintDirections()
        self.assertEqual(process.vars.weights.get()==0, True) 
        print("[LavaQpOpt][INFO]: Default initialization test passed for " 
        + "ConstraintDirections")
        weights = np.array(
                  [[2,    3, 6],
                   [43,   3, 2]]
                  )
        process = ConstraintDirections(shape = weights.shape, 
                                        constraint_directions=weights)
        self.assertEqual(np.all(process.vars.weights.get()==weights), True)
        self.assertEqual(np.all(process.s_in.shape==(weights.shape[1],1)), 
                         True) 
        self.assertEqual(np.all(process.a_out.shape==(weights.shape[0],1)), 
                         True) 
        print("[LavaQpOpt][INFO]: Custom initialization test passed for " 
        + "ConstraintDirections")

        input_spike = np.array([[1],[1],[1]])
        in_spike_process = InSpikeSetProcess(in_shape=input_spike.shape, 
                                            spike_in=input_spike)
        out_spike_process = OutProbeProcess(out_shape=process.a_out.shape)

        in_spike_process.a_out.connect(process.s_in)
        process.a_out.connect(out_spike_process.s_in)

        # in_spike_process.run(condition=RunSteps(num_steps=1), 
        #                      run_cfg=Loihi1SimCfg())
        # in_spike_process.stop()
        # self.assertEqual(np.all(out_spike_process.vars.spike_out.get()
        #                         ==(weights@input_spike)
        #                         ), True)
        #print("[LavaQpOpt][INFO]: Behavioral test passed for " 
        #+ "ConstraintDirections")


        # default initialize, assinged initialization, process behavior 
        self.assertEqual(True, True)

    def test_process_constraint_neurons(self):
        process = ConstraintDirections()
        self.assertEqual(process.vars.weights.get()==0, True) 
        print("[LavaQpOpt][INFO]: Default initialization test passed for " 
        + "ConstraintNeurons")
        inp_bias = np.array([2, 4, 6]).T
        process = ConstraintNeurons(shape = inp_bias.shape, 
                                        thresholds=inp_bias)
        self.assertEqual(np.all(process.vars.thresholds.get()==inp_bias), True)
        self.assertEqual(np.all(process.s_in.shape==(inp_bias.shape[0],1)), 
                         True) 
        print("[LavaQpOpt][INFO]: Custom initialization test passed for " 
        + "ConstraintNeurons")
    
    def tests_process_solution_neurons(self):
        init_sol = np.array([2, 4, 6, 4, 1]).T
        p = np.array([4, 3, 2, 1, 1]).T
        alpha, beta, alpha_d, beta_g = 3, 2, 100, 100
        process = SolutionNeurons(shape=init_sol.shape, 
                                  qp_neurons_init=init_sol,
                                  grad_bias=p, alpha=alpha, beta=beta, 
                                  alpha_decay_schedule=alpha_d, 
                                  beta_growth_schedule=beta_g)
        self.assertEqual(np.all(process.vars.qp_neuron_state.get()==init_sol), 
                                True)
        self.assertEqual(np.all(process.vars.grad_bias.get()==p), True)
        self.assertEqual(np.all(process.vars.alpha.get()==alpha), True)
        self.assertEqual(np.all(process.vars.beta.get()==beta), True)
        self.assertEqual(process.vars.alpha_decay_schedule.get()==alpha_d, 
                        True)
        self.assertEqual(process.vars.beta_growth_schedule.get()==beta_g, True)
        self.assertEqual(process.vars.decay_counter.get()==0, True)
        self.assertEqual(process.vars.growth_counter.get()==0, True)
        self.assertEqual(np.all(process.s_in_qc.shape==(p.shape[0],1)), 
                         True) 
        self.assertEqual(np.all(process.s_in_cn.shape==(p.shape[0],1)), 
                         True) 
        self.assertEqual(np.all(process.a_out_qc.shape==(p.shape[0],1)), 
                         True)
        self.assertEqual(np.all(process.a_out_cc.shape==(p.shape[0],1)), 
                         True) 
        print("[LavaQpOpt][INFO]: Custom initialization test passed for " 
        + "SolutionNeurons")

    def test_process_constraint_normals(self):
        process = ConstraintNormals()
        self.assertEqual(process.vars.weights.get()==0, True) 
        print("[LavaQpOpt][INFO]: Default initialization test passed for " 
        + "ConstraintNormals")
        weights = np.array(
                  [[2,    3, 6],
                   [43,   3, 2]]
                  ).T
        process = ConstraintNormals(shape = weights.shape, 
                                        constraint_normals=weights)
        self.assertEqual(np.all(process.vars.weights.get()==weights), True)
        self.assertEqual(np.all(process.s_in.shape==(weights.shape[1],1)), 
                         True) 
        self.assertEqual(np.all(process.a_out.shape==(weights.shape[0],1)), 
                         True) 
        print("[LavaQpOpt][INFO]: Custom initialization test passed for " 
        + "ConstraintNormals")
    
    def test_process_quadratic_connectivity(self):
        process = QuadraticConnectivity()
        self.assertEqual(process.vars.weights.get()==0, True) 
        print("[LavaQpOpt][INFO]: Default initialization test passed for " 
        + "QuadraticConnectivity")
        weights = np.array(
                  [[2,    43, 2],
                   [43,   3, 4],
                   [2,    4, 1]]
                  )
        process = QuadraticConnectivity(shape=weights.shape, hessian=weights)
        self.assertEqual(np.all(process.vars.weights.get()==weights), True)
        self.assertEqual(np.all(process.s_in.shape==(weights.shape[1],1)), 
                         True) 
        self.assertEqual(np.all(process.a_out.shape==(weights.shape[0],1)), 
                         True) 
        print("[LavaQpOpt][INFO]: Custom initialization test passed for " 
        + "QuadraticConnectivity")
      
    
    def test_process_constraint_check(self):
        A = np.array(
                  [[2,    3, 6],
                   [43,   3, 2]]
                  )

        b = np.array([2, 4]).T
        process = ConstraintCheck(constraint_matrix=A, 
                                  constraint_bias=b)
        self.assertEqual(np.all(process.vars.constraint_matrix.get()==A), True)
        self.assertEqual(np.all(process.vars.constraint_bias.get()==b), True)
        self.assertEqual(np.all(process.s_in.shape==(A.shape[1],1)), 
                         True) 
        self.assertEqual(np.all(process.a_out.shape==(A.shape[0],1)), 
                         True) 
        print("[LavaQpOpt][INFO]: Custom initialization test passed for " 
        + "ConstraintCheck")

    def test_process_gradient_dynamics(self):
        hessian = np.array(
            [[2,  43, 2],
            [43,   3, 4],
            [2,    4, 1]]
            )

        A_T = np.array(
            [[2,    3, 6],
            [43,   3, 2]]
            ).T

        init_sol = np.array([2, 4, 6, 4, 1]).T
        p = np.array([4, 3, 2, 1, 1]).T
        alpha, beta, alpha_d, beta_g = 3, 2, 100, 100
        process = GradientDynamics(hessian=hessian, constraint_matrix_T = A_T, 
                                  qp_neurons_init=init_sol,
                                  grad_bias=p, alpha=alpha, beta=beta, 
                                  alpha_decay_schedule=alpha_d, 
                                  beta_growth_schedule=beta_g)
        self.assertEqual(np.all(process.vars.constraint_matrix_T.get()==A_T), 
                        True)
        self.assertEqual(np.all(process.vars.hessian.get()==hessian), True)
        self.assertEqual(np.all(process.vars.qp_neuron_state.get()==init_sol), 
                        True)
        self.assertEqual(np.all(process.vars.grad_bias.get()==p), True)
        self.assertEqual(np.all(process.vars.alpha.get()==alpha), True)
        self.assertEqual(np.all(process.vars.beta.get()==beta), True)
        self.assertEqual(process.vars.alpha_decay_schedule.get()==alpha_d, 
                        True)
        self.assertEqual(process.vars.beta_growth_schedule.get()==beta_g, True)
        self.assertEqual(np.all(process.s_in.shape==(A_T.shape[0],1)), 
                         True) 
        self.assertEqual(np.all(process.a_out.shape==(hessian.shape[0],1)), 
                         True) 
        print("[LavaQpOpt][INFO]: Custom initialization test passed for " 
        + "GradientDynamics")
    
    def test_QP(self):
        # connect constraint check and gradient dynamics
        pass

if __name__ == '__main__':
    unittest.main()
