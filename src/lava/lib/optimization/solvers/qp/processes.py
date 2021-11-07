# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
import numpy as np


class ConstraintDirections(AbstractProcess):
    """Connections in the constraint-checking group of neurons.
    Realizes the following abstract behavior:
    a_out = A * s_in

    intialize the constraintDirectionsProcess

    Kwargs:
        shape (int tuple): Define the shape of the connections
        matrix as a tuple. Defaults to (1,1)
        constraint_directions (1-D  or 2-D np.array): Define the directions
        of the linear constraint hyperplanes. This is 'A' in the constraints
        of the QP. Defaults to 0
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        self.s_in = InPort(shape=(shape[1], 1))
        self.a_out = OutPort(shape=(shape[0], 1))
        self.weights = Var(
            shape=shape, init=kwargs.pop("constraint_directions", 0)
        )


class ConstraintNeurons(AbstractProcess):
    """Process to check the violation of the linear constraints of the QP. A
    graded spike corresponding to the violated constraint is sent from the out
    port.

    Intialize the constraintNeurons Process.

    Kwargs:
        shape (int tuple): Define the shape of the thresholds vector.
        Defaults to (1,1).
        thresholds (1-D np.array): Define the thresholds of the neurons
        in the constraint checking layer. This is usually 'b' in the
        constraints of the QP. Default value of thresholds is 0.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        self.s_in = InPort(shape=(shape[0], 1))
        self.a_out = OutPort(shape=(shape[0], 1))
        self.thresholds = Var(shape=shape, init=kwargs.pop("thresholds", 0))


class QuadraticConnectivity(AbstractProcess):
    """The connections that define the Hessian of the quadratic cost function
    Realizes the following abstract behavior:
    a_out = A * s_in

    Intialize the quadraticConnectivity process.

    Kwargs:
        shape (int tuple): A tuple defining the shape of the connections
        matrix. Defaults to (1,1).
        hessian (1-D  or 2-D np.array): Define the hessian matrix ('P' in
        the cost function of the QP) in the QP. Defaults to 0.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        self.s_in = InPort(shape=(shape[1], 1))
        self.a_out = OutPort(shape=(shape[0], 1))
        self.weights = Var(shape=shape, init=kwargs.pop("hessian", 0))


class SolutionNeurons(AbstractProcess):
    """The neurons that evolve according to the constraint-corrected gradient
    dynamics.

    Intialize the solutionNeurons process.

    Kwargs:
        shape (int tuple): A tuple defining the shape of the qp neurons.
        Defaults to (1,1).
        qp_neurons_init (1-D np.array): initial value of qp solution neurons
        grad_bias (1-D np.array): The bias of the gradient of the QP. This
        is the value 'p' in the QP definition.
        alpha (1-D np.array): Defines the learning rate for gradient
        descent. Defaults to 1.
        beta (1-D np.array): Defines the learning rate for constraint-
        checking. Defaults to 1.
        alpha_decay_schedule (int): The number of iterations after which one
        right shift operation takes place for alpha. Default intialization
        to a very high value of 10000.
        beta_growth_schedule (int): The number of iterations after which one
        left shift operation takes place for beta. Default intialization
        to a very high value of 10000.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        # In/outPorts that come from/go to the quadratic connectivity process
        self.s_in_qc = InPort(shape=(shape[0], 1))
        self.a_out_qc = OutPort(shape=(shape[0], 1))
        # In/outPorts that come from/go to the constraint normals process
        self.s_in_cn = InPort(shape=(shape[0], 1))
        # OutPort for constraint checking
        self.a_out_cc = OutPort(shape=(shape[0], 1))
        self.qp_neuron_state = Var(
            shape=shape, init=kwargs.pop("qp_neurons_init", np.zeros(shape))
        )
        self.grad_bias = Var(
            shape=shape, init=kwargs.pop("grad_bias", np.zeros(shape))
        )
        self.alpha = Var(
            shape=shape, init=kwargs.pop("alpha", np.ones((shape[0], 1)))
        )
        self.beta = Var(
            shape=shape, init=kwargs.pop("beta", np.ones((shape[0], 1)))
        )
        self.alpha_decay_schedule = Var(
            shape=(1, 1), init=kwargs.pop("alpha_decay_schedule", 10000)
        )
        self.beta_growth_schedule = Var(
            shape=(1, 1), init=kwargs.pop("beta_growth_schedule", 10000)
        )
        self.decay_counter = Var(shape=(1, 1), init=0)
        self.growth_counter = Var(shape=(1, 1), init=0)


class ConstraintNormals(AbstractProcess):
    """Connections influencing the gradient dynamics when constraints are
    violated.
    Realizes the following abstract behavior:
    a_out = A * s_in

    Intialize the constraint normals to assign weights to constraint
    violation spikes.

    Kwargs:
        shape (int tuple): A tuple defining the shape of the connections
        matrix. Defaults to (1,1).
        constraint_normals (1-D  or 2-D np.array): Define the normals
        of the linear constraint hyperplanes. This is A^T in the constraints
        of the QP. Defaults to 0
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        self.s_in = InPort(shape=(shape[1], 1))
        self.a_out = OutPort(shape=(shape[0], 1))
        self.weights = Var(
            shape=shape, init=kwargs.pop("constraint_normals", 0)
        )


class ConstraintCheck(AbstractProcess):
    """Check if linear constraints (equality/inequality) are violated for the
    qp. Recieves and sends graded spike from and to the gradientDynamics
    process. House the constraintDirections and constraintNeurons as
    sub-processes.

    Initialize constraintCheck Process.

    Kwargs:
        constraint_matrix (1-D  or 2-D np.array):  The value of the constraint
        matrix. This is 'A' in the linear constraints
        constraint_bias (1-D np.array):  The value of the constraint
        bias. This is 'b' in the linear constraints
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        constraint_matrix = kwargs.pop("constraint_matrix", 0)
        shape = constraint_matrix.shape
        self.s_in = InPort(shape=(shape[1], 1))
        self.constraint_matrix = Var(shape=shape, init=constraint_matrix)
        self.constraint_bias = Var(
            shape=(shape[0], 1), init=kwargs.pop("constraint_bias", 0)
        )
        self.a_out = OutPort(shape=(shape[0], 1))


class GradientDynamics(AbstractProcess):
    """Perform gradient descent with constraint correction to converge at the
    solution of the QP.
    Initialize gradientDynamics Process.

    Kwargs:
        hessian (1-D  or 2-D np.array): Define the hessian matrix ('P' in
        the cost function of the QP) in the QP. Defaults to 0.
        constraint_matrix_T (1-D  or 2-D np.array):  The value of the transpose
        of the constraint matrix. This is 'A^T' in the linear constraints.
        grad_bias (1-D np.array): The bias of the gradient of the QP. This
        is the value 'p' in the QP definition.
        qp_neurons_init (1-D np.array): initial value of qp solution neurons
        alpha (1-D np.array): Defines the learning rate for gradient
        descent. Defaults to 1.
        beta (1-D np.array): Defines the learning rate for constraint-
        checking. Defaults to 1.
        alpha_decay_schedule (int): The number of iterations after which one
        right shift operation takes place for alpha. Default intialization
        to a very high value of 10000.
        beta_growth_schedule (int): The number of iterations after which one
        left shift operation takes place for beta. Default intialization
        to a very high value of 10000.
    """

    def __init__(self, **kwargs):
        """ """
        super().__init__(**kwargs)
        hessian = kwargs.pop("hessian", 0)
        A_T = kwargs.pop("constraint_matrix_T", 0)
        shape_hess = hessian.shape
        shape_A_T = A_T.shape
        self.s_in = InPort(shape=(shape_A_T[1], 1))
        self.hessian = Var(shape=shape_hess, init=hessian)
        self.constraint_matrix_T = Var(shape=shape_A_T, init=A_T)
        self.grad_bias = Var(
            shape=(shape_hess[0], 1),
            init=kwargs.pop("grad_bias", np.zeros((shape_hess[0], 1))),
        )
        self.qp_neuron_state = Var(
            shape=(shape_hess[0], 1),
            init=kwargs.pop("qp_neurons_init", np.zeros((shape_hess[0], 1))),
        )
        self.alpha = Var(
            shape=(shape_hess[0], 1),
            init=kwargs.pop("alpha", np.ones((shape_hess[0], 1))),
        )
        self.beta = Var(
            shape=(shape_hess[0], 1),
            init=kwargs.pop("beta", np.ones((shape_hess[0], 1))),
        )
        self.alpha_decay_schedule = Var(
            shape=(1, 1), init=kwargs.pop("alpha_decay_schedule", 10000)
        )
        self.beta_growth_schedule = Var(
            shape=(1, 1), init=kwargs.pop("beta_growth_schedule", 10000)
        )

        self.a_out = OutPort(shape=(shape_hess[0], 1))
