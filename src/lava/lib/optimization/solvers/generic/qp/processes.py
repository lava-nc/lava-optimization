# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort

import numpy as np
import typing as ty


class QPDense(AbstractProcess):
    """Connections in the between two neurons in the QP solver. Does not buffer
    connections like the dense process in Lava. Meant for the GDCC QP solver
    only.

    Realizes the following abstract behavior:
    a_out = weights * s_in

    intialize the constraintDirectionsProcess

    Parameters
    ----------

    shape : int tuple, optional
        Define the shape of the connections matrix as a tuple. Defaults to
        (1,1)
    weights : (1-D  or 2-D np.array), optional
        Define the weights for the dense connection process

    """
    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        self.s_in = InPort(shape=(shape[1], 1))
        self.a_out = OutPort(shape=(shape[0], 1))
        weights = kwargs.pop("weights", 0)
        self.weights = Var(shape=shape, init=weights)


class ConstraintNeurons(AbstractProcess):
    """Process to check the violation of the linear constraints of the QP. A
    graded spike corresponding to the violated constraint is sent from the out
    port.

    Realizes the following abstract behavior:
    s_out = (a_in - thresholds) * (a_in < thresholds)

    Intialize the constraintNeurons Process.

    Parameters
    ----------
    shape : int tuple, optional
        Define the shape of the thresholds vector. Defaults to (1,1).
    thresholds : 1-D np.array, optional
        Define the thresholds of the neurons in the
        constraint checking layer. This is usually 'k' in the constraints
        of the QP. Default value of thresholds is 0.

    """
    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        self.a_in = InPort(shape=(shape[0], 1))
        self.s_out = OutPort(shape=(shape[0], 1))
        self.thresholds = Var(shape=shape, init=kwargs.pop("thresholds", 0))


class SolutionNeurons(AbstractProcess):
    """The neurons that evolve according to the constraint-corrected gradient
    dynamics.
    Implements the abstract behaviour
    qp_neuron_state += (-alpha * (s_in_qc + grad_bias) - beta * s_in_cn)

    Intialize the solutionNeurons process.

    Parameters
    ----------

    shape : int tuple, optional
        A tuple defining the shape of the qp neurons. Defaults to (1,1).
    qp_neurons_init : 1-D np.array, optional
        initial value of qp solution neurons
    grad_bias : 1-D np.array, optional
        The bias of the gradient of the QP. This is the value 'p' in the
        QP definition.
    alpha : 1-D np.array, optional
        Defines the learning rate for gradient descent. Defaults to 1.
    beta : 1-D np.array, optional
        Defines the learning rate for constraint-checking. Defaults to 1.
    alpha_decay_schedule : int, optional
        The number of iterations after which one right shift operation
        takes place for alpha. Default intialization to a very high value
        of 10000.
    beta_growth_schedule : int, optional
        The number of iterations after which one left shift operation takes
        place for beta. Default intialization to a very high value of
        10000.

    """
    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        # In/outPorts that come from/go to the quadratic connectivity process
        self.a_in_qc = InPort(shape=(shape[0], 1))
        self.s_out_qc = OutPort(shape=(shape[0], 1))
        # In/outPorts that come from/go to the constraint normals process
        self.a_in_cn = InPort(shape=(shape[0], 1))
        # OutPort for constraint checking
        self.s_out_cc = OutPort(shape=(shape[0], 1))
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


class ConstraintCheck(AbstractProcess):
    """Check if linear constraints (equality/inequality) are violated for the
    qp.

    Recieves and sends graded spike from and to the gradientDynamics
    process. House the constraintDirections and constraintNeurons as
    sub-processes.

    Implements Abstract behavior:
    (constraint_matrix*x-constraint_bias)*(constraint_matrix*x<constraint_bias)

    Initialize constraintCheck Process.

    Parameters
    ----------

    constraint_matrix : 1-D  or 2-D np.array, optional
    The value of the constraint matrix. This is 'A' in the linear
    constraints.
    constraint_bias : 1-D np.array, optional
        The value of the constraint bias. This is 'k' in the linear
        constraints.
    sparse: bool, optional
        Sparse is true when using sparsifying neuron-model eg. sigma-delta
    x_int_init : 1-D np.array, optional
        initial value of internal sigma neurons

    """
    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        constraint_matrix = kwargs.pop("constraint_matrix", np.zeros((1, 1)))
        shape = constraint_matrix.shape

        constraint_bias = kwargs.pop("constraint_bias", np.zeros((shape[0], 1)))
        self.s_in = InPort(shape=(shape[1], 1))

        x_int_init = kwargs.pop("x_int_init", np.zeros((shape[0], 1)))
        # Creating Vars to get references to vars of subprocesses
        self.constraint_matrix = Var(shape=shape, init=constraint_matrix)
        self.constraint_bias = Var(shape=(shape[0], 1), init=constraint_bias)
        self.s_out = OutPort(shape=(shape[0], 1))

        # Passing arguments to initialize the Subprocesses
        self.proc_params["sparsity"] = kwargs.pop("sparse", False)
        self.proc_params["con_mat"] = constraint_matrix
        self.proc_params["con_bias"] = constraint_bias
        self.proc_params["x_init"] = x_int_init


class GradientDynamics(AbstractProcess):
    """Perform gradient descent with constraint correction to converge at the
    solution of the QP.

    Implements Abstract behavior:
    -alpha*(Q@x_init + p)- beta*A_T@graded_constraint_spike
    Initialize gradientDynamics Process.

    Parameters
    ----------

    hessian : 1-D  or 2-D np.array, optional
        Define the hessian matrix ('Q' in the cost function of the QP) in
        the QP. Defaults to 0.
    constraint_matrix_T : 1-D  or 2-D np.array, optional
        The value of the transpose of the constraint matrix. This is 'A^T'
        in the linear constraints.
    grad_bias : 1-D np.array, optional
        The bias of the gradient of the QP. This is the value 'p' in the QP
        definition.
    qp_neurons_init : 1-D np.array, optional
        Initial value of qp solution neurons
    sparse: bool, optional
        Sparse is true when using sparsifying neuron-model eg. sigma-delta
    model: str, optional
        "SigDel" for sigma delta neurons and "TLIF" for Ternary LIF
        neurons. Defines the type of neuron to be used for sparse activity.
    vth_lo : 1-D np.array, optional
        Defines the lower threshold for TLIF spiking. Defaults to 10.
    vth_hi : 1-D np.array, optional
        Defines the upper threshold for TLIF spiking. Defaults to -10.
    theta : 1-D np.array, optional
        Defines the threshold for sigma-delta spiking. Defaults to 0.
    alpha : 1-D np.array, optional
        Define the learning rate for gradient descent. Defaults to 1.
    beta : 1-D np.array, optional
        Define the learning rate for constraint-checking. Defaults to 1.
    theta_decay_schedule : int, optional
        The number of iterations after which one right shift operation
        takes place for theta. Default intialization to a very high value
        of 10000.
    alpha_decay_schedule : int, optional
        The number of iterations after which one right shift operation
        takes place for alpha. Default intialization to a very high value
        of 10000.
    beta_growth_schedule : int, optional
        The number of iterations after which one left shift operation takes
        place for beta. Default intialization to a very high value of
        10000.

    """
    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        hessian = kwargs.pop("hessian", 0)
        constraint_matrix_T = kwargs.pop("constraint_matrix_T", 0)
        shape_hess = hessian.shape
        shape_constraint_matrix_T = constraint_matrix_T.shape
        grad_bias = kwargs.pop("grad_bias", np.zeros((shape_hess[0], 1)))
        qp_neuron_state = kwargs.pop(
            "qp_neurons_init", np.zeros((shape_hess[0], 1))
        )
        alpha = kwargs.pop("alpha", np.ones((shape_hess[0], 1)))
        beta = kwargs.pop("beta", np.ones((shape_hess[0], 1)))
        theta = kwargs.pop("theta", np.zeros((shape_hess[0], 1)))
        theta_decay_schedule = kwargs.pop("theta_decay_schedule", 10000)
        alpha_decay_schedule = kwargs.pop("alpha_decay_schedule", 10000)
        beta_growth_schedule = kwargs.pop("beta_growth_schedule", 10000)

        self.s_in = InPort(shape=(shape_constraint_matrix_T[1], 1))

        # Creating Vars to get references to vars of subprocesses
        self.hessian = Var(shape=shape_hess, init=hessian)
        self.constraint_matrix_T = Var(
            shape=shape_constraint_matrix_T, init=constraint_matrix_T
        )
        self.grad_bias = Var(
            shape=(shape_hess[0], 1),
            init=grad_bias,
        )
        self.qp_neuron_state = Var(
            shape=(shape_hess[0], 1),
            init=qp_neuron_state,
        )
        self.vth_lo = Var(
            shape=(shape_hess[0], 1), init=kwargs.pop("vth_lo", -10)
        )
        self.vth_hi = Var(
            shape=(shape_hess[0], 1), init=kwargs.pop("vth_hi", 10)
        )
        self.theta = Var(
            shape=(shape_hess[0], 1),
            init=theta,
        )
        self.alpha = Var(shape=(shape_hess[0], 1), init=alpha)
        self.beta = Var(shape=(shape_hess[0], 1), init=beta)
        self.theta_decay_schedule = Var(shape=(1, 1), init=theta_decay_schedule)
        self.alpha_decay_schedule = Var(shape=(1, 1), init=alpha_decay_schedule)
        self.beta_growth_schedule = Var(shape=(1, 1), init=beta_growth_schedule)

        self.s_out = OutPort(shape=(shape_hess[0], 1))

        # Passing arguments to initialize the Subprocesses
        self.proc_params["hess"] = hessian
        self.proc_params["grd_bs"] = grad_bias
        self.proc_params["qp_neur_init"] = qp_neuron_state
        self.proc_params["con_mat_T"] = constraint_matrix_T
        self.proc_params["sparsity"] = kwargs.pop("sparse", False)
        self.proc_params["mod"] = kwargs.pop("model", "SigDel")
        self.proc_params["thet"] = theta
        self.proc_params["al"] = alpha
        self.proc_params["bet"] = beta
        self.proc_params["thet_dec"] = theta_decay_schedule
        self.proc_params["al_dec"] = alpha_decay_schedule
        self.proc_params["bet_dec"] = beta_growth_schedule


class ProjectedGradientNeuronsPIPGeq(AbstractProcess):
    """The neurons that evolve according to the projected gradient
    dynamics specified in the PIPG algorithm.

    Do NOT use QPDense connection
    process with this solver. Use the standard Lava Dense process.
    Intialize the ProjectedGradientNeuronsPIPGeq process.
    Implements the abstract behaviour
    qp_neuron_state -= alpha*(a_in + grad_bias)

    Parameters
    ----------

    shape : int tuple, optional
        A tuple defining the shape of the qp neurons. Defaults to (1,).
    da_exp: int, optional
        Exponent of base 2 used to scale magnitude at dendritic accumulator,
        if needed.
        The correction exponent (min(Connectivity_exp, constraint_exp)) global
        has to be passed in to this parameter to right shift the dendritic
        accumulator. Value can only be -ve Used for fixed point implementations.
        Unnecessary for floating point implementations. Default value is 0.
    qp_neurons_init : 1-D np.array, optional
        initial value of qp solution neurons
    grad_bias : 1-D np.array, optional
        The bias of the gradient of the QP. This is the value 'p' in the
        QP definition.
    grad_bias_exp : int, optional
        Shared exponent of base 2 used to scale magnitude of
        the grad_bias vector, if needed. Value can only be -ve.
        Mostly for fixed point implementations. Unnecessary for floating point
        implementations. Default value is 0.
    alpha : 1-D np.array, optional
        Defines the learning rate for gradient descent. Defaults to 1.
    alpha_exp : int, optional
        Exponent of base 2 used to scale magnitude of
        alpha, if needed. Value can only be -ve. Mostly for fixed point
        implementations. Unnecessary for floating point implementations.
        Default value is 0.
    lr_decay_type: string, optional
        Defines the nature of the learning rate, alpha's decay. "schedule"
        decays it for every alpha_decay_schedule timesteps. "indices"
        halves the learning rate for every timestep defined in
        alpha_decay_indices.
    alpha_decay_schedule : int, optional
        The number of iterations after which one right shift operation
        takes place for alpha. Default intialization to a very high value
        of 10000.
    alpha_decay_indices: list, optional
        The timesteps at which the learning rate, alpha, halves. By default an
        empty list.
    alpha_decay_params: tuple, optional
        The indices at which value of alpha gets halved (right-shifted).
        The tuple contains (decay_index, decay_interval, decay_factor).
        Default values are (1,1,1).  Note that if decay_index is set to 0,
        it is automatically set to one. Setting decay_index to zero is not
        allowable behavior, instead change the initial value of alpha if
        decay has to take place at the 0th timestep.
        The index is calculated using the formula.
        decay_factor = decay_factor + 1
        decay_index = decay_index + decay_interval*decay_factor
        This mimics the hyperbolic decrease of learning rate, alpha, with only
        right shifts at particular intervals.

    """
    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1,))

        # # Ports
        # # In/outPorts that come from/go to the quadratic connectivity process
        self.a_in = InPort(shape=(shape[0],))
        self.s_out = OutPort(shape=(shape[0],))

        # Keyword Readouts
        grad_bias = kwargs.pop("grad_bias", np.zeros(shape))
        da_exp = kwargs.pop("da_exp", 0)
        grad_bias_exp = kwargs.pop("grad_bias_exp", 0)
        alpha = kwargs.pop("alpha", np.ones((shape[0],)).astype(np.int32))
        alpha_exp = kwargs.pop("alpha_exp", 0)
        lr_decay_type = kwargs.pop("lr_decay_type", "schedule")
        decay_index, decay_interval, decay_factor = kwargs.pop(
            "alpha_decay_params", (1, 1, 1)
        )
        alpha_decay_indices = kwargs.pop("alpha_decay_indices", [])
        if decay_index == 0:
            decay_index = 1

        self.proc_params["shape_nc"] = shape
        self.proc_params["lr_dec_type"] = lr_decay_type
        self.proc_params["alpha_dec_params"] = (
            decay_index,
            decay_interval,
            decay_factor,
        )
        self.proc_params["alpha_dec_list"] = alpha_decay_indices

        if lr_decay_type == "indices":
            decay_flag = 1
        else:
            decay_flag = 0

        # Vars
        self.qp_neuron_state = Var(
            shape=shape,
            init=kwargs.pop(
                "qp_neurons_init", np.zeros(shape).astype(np.int32)
            ),
        )
        self.grad_bias = Var(shape=shape, init=grad_bias)
        self.alpha = Var(shape=shape, init=alpha)
        self.alpha_decay_schedule = Var(
            shape=(1,), init=kwargs.pop("alpha_decay_schedule", 10000)
        )
        # Extra Vars for NcProcModels
        # changing all exponents (assumed to be -ve) to have positive sign in
        # order to enable right shift in microcode
        self.grad_bias_exp = Var(shape=(1,), init=-grad_bias_exp)
        self.grad_bias_man = Var(shape=shape, init=grad_bias.astype(np.int32))
        self.da_exp = Var(shape=(1,), init=-da_exp)
        self.alpha_exp = Var(shape=(1,), init=-alpha_exp)
        self.alpha_man = Var(shape=shape, init=alpha)
        self.decay_inter = Var(shape=(1,), init=decay_interval)
        self.decay_index = Var(shape=shape, init=decay_index)
        self.decay_factor = Var(shape=shape, init=decay_factor)


class ProportionalIntegralNeuronsPIPGeq(AbstractProcess):
    """The neurons that evolve according to the proportional integral
    dynamics specified in the PIPG algorithm. Do NOT use QPDense connection
    process with this solver. Use the standard Lava Dense process.
    Implements the abstract behaviour.

    constraint_neuron_state += beta * (a_in - constraint_bias)
    s_out = constraint_neuron_state + beta * (a_in - constraint_bias)

    Intialize the ProportionalIntegralNeuronsPIPGeq process.

    Parameters
    ----------

    shape : int tuple, optional
        A tuple defining the shape of the qp neurons. Defaults to (1,).
    da_exp: int, optional
        Exponent of base 2 used to scale magnitude at dendritic accumulator,
        if needed. The global exponent of the constraint matrix (based on
        max element), A, has to be passed in to this parameter. Value can only
        be -ve. Used for fixed point implementations. Unnecessary for floating
        point implementations. Default value is 0.
    constraint_neurons_init : 1-D np.array, optional
        Initial value of constraint neurons
    thresholds : 1-D np.array, optional
        Define the thresholds of the neurons in the
        constraint checking layer. This is usually 'k' in the constraints
        of the QP. Default value of thresholds is 0.
    thresholds_exp : int, optional
        Shared exponent of base 2 used to scale magnitude of
        the thresholds vector, if needed. Mostly for fixed point
        implementations. Value can only be -ve
        Unnecessary for floating point implementations.
        Default value is 0.
    beta : 1-D np.array, optional
        Defines the learning rate for constraint-checking. Defaults to 1.
    beta_exp : int, optional
        Exponent of base 2 used to scale magnitude of
        beta, if needed. Value can only be -ve. Mostly for fixed point
        implementations. Unnecessary for floating point implementations.
        Default value is 0.
    lr_growth_type: string, optional
        Defines the nature of the learning rate, beta's growth. "schedule"
        grows it for every beta_growth_schedule timesteps. "indices"
        doubles the learning rate for every timestep defined in
        beta_growth_indices.
    beta_growth_schedule : int, optional
        The number of iterations after which one left shift operation takes
        place for beta. Default intialization to a very high value of
        10000.
    beta_growth_indices : list, optional
        The timesteps at which the learning rate, beta, doubles. By default an
        empty list.
    beta_growth_params: tuple of ints, optional
        The indices at which value of beta gets doubled (left-shifted).
        The tuple contains (growth_index, growth_factor).
        Default values are (1,1). Note that if growth_index is set to 0,
        it is automatically set to one. Setting growth_index to zero is not
        allowable behavior, instead change the initial value of beta if
        growth has to take place at the 0th timestep.
        growth_interval is hard-coded as 2
        The index is calculated using the formula.
        growth_factor = growth_factor + 1
        N_growth = N_growth + growth_interval*growth factor
        This mimics the linear increase of learning rate beta with only
        left shifts at particular intervals.

    """
    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1,))

        # Ports
        self.a_in = InPort(shape=(shape[0],))
        self.s_out = OutPort(shape=(shape[0],))

        # Keyword Readouts
        thresholds = kwargs.pop("thresholds", np.zeros(shape).astype(np.int32))
        da_exp = kwargs.pop("da_exp", 0)
        con_bias_exp = kwargs.pop("thresholds_exp", 0)
        beta = kwargs.pop("beta", np.ones((shape[0],)).astype(np.int32))
        beta_exp = kwargs.pop("beta_exp", 0)
        lr_growth_type = kwargs.pop("lr_growth_type", "schedule")
        growth_index, growth_factor = kwargs.pop("beta_growth_params", (1, 1))
        beta_growth_indices = kwargs.pop("beta_growth_indices", [])

        if growth_index == 0:
            growth_index = 1

        if growth_factor == 0:
            growth_factor = 1

        self.proc_params["shape_nc"] = shape
        self.proc_params["lr_grw_type"] = lr_growth_type
        self.proc_params["beta_grw_params"] = (
            growth_index,
            growth_factor,
        )
        self.proc_params["beta_grw_list"] = beta_growth_indices

        if lr_growth_type == "indices":
            growth_flag = 1
        else:
            growth_flag = 0

        # Vars
        self.constraint_neuron_state = Var(
            shape=shape,
            init=kwargs.pop(
                "constraint_neurons_init", np.zeros(shape).astype(np.int32)
            ),
        )
        self.constraint_bias = Var(shape=shape, init=thresholds)
        self.beta = Var(shape=shape, init=beta)
        self.beta_growth_schedule = Var(
            shape=(1,), init=kwargs.pop("beta_growth_schedule", 10000)
        )

        # Extra Vars for NcProcModels
        # changing all exponents (assumed to be -ve) to have positive sign in
        # order to enable right shift in microcode
        self.con_bias_exp = Var(shape=(1,), init=-con_bias_exp)
        self.con_bias_man = Var(shape=shape, init=thresholds)
        self.beta_exp = Var(shape=(1,), init=-beta_exp)
        self.da_exp = Var(shape=(1,), init=-da_exp)
        self.beta_man = Var(shape=shape, init=beta)
        self.growth_inter = Var(shape=(1,), init=2)
        self.growth_index = Var(shape=shape, init=growth_index)
        self.growth_factor = Var(shape=shape, init=growth_factor)


class SigmaNeurons(AbstractProcess):
    """Process to accumate spikes into a state variable before being fed to
     another process.

     Realizes the following abstract behavior:
     a_out = self.x_internal + s_in

    Parameters
    ----------

    shape : int tuple, optional
        Define the shape of the thresholds vector. Defaults to (1,1).
    x_sig_init : 1-D np.array, optional
        initial value of internal sigma neurons. Should be the same as
        qp_neurons_init. Default value is 0.

    """
    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        self.s_in = InPort(shape=(shape[0], 1))
        self.s_out = OutPort(shape=(shape[0], 1))
        # should be same as x_int_
        self.x_internal = Var(shape=shape, init=kwargs.pop("x_sig_init", 0))


class DeltaNeurons(AbstractProcess):
    """Process to simulate Delta coding.

    A graded spike is sent only if the
    difference delta for a neuron exceeds the spiking threshold, Theta
    Realizes the following abstract behavior:
    delta = np.abs(s_in - self.x_internal)
    s_out =  delta[delta > theta]

    Parameters
    ----------

    shape : int tuple, optional
        Define the shape of the thresholds vector. Defaults to (1,1).
    x_del_init : 1-D np.array, optional
        initial value of internal delta neurons. Should be the same as
        qp_neurons_init. Default value is 0.
    theta : 1-D np.array, optional
        Defines the learning rate for gradient descent. Defaults to 1.
    theta_decay_type: string, optional
        Defines the nature of the learning rate, theta's decay. "schedule"
        decays it for every theta_decay_schedule timesteps.
        "indices" halves the learning rate for every timestep defined
        in alpha_decay_indices.
    theta_decay_schedule : int, optional
        The number of iterations after which one right shift operation
        takes place for theta. Default intialization to a very high value
        of 10000.
    theta_decay_indices: list, optional
        The iteration numbers at which value of theta gets halved
        (right-shifted).

    """
    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        self.s_in = InPort(shape=(shape[0], 1))
        self.s_out = OutPort(shape=(shape[0], 1))
        self.x_internal = Var(shape=shape, init=kwargs.pop("x_del_init", 0))
        self.theta = Var(
            shape=shape, init=kwargs.pop("theta", np.ones((shape[0], 1)))
        )
        self.theta_decay_schedule = Var(
            shape=(1, 1), init=kwargs.pop("theta_decay_schedule", 10000)
        )
        self.proc_params["theta_decay_indices"] = kwargs.pop(
            "theta_decay_indices", [10000]
        )
        self.proc_params["theta_decay_type"] = kwargs.pop(
            "theta_decay_type", "schedule"
        )
