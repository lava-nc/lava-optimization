# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
import numpy as np

class constraintDirections(AbstractProcess):
    """Connections in the constraint-checking group of neurons. 
    Realizes the following abstract behavior:
    a_out = A * s_in
    """

    def __init__(self, **kwargs):
        """intialize the constraintDirectionsProcess
        
        Kwargs:
            shape (int tuple): Define the shape of the connections 
            matrix as a tuple. Defaults to (1,1)
            constraint_directions (1-D  or 2-D np.array): Define the directions 
            of the linear constraint hyperplanes. This is 'A' in the constraints 
            of the QP. Defaults to 0   
        """
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        self.s_in = InPort(shape=(shape[1],))      
        self.a_out = OutPort(shape=(shape[0],))
        self.weights = Var(shape=shape, 
                           init=kwargs.pop("constraint_directions", 0)
                           )

class constraintNeurons(AbstractProcess):
    """Process to check the violation of the linear constraints of the QP. A 
    graded spike corresponding to the violated constraint is sent from the out 
    port.
    """

    def __init__(self, **kwargs):
        """Intialize the constraintNeurons Process.

        Kwargs:
            shape (int tuple): Define the shape of the connections 
            matrix as a tuple. Defaults to (1,1).
            thresholds (1-D np.array): Define the thresholds of the neurons 
            in the constraint checking layer. This is usually 'b' in the 
            constraints of the QP. Default value of thresholds is 0.
        """
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        self.s_in = InPort(shape=(shape[0],))      
        self.a_out = OutPort(shape=(shape[0],))
        self.threshold = Var(shape=shape, 
                             init=kwargs.pop("thresholds", 0)
                            )



class quadraticConnectivity(AbstractProcess):
    """The connections that define the Hessian of the quadratic cost function 
    Realizes the following abstract behavior:
    a_out = A * s_in
    """

    def __init__(self, **kwargs):
        """Intialize the quadraticConnectivity process.
        
        Kwargs:
            shape (int tuple): A tuple defining the shape of the connections 
            matrix. Defaults to (1,1).
            hessian (1-D  or 2-D np.array): Define the hessian matrix ('P' in 
            the cost function of the QP) in the QP. Defaults to 0.   
        """
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        self.s_in = InPort(shape=(shape[1],))      
        self.a_out = OutPort(shape=(shape[0],))
        self.weights = Var(shape=shape, 
                           init=kwargs.pop("hessian", 0))

class solutionNeurons(AbstractProcess):
    """The neurons that evolve according to the constraint-corrected gradient
    dynamics. 
    """

    def __init__(self, **kwargs):
        """Intialize the solutionNeurons process.
        
        Kwargs:
            shape (int tuple): A tuple defining the shape of the qp neurons. 
            Defaults to (1,1).
            qp_neurons_init (1-D np.array): initial value of qp solution neurons
            grad_bias (1-D np.array): The bias of the gradient of the QP. This 
            is the value 'p' in the QP definition.          
        """
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        # In/outPorts that come from/go to the quadratic connectivity process 
        self.s_in_qc = InPort(shape=(shape[0],))      
        self.a_out_qc = OutPort(shape=(shape[0],))
        # In/outPorts that come from/go to the constraint normals process
        self.s_in_cn = InPort(shape=(shape[0],))      
        self.a_out_cn = OutPort(shape=(shape[0],))
        # Inports for learning constants
        self.s_in_alpha = InPort(shape=(shape[1],))
        self.s_in_beta = InPort(shape=(shape[1],))  
        # OutPort for constraint checking  
        self.a_out_cc = OutPort(shape=(shape[0],))
        self.qp_neuron_state = Var(shape=shape,
                                   init=kwargs.pop("qp_neurons_init", 
                                                    np.zeros((shape))
                                                  )
                                   )
        self.grad_bias = Var(shape=shape, 
                             init=kwargs.pop("grad_bias", 
                                              np.zeros((shape))
                                            )
                            )

class constraintNormals(AbstractProcess):
    """Connections influencing the gradient dynamics when constraints are 
    violated. 
    Realizes the following abstract behavior:
    a_out = A * s_in
    """

    def __init__(self, **kwargs):
        """
        intialize the constraint normals to assign weights to constraint 
        violation spikes.

        Kwargs:
            shape (int tuple): A tuple defining the shape of the connections 
            matrix. Defaults to (1,1).
            constraint_directions (1-D  or 2-D np.array): Define the normals 
            of the linear constraint hyperplanes. This is A^T in the constraints 
            of the QP. Defaults to 0   
        """
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        self.s_in = InPort(shape=(shape[1],))      
        self.a_out = OutPort(shape=(shape[0],))
        self.weights = Var(shape=shape, 
                           init=kwargs.pop("constraint_normals", 0)
                           )

class learningConstantAlpha(AbstractProcess):
    """The learning constants that define the speed of convergence of the 
    vanilla gradient descent
    """

    def __init__(self, **kwargs):
        """
        Intialize the values of alpha, beta and the decay/growth schedules.
        
        Kwargs:
            shape (int tuple): A tuple defining the shape of the connections 
            matrix. Defaults to (1,1).
            alpha (1-D np.array): Defines the learning rate for gradient 
            descent. Defaults to 1.
            alpha_decay_schedule: The number of iterations after which one right 
            shift operation takes place for alpha. Default intialization 
            to a very high value of 10000.
        """
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))      
        self.a_out = OutPort(shape=(shape[0],))
        self.alpha = Var(shape=shape, 
                         init=kwargs.pop("alpha", 1))                 
        self.alpha_decay_schedule = Var(shape=shape, 
                                        init=kwargs.pop("alpha_decay_schedule", 
                                                        10000
                                                        )
                                        )  
        self.decay_counter = Var(shape=shape, 
                                  init=0)

class learningConstantBeta(AbstractProcess):
    """The learning constants that define the magnitude of correction from 
    constraint checking
    """

    def __init__(self, **kwargs):
        """
        Intialize the values of beta and the growth schedules.
        
        Kwargs:
            shape (int tuple): A tuple defining the shape of the connections 
            matrix. Defaults to (1,1).
            beta (1-D np.array): Defines the learning rate for constraint-
            checking. Defaults to 1.
            beta_growth_schedule: The number of iterations after which one left 
            shift operation takes place for beta. Default intialization 
            to a very high value of 10000. 
        """
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))      
        self.a_out = OutPort(shape=(shape[0],))                
        self.beta = Var(shape=shape, 
                        init=kwargs.pop("beta", 1))
        self.beta_growth_schedule = Var(shape=shape, 
                                        init=kwargs.pop("beta_growth_schedule", 
                                                        10000
                                                        )
                                        )
        self.growth_counter = Var(shape=shape, 
                                  init=0)