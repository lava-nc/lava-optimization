#  Copyright (C) 2023 Battelle Memorial Institute
#  SPDX-License-Identifier: BSD-2-Clause
#  See: https://spdx.org/licenses/

import typing as ty

import numpy as np
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var


class V1Neuron(AbstractProcess):
    """V1 Neurons used in 1 layer and 2 layer LCA. See corresponding LCA
    processes for full dynamics.

    Parameters
    ----------
    vth: activation threshold
    tau: time constant
    bias: bias applied every timestep for 1 layer dynamics
    two_layer: If false, use 1 layer dynamics, otherwise use 2 layer dynamics
    """

    def __init__(self,
                 vth: float,
                 tau: float,
                 tau_exp: int,
                 shape: ty.Optional[tuple] = (1,),
                 bias: ty.Optional[ty.Union[int, np.ndarray]] = 0,
                 two_layer: ty.Optional[bool] = True,
                 **kwargs) -> None:
        super().__init__(shape=shape,
                         vth=vth,
                         tau=tau,
                         tau_exp=tau_exp,
                         two_layer=two_layer,
                         **kwargs)

        self.vth = Var(shape=(1,), init=vth)
        self.tau = Var(shape=(1,), init=tau)
        self.tau_exp = Var(shape=(1,), init=tau_exp)
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.v = Var(shape=shape)
        self.bias = Var(shape=shape, init=bias)
