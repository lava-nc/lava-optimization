#  Copyright (C) 2023 Battelle Memorial Institute
#  SPDX-License-Identifier: BSD-2-Clause
#  See: https://spdx.org/licenses/

import typing as ty

import numpy as np
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var


class ResidualNeuron(AbstractProcess):
    """Accumulates all input and bias into voltage until it exceeds
    spike_height, then fires and resets.

    Parameters
    ----------
    spike_height: the threshold to fire and reset at
    bias: added to voltage every timestep
    """

    def __init__(self,
                 spike_height: float,
                 bias: ty.Union[int, np.ndarray],
                 shape: ty.Optional[tuple] = (1,),
                 **kwargs) -> None:
        super().__init__(shape=shape,
                         spike_height=spike_height,
                         **kwargs)

        self.spike_height = Var(shape=(1,), init=spike_height)
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.v = Var(shape=shape)
        self.bias = Var(shape=shape, init=bias)
