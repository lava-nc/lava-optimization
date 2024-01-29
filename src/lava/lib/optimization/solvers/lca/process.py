#  Copyright (C) 2023 Battelle Memorial Institute
#  SPDX-License-Identifier: BSD-2-Clause
#  See: https://spdx.org/licenses/

import typing as ty

import numpy as np
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import OutPort


class LCA1Layer(AbstractProcess):
    """Implements LCA based on https://doi.org/10.1162/neco.2008.03-07-486
    LCA minimizes sparse coding |a|_1 such that Φa ≈ s for some dictionary 'Φ'
    and vector to reconstruct 's' by using a set of neurons with voltage v and
    the dynamics below. As LCA accounts for the response properties of V1 simple
    cells, we refer to these neurons as v1.

    V1 Dynamics
    -----------
    dv = -v + a(-Φ^T*Φ+I) + Φ^T*s
    a = soft_threshold(v)

    Parameters
    ----------
    weights: Excitation / Inhibition weights mantissa - Φ^T*Φ-I * tau
    bias: Neuron bias - Φ^T*s
    weights_exp: Weights exponent
    threshold: neuron activation threshold
    tau: time constant mantissa
    tau_exp: time constant exponent
    """

    def __init__(
            self,
            weights: np.ndarray,
            bias: np.ndarray,
            weights_exp: ty.Optional[int] = 0,
            threshold: ty.Optional[float] = 1,
            tau: ty.Optional[float] = 0.1,
            tau_exp: ty.Optional[int] = 0,
            **kwargs) -> None:
        super().__init__(**kwargs)

        self.threshold = Var(shape=(1,), init=threshold)
        self.tau = Var(shape=(1,), init=tau)
        self.tau_exp = Var(shape=(1,), init=tau_exp)
        self.weights = Var(shape=weights.shape, init=weights)
        self.weights_exp = Var(shape=(1,), init=weights_exp)
        self.v1 = OutPort(shape=(weights.shape[0],))
        self.voltage = Var(shape=(weights.shape[0],))
        self.bias = Var(shape=bias.shape, init=bias)


class LCA2Layer(AbstractProcess):
    """Implements of 2-Layer LCA based on https://arxiv.org/abs/2205.15386
    LCA minimizes sparse coding |a|_1 such that Φa ≈ s for some dictionary 'Φ'
    and vector to reconstruct 's'. In two layer LCA, the reconstruction error Φa
    is separated out into its own layer r=s-Φa. This residual is made spiking by
    accumulating the error and spiking if it exceeds some spike_height.

    V1 Layer Dynamics
    -----------------
    dv = -v + Φ^T(R_spike) + a
    a = soft_threshold(v)

    Residual Layer Dynamics
    -----------------------
    r += s-Φa
    if |r| > spike_height
        R_spike = r
        r = 0

    Parameters
    ----------
    weights: Excitation / Inhibition weights mantissa - Φ
    input_vec: Input to reconstruct - s
    weights_exp: Weights exponent
    input_exp: Input exponent
    threshold: Neuron activation threshold
    tau: Time constant mantissa
    tau_exp: Time constant exponent
    spike_height: Accumulator spike height
    """

    def __init__(
            self,
            weights: np.ndarray,
            input_vec: np.ndarray,
            weights_exp: ty.Optional[int] = 0,
            input_exp: ty.Optional[int] = 0,
            threshold: ty.Optional[float] = 1,
            tau: ty.Optional[float] = 0.1,
            tau_exp: ty.Optional[int] = 0,
            spike_height: ty.Optional[int] = 1,
            **kwargs) -> None:
        super().__init__(**kwargs)

        self.threshold = Var(shape=(1,), init=threshold)
        self.tau = Var(shape=(1,), init=tau)
        self.tau_exp = Var(shape=(1,), init=tau_exp)
        self.weights = Var(shape=weights.shape, init=weights)
        self.weights_exp = Var(shape=(1,), init=weights_exp)
        self.v1 = OutPort(shape=(weights.shape[0],))
        self.res = OutPort(shape=(weights.shape[1],))
        self.voltage = Var(shape=(weights.shape[0],))
        self.spike_height = Var(shape=(1,), init=spike_height)
        self.input = Var(shape=input_vec.shape, init=input_vec)
        self.input_exp = Var(shape=input_vec.shape, init=input_exp)
