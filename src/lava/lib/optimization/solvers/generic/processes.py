# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty

import numpy as np
from lava.lib.optimization.problems.coefficients import CoefficientTensorsMixin
from lava.magma.core.process.interfaces import AbstractProcessMember
from lava.magma.core.process.ports.ports import InPort
from lava.magma.core.process.variable import Var


def _vars_from_coefficients(
    coefficients: CoefficientTensorsMixin,
) -> ty.Dict[int, AbstractProcessMember]:
    vars = dict()
    for rank, coeff in coefficients.items():
        if rank == 1:
            init = -coeff
        if rank == 2:
            linear_component = -coeff.diagonal()
            quadratic_component = coeff * np.logical_not(np.eye(*coeff.shape))
            if 1 in vars.keys():
                vars[1].init = vars[1].init + linear_component
            else:
                vars[1] = Var(
                    shape=linear_component.shape, init=linear_component
                )
            init = -quadratic_component
        vars[rank] = Var(shape=coeff.shape, init=init)
    return vars


def _in_ports_from_coefficients(
    coefficients: CoefficientTensorsMixin,
) -> ty.List[AbstractProcessMember]:
    in_ports = [
        InPort(shape=coeff.shape) for coeff in coefficients.coefficients
    ]
    return in_ports
