# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty

import numpy as np
from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var


class SolutionFinder(AbstractProcess):
    def __init__(
        self,
        cost_diagonal,
        cost_coefficients,
        constraints,
        hyperparameters,
        discrete_var_shape,
        continuous_var_shape,
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
    ):
        super().__init__(
            cost_diagonal=cost_diagonal,
            cost_coefficients=cost_coefficients,
            constraints=constraints,
            hyperparameters=hyperparameters,
            discrete_var_shape=discrete_var_shape,
            continuous_var_shape=continuous_var_shape,
            name=name,
            log_config=log_config,
        )
        self.variables_assignment = Var(shape=discrete_var_shape, init=(1,))
        self.cost_last_bytes = Var(shape=(1,), init=(0,))
        self.cost_first_byte = Var(shape=(1,), init=(0,))
        self.cost_out_last_bytes = OutPort(shape=(1,))
        self.cost_out_first_byte = OutPort(shape=(1,))
