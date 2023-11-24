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
        backend,
        hyperparameters,
        discrete_var_shape,
        continuous_var_shape,
        problem,
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
    ):
        super().__init__(
            cost_diagonal=cost_diagonal,
            cost_coefficients=cost_coefficients,
            constraints=constraints,
            backend=backend,
            hyperparameters=hyperparameters,
            discrete_var_shape=discrete_var_shape,
            continuous_var_shape=continuous_var_shape,
            problem=problem,
            name=name,
            log_config=log_config,
        )
        self.variables_assignment = Var(
            shape=discrete_var_shape or continuous_var_shape, init=(1,)
        )
        self.cost_last_bytes = Var(shape=(1,), init=(0,))
        self.cost_first_byte = Var(shape=(1,), init=(0,))
        self.cost_out_last_bytes = OutPort(shape=(1,))
        self.cost_out_first_byte = OutPort(shape=(1,))

    def reconfigure_cost_coefficients(self, second_order_coeff: np.ndarray):
        print("MODEL                        ", self.model_class)
        if self.cost_minimizer is None:
            raise Error("Cost coefficients cannot be reconfigured before the proc model is created.")

        self.cost_minimizer.coefficients_2nd_order.weights.set(
            second_order_coeff * np.logical_not(np.eye(*second_order_coeff.shape))
        )
        #self.variables.importances.set(
        #    second_order_coeff.diagonal()
        #)
