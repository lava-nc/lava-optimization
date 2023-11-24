# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import numpy as np
import typing as ty
from numpy import typing as npty

from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var


class Annealing(AbstractProcess):
    """
    Neuron that updates the temperature for simulated annealing.

    Parameters
        ----------
        shape: Tuple
            Number of neurons. Default is (1,).
        max_temperature: int, ArrayLike
            Both maximum and initial temperature of the annealing schedule.
            The temperature defines the noise of the system
        min_temperature: ArrayLike
            Minimum temperature of the annealing schedule.
        steps_per_temperature: int, ArrayLike
            The number of time steps between two annealing steps.
        delta_temperature: ArrayLike
            Defines the change in temperature in each annealing step.
            If annealing_schedule is 'linear', the temperature is decreased by
                temperature -= delta_temperature .
            If annealing_schedule is 'geometric', the temperature is changed by
                temperature *= delta_temperature * 2^(exp_temperature) .
        exp_temperature: ArrayLike
            Defines the change in temperature in each annealing step. For
            details, refer to 'delta_temperature'
        annealing_schedule: str
            Defines the annealing schedule. Supported values are 'linear' and
            'geometric'.
    """

    # annealing schedules that are currently supported
    supported_anneal_schedules = ['linear', 'geometric']

    def __init__(
        self,
        *,
        max_temperature: ty.Union[int, npty.NDArray],
        min_temperature: ty.Union[int, npty.NDArray],
        delta_temperature: ty.Union[int, npty.NDArray],
        steps_per_temperature: ty.Union[int, npty.NDArray],
        exp_temperature: ty.Union[int, npty.NDArray],
        annealing_schedule: str,
        shape: ty.Tuple[int, ...] = (1,),
    ):

        self._validate_input(
            shape=shape,
            min_temperature=min_temperature,
            max_temperature=max_temperature,
            delta_temperature=delta_temperature,
            steps_per_temperature=steps_per_temperature,
            exp_temperature=exp_temperature,
            annealing_schedule=annealing_schedule,
        )

        super().__init__(
            shape=shape,
            min_temperature=min_temperature,
            max_temperature=max_temperature,
            delta_temperature=delta_temperature,
            steps_per_temperature=steps_per_temperature,
            exp_temperature=exp_temperature,
            annealing_schedule=annealing_schedule,
        )

        self.delta_temperature_out = OutPort(shape=shape)

        self.temperature = Var(shape=shape, init=np.int_(max_temperature))

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        return self.proc_params["shape"]

    def _validate_input(self, shape, min_temperature, max_temperature,
                        delta_temperature, steps_per_temperature,
                        exp_temperature, annealing_schedule) -> None:
        """Validates input to the annealing neuron."""

        if min_temperature < 0:
            raise ValueError("min_temperature must be >= 0.")
        if max_temperature > 2**(16) - 1:
            raise ValueError("max_temperature must be < 2^16 - 1")
        if min_temperature > max_temperature:
            raise ValueError("max_temperature must be >= min_temperature.")
        if delta_temperature < 0:
            raise ValueError("delta_temperature must be >=0.")
        if annealing_schedule == 'geometric' and exp_temperature < 0:
            raise ValueError("exp_temperature must be >=0.")
        if annealing_schedule not in self.supported_anneal_schedules:
            raise ValueError(f"At the moment only the annealing schedules "
                             f"{self.supported_anneal_schedules} are "
                             f"supported.")
        if steps_per_temperature < 0:
            raise ValueError(f"steps_per_temperature is "
                             f"{steps_per_temperature} but must be > 0.")
        if annealing_schedule == 'geometric':
            geometric_constant = np.right_shift(delta_temperature,
                                                exp_temperature)
            if geometric_constant > 1 or geometric_constant < 0:
                raise ValueError(f"delta_temperature >> exp_temperature "
                                 f"should be between 0 to 1, but is"
                                 f" {geometric_constant}.")
