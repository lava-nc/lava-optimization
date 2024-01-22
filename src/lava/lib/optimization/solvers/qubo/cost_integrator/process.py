# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty

import numpy as np
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var


class CostIntegrator(AbstractProcess):
    """Node that integrates cost components and produces output when a better
    cost is found.

    Parameters
    ----------
    shape : tuple(int)
        The expected number and topology of the input cost components.
    name : str, optional
        Name of the Process. Default is 'Process_ID', where ID is an
        integer value that is determined automatically.
        log_config: Configuration options for logging.

    InPorts
    -------
    cost_in
        input to be additively integrated.

    OutPorts
    --------
    cost_out_last_bytes: OutPort
        Notifies the next process about the detection of a better cost.
        Messages the last 3 byte of the new best cost.
        Total cost = cost_out_first_byte << 24 + cost_out_last_bytes.
    cost_out_first_byte: OutPort
        Notifies the next process about the detection of a better cost.
        Messages the first byte of the new best cost.

    Vars
    ----
    cost
        Holds current cost as addition of input spikes' payloads

    cost_min_last_bytes
        Current minimum cost, i.e., the lowest reported cost so far.
        Saves the last 3 bytes.
        cost_min = cost_min_first_byte << 24 + cost_min_last_bytes
    cost_min_first_byte
        Current minimum cost, i.e., the lowest reported cost so far.
        Saves the first byte.
    """

    def __init__(
        self,
        *,
        shape: ty.Tuple[int, ...] = (1,),
        target_cost: int = -2**31 + 1,
        timeout: int = 2**24 - 1,
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
    ) -> None:

        self._input_validation(target_cost=target_cost,
                               timeout=timeout)

        super().__init__(shape=shape,
                         target_cost = target_cost,
                         timeout = timeout,
                         name=name,
                         log_config=log_config)
        self.cost_in = InPort(shape=shape)
        self.control_states_out = OutPort(shape=shape)
        self.best_cost_out = OutPort(shape=shape)
        self.best_timestep_out = OutPort(shape=shape)

        # Counter for timesteps
        self.timestep = Var(shape=shape, init=0)
        # Storage for best current time step
        self.best_timestep = Var(shape=shape, init=0)

        # Var to store current cost
        # Note: Total cost = cost_first_byte << 24 + cost_last_bytes
        # last 24 bit of cost
        self.cost_last_bytes = Var(shape=shape, init=0)
        # first 8 bit of cost
        self.cost_first_byte = Var(shape=shape, init=0)

        # Var to store best cost found to far
        # Note: Total min cost = cost_min_first_byte << 24 + cost_min_last_bytes
        # last 24 bit of cost
        self.cost_min_last_bytes = Var(shape=shape, init=0)
        # first 8 bit of cost
        self.cost_min_first_byte = Var(shape=shape, init=0)

    @staticmethod
    def _input_validation(target_cost, timeout) -> None:

        assert (target_cost is not None and timeout is not None), \
            f"Both the target_cost and the timeout must be defined"
        assert 0 > target_cost >= -2**31 + 1, \
            f"The target cost must in the range [-2**32 + 1, 0), " \
            f"but is {target_cost}."
        assert 0 < timeout <= 2**24 - 1, f"The timeout must be in the range (" \
                                         f"0, 2**24 - 1], but is {timeout}."