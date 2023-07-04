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
        min_cost: int = 0,  # trivial solution, where all variables are 0
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
    ) -> None:
        super().__init__(shape=shape, name=name, log_config=log_config)
        self.cost_in = InPort(shape=shape)
        self.cost_out_last_bytes = OutPort(shape=shape)
        self.cost_out_first_byte = OutPort(shape=shape)

        # Current cost initiated to zero
        # Note: Total cost = cost_first_byte << 24 + cost_last_bytes
        self.cost_last_bytes = Var(shape=shape, init=0)
        # first 8 bit of cost
        self.cost_first_byte = Var(shape=shape, init=0)

        # Note: Total min cost = cost_min_first_byte << 24 + cost_min_last_bytes
        # Extract first 8 bit
        cost_min_first_byte = np.right_shift(min_cost, 24)
        cost_min_first_byte = max(-2 ** 7, min(cost_min_first_byte, 2 ** 7 - 1))
        # Extract last 24 bit
        cost_min_last_bytes = min_cost & 2 ** 24 - 1
        # last 24 bit of cost
        self.cost_min_last_bytes = Var(shape=shape, init=cost_min_last_bytes)
        # first 8 bit of cost
        self.cost_min_first_byte = Var(shape=shape, init=cost_min_first_byte)
