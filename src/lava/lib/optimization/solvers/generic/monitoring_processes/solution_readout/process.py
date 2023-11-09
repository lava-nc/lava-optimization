# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import numpy as np
import typing as ty

from lava.magma.core.process.ports.ports import InPort
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var


class SolutionReadout(AbstractProcess):
    """Process to readout solution from SNN and make it available on host.

    Parameters
    ----------
    shape: The shape of the set of nodes, or process, which state will be read.
    target_cost: cost value at which, once attained by the network,
    this process will stop execution.
    name: Name of the Process. Default is 'Process_ID', where ID is an
    integer value that is determined automatically.
    log_config: Configuration options for logging.
    time_steps_per_algorithmic_step: the number of iteration steps that a
    single algorithmic step requires. This value is required to decode the
    variable values from the spk_hist of a process.

    Attributes
    ----------
    read_solution: InPort
        A message received on this ports signifies the process
        should call read on its RefPort.
    ref_port: RefPort
        A reference port to a variable in another process which state
        will be remotely accessed upon read request. Here, it reads the
        current variables assignment by a solver to an optimization problem.
    target_cost: Var
        Cost value at which, once attained by the network.

    """

    def __init__(
        self,
        shape: ty.Tuple[int, ...],
        target_cost=None,
        time_steps_per_algorithmic_step: int = 1,
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
    ) -> None:
        super().__init__(
            shape=shape,
            target_cost=target_cost,
            name=name,
            log_config=log_config,
        )
        self.solution = Var(shape=shape, init=-1)
        self.solution_step = Var(shape=(1,), init=-1)
        self.min_cost = Var(shape=(2,), init=-1)
        self.target_cost = Var(shape=(1,), init=target_cost)
        self.time_steps_per_algorithmic_step = Var(
            shape=(1,),
            init=time_steps_per_algorithmic_step)
        self.read_solution = InPort(shape=shape)
        self.cost_in = InPort(shape=(2,))
        self.timestep_in = InPort(shape=(1,))
