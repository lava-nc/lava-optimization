# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import numpy as np
import typing as ty
import numpy.typing as npty

from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var

from lava.magma.core.process.ports.connection_config import ConnectionConfig


class SpikeIntegrator(AbstractProcess):
    """GradedVec
    Graded spike vector layer. Accumulates and forwards 32bit spikes.

    Parameters
    ----------
    shape: tuple(int)
        number and topology of neurons
    """

    def __init__(
            self,
            shape: ty.Tuple[int, ...]) -> None:
        super().__init__(shape=shape)

        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)


class SolutionReadoutEthernet(AbstractProcess):
    r"""Process which implements the solution readout layer
    on the solver of an optimization problem.

    Attributes
    ----------
    best_state: Var
        Best binary variables assignment.
    best_cost: Var
        Cost of best solution.
    best_timestep: Var
        Time step when best solution was found.

    InPorts:
    ----------
    states_in: InPort
        Receives the best binary (1bit) states. Shape is determined by the
        number of
        binary variables.
    cost_in: InPort
        Receives the best 32bit cost.
    timestep_in: InPort
        Receives the best 32bit timestep.

    OutPorts:
    ----------
    """

    def __init__(
            self,
            shape: ty.Tuple[int, ...],
            timeout: int,
            num_bin_variables: int,
            connection_config: ConnectionConfig,
            num_message_bits=32,
            name: ty.Optional[str] = None,
            log_config: ty.Optional[LogConfig] = None,
    ) -> None:
        """
        Parameters
        ----------
        shape: tuple
            A tuple of the form (number of variables, domain size).
        num_bin_variables: int
            The number of binary (1bit) variables.
        num_message_bits: int
            Defines the number of bits of a single message via spikeIO.
            Currently only tested for 32bits.
        name: str, optional
            Name of the Process. Default is 'Process_ID', where ID is an integer
            value that is determined automatically.
        log_config: LogConfig, optional
            Configuration options for logging.z"""

        num_spike_integrators = 2 + np.ceil(
            num_bin_variables / num_message_bits).astype(int)

        super().__init__(
            shape=shape,
            num_spike_integrators=num_spike_integrators,
            num_bin_variables=num_bin_variables,
            timeout=timeout,
            num_message_bits=num_message_bits,
            connection_config=connection_config,
            name=name,
            log_config=log_config,
        )

        self.states_in = InPort(shape=(num_bin_variables,))
        self.cost_in = InPort((1,))
        self.timestep_in = InPort((1,))
        self.best_state = Var(shape=(num_bin_variables,), init=0)
        self.best_timestep = Var(shape=(1,), init=1)
        self.best_cost = Var(shape=(1,), init=0)


class SolutionReceiver(AbstractProcess):
    """Process to readout solution from SNN and make it available on host.

    Parameters
    ----------
    shape: The shape of the set of nodes, or process, which state will be read.
    target_cost: int
        cost value at which, once attained by the network, this process will
        stop execution.
    name: str
        Name of the Process. Default is 'Process_ID', where ID is an integer
        value that is determined automatically.
    log_config:
        Configuration options for logging.

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
            num_variables: int,
            timeout: int,
            best_cost_init: int,
            best_state_init: ty.Union[npty.ArrayLike, int],
            num_spike_integrators: int,
            best_timestep_init: int,
            num_message_bits: int = 24,
            name: ty.Optional[str] = None,
            log_config: ty.Optional[LogConfig] = None,
    ) -> None:
        super().__init__(
            shape=shape,
            name=name,
            log_config=log_config,
        )

        self.best_state = Var(shape=(num_variables,), init=best_state_init)
        self.best_timestep = Var(shape=(1,), init=best_timestep_init)
        self.best_cost = Var(shape=(1,), init=best_cost_init)
        self.num_message_bits = Var(shape=(1,), init=num_message_bits)
        self.timeout = Var(shape=(1,), init=timeout)
        self.results_in = InPort(shape=(num_spike_integrators,))
