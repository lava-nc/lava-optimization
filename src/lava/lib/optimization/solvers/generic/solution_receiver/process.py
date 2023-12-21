# Copyright (C) 2023 Intel Corporation
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


class SolutionReadout(AbstractProcess):
    r"""Process which implements the solution readout layer
    on the solver of an optimization problem.

    Attributes
    ----------
    a_in: InPort
        The addition of all inputs (per dynamical system) at this timestep
        will be received by this port.
    s_out: OutPort
        The payload to be exchanged between the underlying dynamical systems
        when these fire.
    local_cost: OutPort
        The cost components per dynamical system underlying these
        variables, i.e., c_i = sum_j{Q_{ij} \cdot x_i}  will be sent through
        this port. The cost integrator will then complete the cost computation
        by adding all contributions, i.e., x^T \cdot Q \cdot x = sum_i{c_i}.
    variable_assignment: Var
        Holds the current value assigned to the variables by
        the solver network.
    """

    def __init__(
        self,
        shape: ty.Tuple[int, ...],
        connection_config: ConnectionConfig,
        num_bin_variables: int,
        num_message_bits = 24,
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
    ) -> None:
        """
        Parameters
        ----------
        shape: tuple
            A tuple of the form (number of variables, domain size).
        cost_diagonal: npty.ArrayLike
            The diagonal of the coefficient of the quadratic term on the cost
            function.
        cost_off_diagonal: npty.ArrayLike
            The off-diagonal of the coefficient of the quadratic term on the
            cost function.
        hyperparameters: dict, optional
        name: str, optional
            Name of the Process. Default is 'Process_ID', where ID is an integer
            value that is determined automatically.
        log_config: LogConfig, optional
            Configuration options for logging.z"""

        num_spike_integrators = 2 + np.ceil(num_bin_variables / num_message_bits).astype(int)

        super().__init__(
            shape=shape,
            num_spike_integrators=num_spike_integrators,
            num_bin_variables=num_bin_variables,
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
        num_variables: int,
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
            num_variables=num_variables,
            name=name,
            log_config=log_config,
        )

        self.best_state = Var(shape=(num_variables,), init=best_state_init)
        self.best_timestep = Var(shape=(1,), init=best_timestep_init)
        self.best_cost = Var(shape=(1,), init=best_cost_init)
        self.num_message_bits = Var(shape=(1,), init=num_message_bits)
        self.results_in = InPort(shape=(num_spike_integrators,))
