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
    variables_1bit: Var
        Binary variables assignment.
    variables_1bit: Var
        Values of 32 bit variables. Initiated by the parameter
        variables_32bit_init. The shape is determined by variables_32bit_num.

    InPorts:
    ----------
    variables_1bit_in: InPort
        Receives the best binary (1bit) states. Shape is determined by the
        number of binary variables.
    variables_32bit_in: List[InPort]
        Receives 32bit variables. The number of InPorts is defined by
        variables_32bit_num.

    OutPorts:
    ----------
    """

    def __init__(
            self,
            shape: ty.Tuple[int, ...],
            timeout: int,
            variables_1bit_num: int,
            variables_32bit_num: int,
            variables_32bit_init: ty.Union[int, ty.List[int]],
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
        timeout: int
            After timeout time steps, the run will be stopped.
        variables_1bit_num: int
            The number of 1bit (binary) variables.
        variables_32bit_num: int
            The number of 32bit variables and ports.
        variables_32bit_init: int, list[int]
            The initial values for the 32bit variables.
        num_message_bits: int
            Defines the number of bits of a single message via spikeIO.
            Currently only tested for 32bits.
        name: str, optional
            Name of the Process. Default is 'Process_ID', where ID is an integer
            value that is determined automatically.
        log_config: LogConfig, optional
            Configuration options for logging.z
        """

        self._validate_input(variables_32bit_num, variables_32bit_init)

        num_spike_integrators = variables_32bit_num + np.ceil(
            variables_1bit_num / num_message_bits).astype(int)

        super().__init__(
            shape=shape,
            timeout=timeout,
            num_spike_integrators=num_spike_integrators,
            num_message_bits=num_message_bits,
            connection_config=connection_config,
            name=name,
            log_config=log_config,
        )

        # Generate Var and InPort for 1bit variables
        self.variables_1bit = Var(shape=(variables_1bit_num,), init=0)
        self.variables_1bit_in = InPort(shape=(variables_1bit_num,))
        
        # Generate Vars and Inports for 32bit variables
        self.variables_32bit = Var(shape=(variables_32bit_num,),
                                   init=variables_32bit_init)
        self.variables_32bit_in = [InPort((1,))
                                   for _ in range(variables_32bit_num)]

    def _validate_input(self,
                        variables_32bit_num, 
                        variables_32bit_init) -> None:

        if isinstance(variables_32bit_init, int) and variables_32bit_num == 1:
            return
        elif isinstance(variables_32bit_init, list) and len(variables_32bit_init) == variables_32bit_num:
            return
        elif isinstance(variables_32bit_init, np.ndarray) and variables_32bit_init.shape[0] == variables_32bit_num:
            return
        else:
            raise ValueError(f"The variables_32bit_num must match the number "
                                f"of {variables_32bit_init=} provided.")


class SolutionReceiver(AbstractProcess):
    r"""Process which receives a solution via spikeIO on the superhost. Is
    connected within a SolutionReadout process.
    The way how information is processed is defined by the run_async of the
    PyProcModel, which must be defined for each SNN separately.

    Attributes
    ----------
    variables_1bit: Var
        Binary variables assignment.
    <name defined by variables_32bit_names>: Var
        Values of 32 bit variables. Initiated by the parameter
        variables_32bit_init. There will be one 32bit variable for each list
        entry of variables_32bit_names.

    InPorts:
    ----------
    results_in: InPort
        Receives all input from the SpikeIntegrators of a SolutionReadout
        process.

    OutPorts:
    -------
    """

    def __init__(
            self,
            shape: ty.Tuple[int, ...],
            timeout: int,
            variables_1bit_num: int,
            variables_1bit_init: ty.Union[npty.ArrayLike, int],
            variables_32bit_num: int,
            variables_32bit_init: ty.Union[npty.ArrayLike, int],
            num_spike_integrators: int,
            num_message_bits: int,
            name: ty.Optional[str] = None,
            log_config: ty.Optional[LogConfig] = None,
    ) -> None:
        """
        Parameters
        ----------
        shape: tuple
            A tuple of the form (number of variables, domain size).
        timeout: int
            After timeout time steps, the run will be stopped.
        variables_1bit_num: int
            The number of 1bit (binary) variables.
        variables_1bit_init: int
            The initial values of 1bit (binary) variables.
        variables_32bit_num: int
            The number of 32bit variables and ports.
        variables_32bit_init: int, list[int]
            The initial values for the 32bit variables.
        num_message_bits: int
            Defines the number of bits of a single message via spikeIO.
            Currently only tested for 32bits.
        name: str, optional
            Name of the Process. Default is 'Process_ID', where ID is an integer
            value that is determined automatically.
        log_config: LogConfig, optional
            Configuration options for logging.z
        """
        
        super().__init__(
            shape=shape,
            name=name,
            log_config=log_config,
        )

        self.num_message_bits = Var(shape=(1,), init=num_message_bits)
        self.timeout = Var(shape=(1,), init=timeout)

        # Define Vars
        self.variables_1bit = Var(shape=(variables_1bit_num,),
                                  init=variables_1bit_init)
        self.variables_32bit = Var(shape=(variables_32bit_num,),
                                   init=variables_32bit_init)
        
        # Define InPorts
        self.results_in = InPort(shape=(num_spike_integrators,))
