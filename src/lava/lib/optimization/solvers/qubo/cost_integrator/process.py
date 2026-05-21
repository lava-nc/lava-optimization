# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
import numpy as np

from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var


class CostIntegrator(AbstractProcess):
    """Node that monitors execution of the QUBOSolver. It integrates the cost
    components from all variables. Whenever a new better solution is found,
    it stores the new best cost and the associated timestep, while triggering
    the variable neurons to store the new best state. Waits for stopping
    criteria to be reached, either cost_target or timeout. Once reached,
    it spikes out the best cost, timestep, and a trigger for the variable
    neurons to spike out the best state.

    Parameters
    ----------
    shape : tuple(int)
        The expected number and topology of the input cost components.
    cost_target: int
        Target cost of the QUBO solver. Once reached, the best_cost,
        best_timestep, and best_state are spiked out.
    cost_init: int
        The initial value of the cost. Only required if mode=='iterative'.
    is_iterative: str
        Determines if the CostIntegrator receives the cost in a
        sparsified form. If set to False, the cost_in port provides the
        total cost of the current state. If set to True,
        the CostIntegrator receives only the changes since the last state.
    timeout: int
        Timeout of the QUBO solver. Once reached, the best_cost,
        best_timestep, and best_state are spiked out.
    name : str, optional
        Name of the Process. Default is 'Process_ID', where ID is an
        integer value that is determined automatically.
        log_config: Configuration options for logging.

    InPorts
    -------
    cost_in
        input from the variable neurons. Added, this input denotes
        the total cost of the current variable assignment.

    OutPorts
    --------
    control_states_out
        Port to the variable neurons.
        Can send either of the following three values:
            1 -> store the state, since it is the new best state
            2 -> store the state and spike it, since stopping criteria reached
            3 -> spike the best state
    best_cost_out
        Port to the SolutionReadout. Sends the best cost found.
    best_timestep_out
        Port to the SolutionReadout. Sends the timestep when the best cost
        was found.

    Vars
    ----
    timestep
        Holds current timestep
    cost_min_last_bytes
        Current minimum cost, i.e., the lowest reported cost so far.
        Saves the last 3 bytes.
        cost_min = cost_min_first_byte << 24 + cost_min_last_bytes
    cost_min_first_byte
        Current minimum cost, i.e., the lowest reported cost so far.
        Saves the first byte.
    cost_last_bytes
        Current cost.
        Saves the last 3 bytes.
        cost_min = cost_min_first_byte << 24 + cost_min_last_bytes
    cost_first_byte
        Current cost.
        Saves the first byte.
    """

    def __init__(
            self,
            *,
            cost_target: int,
            timeout: int,
            is_iterative: bool = False,
            cost_init: int = None,
            shape: ty.Tuple[int, ...] = (1,),
            name: ty.Optional[str] = None,
            log_config: ty.Optional[LogConfig] = None,
    ) -> None:
        self._input_validation(cost_target=cost_target,
                               timeout=timeout,
                               is_iterative=is_iterative,
                               cost_init=cost_init)
        cost_init_first_byte, cost_init_last_bytes, cost_target_fist_byte, \
            cost_target_last_bytes = self._transform_inputs(
            cost_init, cost_target)

        super().__init__(shape=shape,
                         cost_target_first_byte=cost_target_fist_byte,
                         cost_target_last_bytes=cost_target_last_bytes,
                         is_iterative=is_iterative,
                         timeout=timeout,
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
        self.cost_last_bytes = Var(shape=shape, init=cost_init_last_bytes)
        # first 8 bit of cost
        self.cost_first_byte = Var(shape=shape, init=cost_init_first_byte)

        # Var to store best cost found to far
        # Note: Total min cost = cost_min_first_byte << 24 + cost_min_last_bytes
        # last 24 bit of cost
        self.cost_min_last_bytes = Var(shape=shape, init=0)
        # first 8 bit of cost
        self.cost_min_first_byte = Var(shape=shape, init=0)

    @staticmethod
    def _input_validation(cost_target,
                          timeout,
                          is_iterative,
                          cost_init) -> None:
        if (cost_target is None and timeout is None):
            raise ValueError(
                "Both the cost_target and the timeout must be defined")
        if cost_target > 0 or cost_target < - 2 ** 31 + 1:
            raise ValueError(
                f"The target cost must in the range [-2**32 + 1, 0], "
                f"but is {cost_target}.")
        if timeout <= 0 or timeout > 2 ** 24 - 1:
            raise ValueError(
                f"The timeout must be in the range (0, 2**24 - 1], but is "
                f"{timeout}.")
        if is_iterative and cost_init is None:
            raise ValueError(
                f"For iterative cost integration, an initial cost must be "
                f"provided")

    def _transform_inputs(self, cost_init, cost_target
                          ) -> [int, int, int, int]:
        if cost_init is None:
            cost_init = 0

        return *self.split_cost(cost_init), *self.split_cost(cost_target)

    @staticmethod
    def split_cost(cost_32_bit) -> [int, int]:
        """Splits a 32bit cost into a 8bit and a 24bit component. Allows to
        be stored and processed in ucode. Split happens according to
            cost = (cost_first_byte << 24) + cost_last_bytes"""

        cost = np.int32(cost_32_bit)
        cost_first_byte = cost >> 24
        cost_last_bytes = cost - (cost_first_byte << 24)

        # Ensure that transformation is correct.
        cost_init_check = (cost_first_byte << 24) + cost_last_bytes
        if cost_32_bit != cost_init_check:
            raise ValueError(f"Transformation in CostIntegrator was incorrect. "
                             f"Most likely, the cost {cost_32_bit} is not a "
                             f"32bit integer.")

        return cost_first_byte, cost_last_bytes
