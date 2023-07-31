# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty

from lava.magma.core.process.ports.ports import InPort, OutPort, RefPort
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var


class ReadGate(AbstractProcess):
    """Process that triggers solution readout when problem is solved.

    Parameters
    ----------
    shape: The shape of the set of units in the downstream process whose state
        will be read by ReadGate.
    target_cost: cost value at which, once attained by the network,
        this process will stop execution.
    name: Name of the Process. Default is 'Process_ID', where ID is an
        integer value that is determined automatically.
    log_config: Configuration options for logging.

    InPorts
    -------
    cost_in_last_bytes: OutPort
        Receives a better cost found by the CostIntegrator at the
        previous timestep.
        Messages the last 3 byte of the new best cost.
        Total cost = cost_in_first_byte << 24 + cost_in_last_bytes.
    cost_in_first_byte: OutPort
        Receives a better cost found by the CostIntegrator at the
        previous timestep.
        Messages the first byte of the new best cost.

    OutPorts
    --------
    cost_out: Forwards to an upstream process the better cost notified by the
        CostIntegrator.
    solution_out: Forwards to an upstream process the better variable assignment
        found by the solver network.
    send_pause_request: Notifies upstream process to request execution to pause.
    """

    def __init__(
            self,
            shape: ty.Tuple[int, ...],
            target_cost=None,
            num_in_ports=1,
            name: ty.Optional[str] = None,
            log_config: ty.Optional[LogConfig] = None,
    ) -> None:
        super().__init__(
            shape=shape,
            target_cost=target_cost,
            num_in_ports=num_in_ports,
            name=name,
            log_config=log_config,
        )
        self.target_cost = Var(shape=(1,), init=target_cost)

        self.best_solution = Var(shape=shape, init=-1)
        for id in range(num_in_ports):
            # Cost is transferred as two separate values
            # cost_last_bytes = last 3 byte of cost
            # cost_first_byte = first byte of cost
            # total cost = np.int8(cost_first_byte) << 24 + cost_last_bytes
            setattr(self, f"cost_in_last_bytes_{id}", InPort(shape=(1,)))
            setattr(self, f"cost_in_first_byte_{id}", InPort(shape=(1,)))
        self.cost_out = OutPort(shape=(2,))
        self.best_solution = Var(shape=shape, init=-1)
        self.send_pause_request = OutPort(shape=(1,))
        self.solution_out = OutPort(shape=shape)
        self.solution_reader = RefPort(shape=shape)
