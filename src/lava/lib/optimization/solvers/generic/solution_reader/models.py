# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
from lava.lib.optimization.solvers.generic.read_gate.process import ReadGate
from lava.lib.optimization.solvers.generic.solution_reader.process import (
    SolutionReader,
)
from lava.lib.optimization.solvers.generic.monitoring_processes\
    .solution_readout.process import SolutionReadout
from lava.magma.core.decorator import implements
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol


@implements(proc=SolutionReader, protocol=LoihiProtocol)
class SolutionReaderModel(AbstractSubProcessModel):
    def __init__(self, proc):
        var_shape = proc.proc_params.get("var_shape")
        target_cost = proc.proc_params.get("target_cost")
        num_in_ports = proc.proc_params.get("num_in_ports")
        num_steps = proc.proc_params.get("num_steps")

        self.read_gate = ReadGate(
            shape=var_shape, target_cost=target_cost, num_in_ports=num_in_ports,
            num_steps=num_steps
        )
        self.solution_readout = SolutionReadout(
            shape=var_shape, target_cost=target_cost
        )
        self.read_gate.cost_out.connect(self.solution_readout.cost_in)
        self.read_gate.solution_out.connect(self.solution_readout.read_solution)
        self.read_gate.send_pause_request.connect(
            self.solution_readout.timestep_in
        )
        self.solution_readout.acknowledgement.connect(
            self.read_gate.acknowledgement
        )

        proc.vars.solution.alias(self.solution_readout.solution)
        proc.vars.min_cost.alias(self.solution_readout.min_cost)
        proc.vars.solution_step.alias(self.solution_readout.solution_step)

        self.read_gate.solution_reader.connect(proc.ref_ports.ref_port)
        for idx in range(num_in_ports):
            in_port = getattr(proc.in_ports,
                              f"read_gate_in_port_first_byte_{idx}")
            out_port = getattr(self.read_gate, f"cost_in_first_byte_{idx}")
            in_port.connect(out_port)
            in_port = getattr(proc.in_ports,
                              f"read_gate_in_port_last_bytes_{idx}")
            out_port = getattr(self.read_gate, f"cost_in_last_bytes_{idx}")
            in_port.connect(out_port)
