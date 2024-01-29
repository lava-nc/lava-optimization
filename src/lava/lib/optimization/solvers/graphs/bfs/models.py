# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import numpy as np
from lava.lib.optimization.solvers.graphs.bfs import BreadthFirstSearch
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import (
    PyLoihiProcessModel,
    PyAsyncProcessModel
)
from bitstring import Bits
from lava.magma.core.model.py.ports import PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol

from lava.lib.optimization.solvers.generic.solution_receiver.process import \
    (
    SolutionReceiver, SpikeIntegrator
)
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.sparse.process import Sparse, DelaySparse
from lava.utils.weightutils import SignMode
from lava.proc import embedded_io as eio
from scipy.sparse import csr_matrix

@implements(proc=BreadthFirstSearch, protocol=LoihiProtocol)
@requires(CPU)
class BreadthFirstSearchModel(AbstractSubProcessModel):
    """Model for the SolutionReadout process.

    """

    def __init__(self, proc):
        num_nodes = self.adjacency_matrix.shape[0]
        connection_config = proc.proc_params.get("connection_config")

        self.graph_nodes = BFSNeuron(shape=(num_nodes, ))
        # if the connectivity matrix contains only 2 values, then the weights
        # are automatically scaled to 1-bit while using num_weight_bits=8.
        # When num_message_bits=0, the spikes are binary.
        self.graph_edges = Sparse(weights=self.adjacency_matrix,
                             sign_mode=SignMode.EXCITATORY,
                             num_message_bits=0
                             )
        self.solution_readout = SolutionReadout(
                            connection_config=connection_config,
                            num_bin_variables=num_nodes,
                            num_message_bits=32
                            )
        
        # Implementation of distributor neurons to fan-in the global depth 
        # value to all neurons

        self. 

   

        # Connect the parent InPort to the InPort of the child-Process.
        proc.in_ports.states_in.connect(self.synapses_state_in.s_in)
        proc.in_ports.cost_in.connect(self.synapses_cost_in.s_in)
        proc.in_ports.timestep_in.connect(self.synapses_timestep_in.s_in)

        # Connect intermediate ports
        self.synapses_state_in.a_out.connect(self.spike_integrators.a_in)
        self.synapses_cost_in.a_out.connect(self.spike_integrators.a_in)
        self.synapses_timestep_in.a_out.connect(self.spike_integrators.a_in)

        self.spike_integrators.s_out.connect(
            self.solution_receiver.results_in, connection_config)

        # Create aliases for variables
        proc.vars.best_state.alias(self.solution_receiver.best_state)
        proc.vars.best_timestep.alias(self.solution_receiver.best_timestep)
        proc.vars.best_cost.alias(self.solution_receiver.best_cost)