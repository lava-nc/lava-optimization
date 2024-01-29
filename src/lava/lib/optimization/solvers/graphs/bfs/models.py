# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

# TODO: This is a probably an ncprocess model file. Should this really be here?
# How are heirarchical processes implemented in case of only ncmodels? Is the
# hierarchical abstraction outlined in process.py or is it directly outlined 
# in ncmodels.py? 

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

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.sparse.process import Sparse, DelaySparse
from lava.utils.weightutils import SignMode
from lava.proc import embedded_io as eio
from scipy.sparse import csr_matrix
import math 

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
        
        # Distributor neurons to fan-in the global depth 
        # value to depth neuron from all neurons
        # What is the best way to calculate the shape of DistNeurons?
        # Fan-in cannot be unlimited. For a small system, say a robotics 
        # use-case, the fan-in can be perhaps supported. But Hala-point level 
        # systems
        # number of distributor or aggregator neurons required for the solution
        dist_agg_neuron_shape = math.ceil(num_nodes/32)
        

        # Use distributor neuron to distribute global depth values to 
        # motifs of neuron populations through an identity connection. 
        # Absolutely essential for large graph search problems
        
        # TODO: Build large identity sparse connection process that corresponds
        # to connecting the global depth value to all possible distributor 
        # neurons. The worst case value (for Hala point) are ....
        self.dist_to_graph_neur = Sparse()

        self.distributor_neuron = DistNeuron(shape=(dist_agg_neuron_shape, ))
    

        # TODO: Build large identity sparse connection process that corresponds
        # to the whole system. 
        self.graph_neur_to_agg = Sparse()
        
        self.aggregator_neuron = AggNeuron(shape=(dist_agg_neuron_shape, ))

        # TODO: Synthesize connection process that connects aggregator neurons 
        # to the DepthNeuron. The worst case value (for Hala point) are .....
        self.agg_to_glbl_dpth = Sparse()
        
        self.global_depth_indicator = GlobalDepthNeuron((1,)) 
   

        # Connect the parent InPort to the InPort of the child-Process.
        # Do we spike the start and target ID through a port on the the graph
        # neurons or do we do it through aggregator neurons?
        proc.in_ports.start_search.connect(self.graph_nodes.s_in_spk_i)

        # TODO: Lay out the connectivity of the intermediate ports of all the 
        # different neurons in the system

        # TODO: Check with PS what the name of the outport of the solution 
        # readout is (probably not a_out)
        self.solution_readout.out_ports.a_out. \
        connect(proc.out_ports.shortest_path_nodes)

        # TODO: Create aliases for variables if necessary
        proc.vars.best_state.alias(self.solution_receiver.best_state)
    