# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from lava.lib.optimization.solvers.graphs.bfs.process import BreadthFirstSearch
from lava.lib.optimization.solvers.graphs.bfs.process import BFSNeuron

from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import (
    PyLoihiProcessModel,
    PyAsyncProcessModel
)
from lava.magma.core.model.py.ports import PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.sparse.process import Sparse
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
        
        self.solution_readout = SolutionReadoutEthernet(
                            connection_config=connection_config,
                            num_bin_variables=num_nodes,
                            num_message_bits=32
                            )
        

        # number of distributor or aggregator neurons required for the solution
        num_nodes_per_aggregator = proc.proc_params.get("num_nodes_per_aggregator")
        num_nodes_per_distributor = proc.proc_params.get("num_nodes_per_distributor")
        num_aggregators = math.ceil(num_nodes/num_nodes_per_aggregator)
        num_distributors = math.ceil(num_nodes/num_nodes_per_distributor)
        

        def graph_nodes_to_dist_or_agg(num_nodes, num_aggs_or_dists, 
                                       num_nodes_per_agg_or_dist):
            '''helper function to generate connectivity matrices from graph 
            nodes to aggregator neurons or distributor neurons to graph nodes.
            Note that for distributor to neuron connections, the matrix 
            returned by this function has to be transposed for use in the outer
            routine. 
            '''
            neuron_to_agg_motif = np.ones((1, num_nodes_per_agg_or_dist))
            connectivity_motif = np.eye(num_aggs_or_dists)
            overcomplete_conn_mat = np.kron(connectivity_motif, 
                                            neuron_to_agg_motif)
            conn_mat = overcomplete_conn_mat[:, num_nodes]
            return conn_mat
        
        # TODO: Ask PS if the num_message_bits for connection processes here 
        # is 32 or 24
        # Build large block sparse connection process that corresponds
        # to the graph nodes being connected to aggregator neurons. 
        conn_mat = graph_nodes_to_dist_or_agg(num_nodes, num_aggregators, 
                                              num_nodes_per_aggregator)
        self.graph_neur_to_agg = Sparse(weights=conn_mat,
                             sign_mode=SignMode.EXCITATORY,
                             num_message_bits=32)
        
        self.aggregator_neuron = SpikeIntegrators(shape=(num_aggregators,))
  
        # Connection process that connects aggregator neurons 
        # to the GlobalDepthNeuron.
        conn_mat = np.ones((num_aggregators, 1))
        self.agg_to_glbl_dpth = Sparse(weights=conn_mat,
                                    sign_mode=SignMode.EXCITATORY,
                                    num_message_bits=32)
        
        # TODO: also use spike_integrator for this?
        self.global_depth_neuron = SpikeIntegrator((1,)) 

        # Use distributor neuron to distribute global depth values to 
        # motifs of neuron populations through an identity connection. 
        # Absolutely essential for large graph search problems
        
        conn_mat = np.ones((1, num_distributors))
        self.glbl_dpth_to_dist_neur = Sparse(weights=conn_mat,
                                    sign_mode=SignMode.EXCITATORY,
                                    num_message_bits=32)

        # TODO: two inports are required here?
        self.distributor_neuron = SpikeIntegrators2inPorts(shape=(num_distributors, ))
        
        # Build large block sparse connection process that corresponds
        # to the distributor neurons being connected to graph nodes.
        conn_mat = graph_nodes_to_dist_or_agg(num_nodes, num_distributors, 
                                              num_nodes_per_distributor)
        self.dist_to_graph_neur_conn= Sparse(weights=conn_mat,
                             sign_mode=SignMode.EXCITATORY,
                             num_message_bits=32)
   

        # Connect the parent InPort to the InPort of the child-Process.
        # TODO: Do we spike the start and target ID through a port on the the graph
        # neurons or do we do it through aggregator neurons?
        
        proc.in_ports.start_search.connect(self.distributor_neuron.a_in_2)
        self.distributor_neuron.s_out_2.connect(self.dist_to_graph_neur_conn.s_in)
        self.dist_to_graph_neur_conn.a_out.connect(self.graph_nodes.a_in_spk_i)
        
        # Connections are undirected
            # Forward connections
        self.graph_nodes.s_out_fwd.connect(self.graph_edges.s_in)
        self.graph_edges.a_out(self.graph_nodes.a_in_fwd)

            # Backward connections
        self.graph_nodes.s_out_bwd.connect(self.graph_edges.s_in)
        self.graph_edges.a_out(self.graph_nodes.a_in_bwd)

        # Scaling connections
        self.graph_nodes.s_out_agg.connect(self.graph_neur_to_agg.s_in)
        self.graph_neur_to_agg.a_out.connect(self.aggregator_neuron.a_in)
        self.aggregator_neuron.s_out.connect(self.agg_to_glbl_dpth.s_in)
        self.agg_to_glbl_dpth.connect(self.global_depth_neuron.a_in)
        self.global_depth_neuron.s_out.connect(self.glbl_dpth_to_dist_neur.s_in)
        self.glbl_dpth_to_dist_neur.a_out(self.distributor_neuron.a_in_1)
        self.distributor_neuron.s_out_1.connect(self.dist_to_graph_neur_conn.s_in)
        self.dist_to_graph_neur_conn.a_out.connect(self.graph_nodes.a_in_dist)


        # Readout connections
        # TODO: Can we connect a neuron directly to solution readout module 
        # without a connection process?
        self.graph_neuron.s_out_spk_o(self.solution_readout.state_in)
       
        # TODO: Create aliases for variables if necessary
        proc.vars.shortest_path_value.alias(self.global_depth_neuron.glbl_depth)
    