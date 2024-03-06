# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from lava.lib.optimization.solvers.qubo.solution_readout.process import (
    SolutionReadoutEthernet,
    SpikeIntegrator,
)
from lava.proc.dense.process import Dense
import numpy as np
from lava.lib.optimization.solvers.graphs.bfs.process import BreadthFirstSearch
from lava.lib.optimization.solvers.graphs.bfs.process import BFSNeuron

from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.resources import CPU

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.sparse.process import Sparse
from lava.utils.weightutils import SignMode
from lava.proc import embedded_io as eio
from scipy.sparse import csr_matrix
import math

from tests.lava.lib.optimization.solvers.qubo.solution_readout.test_models import (
    Spiker32bit,
)


@implements(proc=BreadthFirstSearch, protocol=LoihiProtocol)
@requires(CPU)
class BreadthFirstSearchModel(AbstractSubProcessModel):
    """Model for the SolutionReadout process."""

    def __init__(self, proc):
        connection_config = proc.proc_params.get("connection_config")
        self.adjacency_matrix = proc.proc_params.get("adjacency_matrix")
        self.neuron_status = proc.proc_params.get("node_description")
        self.num_nodes = self.adjacency_matrix.shape[0]
        self.graph_nodes = BFSNeuron(
            shape=(self.num_nodes,), status=self.neuron_status
        )
        # if the connectivity matrix contains only 2 values, then the weights
        # are automatically scaled to 1-bit while using num_weight_bits=8.
        # When num_message_bits=0, the spikes are binary.
        self.graph_edges_fwd = Sparse(
            weights=self.adjacency_matrix,
            sign_mode=SignMode.EXCITATORY,
            num_message_bits=24,
        )
        self.graph_edges_bwd = Sparse(
            weights=self.adjacency_matrix,
            sign_mode=SignMode.EXCITATORY,
            num_message_bits=24,
        )
        self.solution_readout = SolutionReadoutEthernet(
            shape=(1,),
            connection_config=connection_config,
            variables_1bit_num=self.num_nodes,
            variables_32bit_num=1,
            variables_1bit_init=np.zeros((self.num_nodes,)),
            variables_32bit_init=0,
            num_message_bits=32,
            timeout=100000,
        )

        # number of distributor or aggregator neurons required for the solution
        num_nodes_per_aggregator = proc.proc_params.get(
            "num_nodes_per_aggregator"
        )
        num_nodes_per_distributor = proc.proc_params.get(
            "num_nodes_per_distributor"
        )

        num_aggregators = math.ceil(self.num_nodes / num_nodes_per_aggregator)
        num_distributors = math.ceil(self.num_nodes / num_nodes_per_distributor)

        # Reconfigure num_nodes_per_aggregator/distributor to num_nodes if the
        # num_nodes do not exceed the capacity of the aggregator/distributor
        num_nodes_per_aggregator = (
            self.num_nodes
            if self.num_nodes < num_nodes_per_aggregator
            else num_nodes_per_aggregator
        )

        num_nodes_per_distributor = (
            self.num_nodes
            if self.num_nodes < num_nodes_per_distributor
            else num_nodes_per_distributor
        )

        def graph_nodes_to_dist_or_agg(
            num_nodes, num_aggs_or_dists, num_nodes_per_agg_or_dist
        ):
            """helper function to generate connectivity matrices from graph
            nodes to aggregator neurons or distributor neurons to graph nodes.
            Note that for distributor to neuron connections, the matrix
            returned by this function has to be transposed for use in the outer
            routine.
            """
            neuron_to_agg_motif = np.ones((1, num_nodes_per_agg_or_dist))
            connectivity_motif = np.eye(num_aggs_or_dists)
            overcomplete_conn_mat = np.kron(
                connectivity_motif, neuron_to_agg_motif
            )
            conn_mat = overcomplete_conn_mat[:, :num_nodes]
            return conn_mat

        # Build large block sparse connection process that corresponds
        # to the graph nodes being connected to aggregator neurons.
        conn_mat = graph_nodes_to_dist_or_agg(
            self.num_nodes, num_aggregators, num_nodes_per_aggregator
        )
        self.graph_neur_to_agg_conn = Sparse(
            weights=conn_mat, sign_mode=SignMode.EXCITATORY, num_message_bits=24
        )

        self.aggregator_neurons = SpikeIntegrator(shape=(num_aggregators,))

        # Connection process that connects aggregator neurons
        # to the GlobalDepthNeuron.
        conn_mat = np.ones((1, num_aggregators))
        self.agg_to_glbl_dpth_conn = Sparse(
            weights=conn_mat, sign_mode=SignMode.EXCITATORY, num_message_bits=24
        )

        self.global_depth_neuron = SpikeIntegrator((1,))

        # Use distributor neuron to distribute global depth values to
        # motifs of neuron populations through an identity connection.
        # Absolutely essential for large graph search problems

        conn_mat = np.ones((num_distributors, 1))
        self.glbl_dpth_to_dist_conn = Sparse(
            weights=conn_mat, sign_mode=SignMode.EXCITATORY, num_message_bits=24
        )

        # TODO: two inports are required here?
        # PS: dont send this via distributor neuron. Spike Input can be probably
        # be used to address a specific neuron efficiently
        # PS: We try to test things without spike input first. Manually set the
        # target and destination by using Vars.
        self.distributor_neurons = SpikeIntegrator(shape=(num_distributors,))

        # Build large block sparse connection process that corresponds
        # to the distributor neurons being connected to graph nodes.
        conn_mat = graph_nodes_to_dist_or_agg(
            self.num_nodes, num_distributors, num_nodes_per_distributor
        )
        self.dist_to_graph_neur_conn = Sparse(
            weights=conn_mat.T,
            sign_mode=SignMode.EXCITATORY,
            num_message_bits=32,
        )

        # Connect the parent InPort to the InPort of the child-Process.
        # TODO: Do we spike the start and target ID through a port on the the graph
        # neurons or do we do it through distributor neurons?
        # Commenting out for now. To be used with spk input to make this work
        # proc.in_ports.start_search.connect(self.distributor_neuron.a_in_2)
        # self.distributor_neuron.s_out_2.connect(
        #     self.dist_to_graph_neur_conn.s_in
        # )

        payload_fwd = np.zeros((self.num_nodes,), dtype=np.int32)
        payload_fwd[0] = 1

        self.input_spike_fwd = Spiker32bit(
            period=1, payload=payload_fwd, shape=(self.num_nodes,)
        )

        self.input_spike_i_o = Spiker32bit(
            period=1,
            payload=np.zeros((self.num_nodes,), dtype=np.int32),
            shape=(self.num_nodes,),
        )

        self.i_o_dense_conn_spiker = Sparse(
            weights=np.eye(
                self.num_nodes,
            ),
            num_message_bits=24,
        )

        self.fwd_dense_conn_spiker = Sparse(
            weights=np.eye(
                self.num_nodes,
            ),
            num_message_bits=24,
        )

        self.dist_to_graph_neur_conn.a_out.connect(self.graph_nodes.a_in_4)

        # Connections are undirected
        # Forward connections
        self.input_spike_fwd.s_out.connect(self.fwd_dense_conn_spiker.s_in)
        self.fwd_dense_conn_spiker.a_out.connect(self.graph_nodes.a_in_4)

        self.graph_nodes.s_out_4.connect(self.graph_edges_fwd.s_in)
        self.graph_edges_fwd.a_out.connect(self.graph_nodes.a_in_4)

        # Backward connections
        self.graph_nodes.s_out_3.connect(self.graph_edges_bwd.s_in)
        self.graph_edges_bwd.a_out.connect(self.graph_nodes.a_in_3)

        # Scaling connections
        self.graph_nodes.s_out_2.connect(self.graph_neur_to_agg_conn.s_in)
        self.graph_neur_to_agg_conn.a_out.connect(self.aggregator_neurons.a_in)
        self.aggregator_neurons.s_out.connect(self.agg_to_glbl_dpth_conn.s_in)
        self.agg_to_glbl_dpth_conn.a_out.connect(self.global_depth_neuron.a_in)
        self.global_depth_neuron.s_out.connect(self.glbl_dpth_to_dist_conn.s_in)
        self.glbl_dpth_to_dist_conn.a_out.connect(self.distributor_neurons.a_in)
        self.distributor_neurons.s_out.connect(
            self.dist_to_graph_neur_conn.s_in
        )
        self.dist_to_graph_neur_conn.a_out.connect(self.graph_nodes.a_in_2)

        # Placeholder spike_i connnections
        self.input_spike_i_o.s_out.connect(self.i_o_dense_conn_spiker.s_in)
        self.i_o_dense_conn_spiker.a_out.connect(self.graph_nodes.a_in_1)

        # Readout connections
        # SolutionReadoutEthernet starts with a connection layer. Therefore
        # neuron can be connected
        self.graph_nodes.s_out_1.connect(
            self.solution_readout.variables_1bit_in
        )
        self.global_depth_neuron.s_out.connect(
            self.solution_readout.variables_32bit_0_in
        )
