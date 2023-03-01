# INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY
#
# Copyright © 2023 Intel Corporation.
#
# This software and the related documents are Intel copyrighted
# materials, and your use of them is governed by the express
# license under which they were provided to you (License). Unless
# the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit  this software or the
# related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with
# no express or implied warranties, other than those that are
# expressly stated in the License.
import numpy as np
from scipy.spatial import distance
# import typing as ty


class QMatrixVRP:
    """Class to generate Q matrix for VRP framed as QUBO problems. Currently 
    supports generation of Q matrices for TSP and clustering problems. The 
    matrix values are computed based on the Euclidean distance between the nodes 
    assuming all-to-all connectivity. 
    """

    def __init__(
        self,
        input_nodes,
        num_vehicles=1,
        problem_type="None",
        lamda_dist=1,
        lamda_cnstrnt=1,
        fixed_pt=False,
        fixed_pt_range=(0, 127),
    ) -> None:
        """constructor of the class generates Q matrices depending on the type 
        of problem specified by the user and assings it the class variables for 
        the matrix. Calls private functions to initialize Q. Nonetype will raise 
        an exception asking for the correct problem type to be specified. The 
        matrix Q is considered to have all-to-all connectivity between the nodes 
        that are specified. 

        Args:
            input_nodes (list<tuples>): Input to matrix generator functions 
            containing a list of nodes specifed as tuples. 

            num_vehicles (int): Number of vehicles in the Vehicle Routing 
            Problem. The first `num_vehicles` nodes in the `input_nodes` 
            variable correspond to positions of vehicles and the rest waypoints.
            Defaults to 1 

            problem_type (str, optional): Specifies for the type of problem for 
            which Q matrix has to be generated. Currently support for 
            1. `tsp` : Travelling salesman problem
            2. `clustering` : Clustering problem framed as a QUBO.
            Defaults to 'None'.
            
            lamda_dist (float, optional):

            lamda_cnstrt (float, optional):

            fixed_pt (bool, optional): Specifies if the Q matrix should
            ultimately be rounded down to integer. If `True`, stochastic rounding 
            to integer range of Loihi 2 is performed. Defaults to `False`.

            fixed_pt_range (tuple<int>, optional): Specifies the absolute value 
            of  min and max values that the Q matrix can have if 
            `fixed_pt =True`.
        """
        self.fixed_pt = fixed_pt
        self.min_fixed_pt_mant = fixed_pt_range[0]
        self.max_fixed_pt_mant = fixed_pt_range[1]
        self.problem_type = problem_type
        self.num_vehicles = num_vehicles
        if self.problem_type == "tsp":
            self.matrix = self._gen_tsp_Q_matrix(
                input_nodes, lamda_dist, lamda_cnstrnt
            )
        elif self.problem_type == "clustering":
            self.matrix = self._gen_clustering_Q_matrix(
                input_nodes, lamda_dist, lamda_cnstrnt
            )
        else:
            raise ValueError(
                "problem_type cannot be None or argument passed cannot be serviced"
            )

    def _gen_tsp_Q_matrix(self, input_nodes, lamda_dist, lamda_cnstrnt):
        """Return the Q matrix that sets up the QUBO for the 
        clustering problem. The cluster centers are assumed to be uniformly 
        distributed across the graph.

        Args:
            input_nodes (list[tuples]): Input to matrix generator functions 
            containing a list of nodes specifed as tuples. All the nodes 
            correspond to waypoints relevant to the tsp problem
        Returns:
            np.ndarray: Returns a 2 dimension connectivity matrix of size 
            n^2
        """
        # Euclidean distances between all nodes input to the graph
        Dist = distance.cdist(input_nodes, input_nodes, "euclidean")

        # Waypoints can only belong to one cluster
        Cnstrnt_wypnts = np.eye(Dist.shape[0], Dist.shape[1])
        Q = lamda_dist * Dist - lamda_cnstrnt * Cnstrnt_wypnts

        if self.fixed_pt:
            Q = self._stochastic_rounding(Q)
        return Q

    def _gen_clustering_Q_matrix(self, input_nodes, lamda_dist, lamda_cnstrnt):
        """Return the Q matrix that sets up the QUBO for the 
        clustering problem. The cluster centers are assumed to be uniformly 
        distributed across the graph.

        Args:
            input_nodes (list[tuples]): Input to matrix generator functions 
            containing a list of nodes specifed as tuples. First `num_vehicles`
            tuples correspond to the vehicle nodes.

        Returns:
            np.ndarray: Returns a 2 dimension connectivity matrix of size 
            n^2
        """
        Dist = distance.cdist(input_nodes, input_nodes, "euclidean")
        # TODO: Introduce cut-off distancing later to sparsify distance matrix 
        # later

        # Vehicles can only belong to one cluster
        Cnstrnt_vehicles = np.pad(
            np.eye(self.num_vehicles, self.num_vehicles),
            (
                (0, Dist.shape[0] - self.num_vehicles),
                (0, Dist.shape[1] - self.num_vehicles),
            ),
            "constant",
            constant_values=(0),
        )

        # Combine all matrices to get final Q matrix
        Q = lamda_dist * Dist - lamda_cnstrnt * Cnstrnt_vehicles
        if self.fixed_pt:
            Q = self._stochastic_rounding(Q)
        return Q

    def _stochastic_rounding(self, tensor):
        """function to rescale and stochastically round tensor to fixed point 
        values compatiable with Unsigned Mode on Loihi 2.

        Args:
            tensor (np.ndarray): floating-point tensor 

        Returns:
            (np.ndarray): fixed-point version of tensor that is passed as input
        """
        tensor_max = np.max(np.abs(tensor))
        tensor_min = np.min(np.abs(tensor))

        # Get sign mask of tensor to furnish signs for matrix later
        tensor_sign_mask = np.sign(tensor)
        scaled_tensor = (
            (self.max_fixed_pt_mant - self.min_fixed_pt_mant)
            * (np.abs(tensor) - tensor_min)
            / (tensor_max - tensor_min)
        )
        stchstc_rnded_tensor = np.floor(
            scaled_tensor + np.random.rand(tensor.shape[0], tensor.shape[1])
        )
        stchstc_rnded_tensor *= tensor_sign_mask
        return stchstc_rnded_tensor
