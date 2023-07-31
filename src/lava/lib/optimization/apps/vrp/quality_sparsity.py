import numpy as np
import networkx as ntx
import argparse
from lava.lib.optimization.apps.vrp.problems import VRP
from lava.lib.optimization.apps.vrp.solver import VRPSolver, VRPConfig, \
    CoreSolver


def main(j=0):
    max_dist_cutoff_fraction_list = np.around(np.geomspace(1.0, 0.1,
                                                           15), 2).tolist()
    max_dist_cutoff_fraction_list.reverse()
    print(f"Cutoff fractions: {max_dist_cutoff_fraction_list}")
    np.random.seed(42313)
    # max_dist_cutoff_fraction_list = [0.0, 0.1, 0.12, 0.14, 0.16, 0.19,
    #                                  0.23, 0.27, 0.32, 0.37, 0.44, 0.52]
    dist_sparsity_list = []
    dist_proxy_sparsity_list = []
    total_cost_list = []
    frac_wp_clustered_list = []
    clustering_extent_list = []

    print(f"Loading vrp_instance_{j}.dat")
    all_coords = np.loadtxt(f"vrp_instance_{j}.dat")
    v_c = [tuple(coords) for coords in all_coords[:10, :].tolist()]
    w_c = [tuple(coords) for coords in all_coords[10:, :].tolist()]
    vrp_instance = VRP(node_coords=w_c, vehicle_coords=v_c)
    solver = VRPSolver(vrp=vrp_instance)
    print(f"Iterating over cutoff fractions\n")
    for cutoff_factor in max_dist_cutoff_fraction_list:
        scfg = VRPConfig(backend="Loihi2",
                         core_solver=CoreSolver.LAVA_QUBO,
                         do_distance_sparsification=True,
                         sparsification_algo="edge_prune",
                         max_dist_cutoff_fraction=cutoff_factor,
                         hyperparameters={},
                         target_cost=-1000000,
                         timeout=10000,
                         probe_time=False,
                         only_gen_q_mats=False,
                         only_cluster=True,
                         profile_q_mat_gen_clust=True,
                         profile_q_mat_gen_tsp=False,
                         log_level=40)
        try:
            clusters, routes = solver.solve(scfg=scfg)
            cluster_check = np.sum(clusters) / 110
        except ValueError:
            cluster_check = -1.0
            routes = dict(
                zip(
                    range(1, vrp_instance.num_vehicles + 1),
                    [[-1]] * vrp_instance.num_vehicles
                )
            )
        dist_sparsity_list.append(solver.dist_sparsity)
        dist_proxy_sparsity_list.append(solver.dist_proxy_sparsity)
        flat_waypoint_list = []
        total_cost = 0
        for route in routes.values():
            flat_waypoint_list.extend(route)
            try:
                route_cost = ntx.path_weight(
                    solver.problem.problem_graph, route, weight="cost")
            except ntx.exception.NetworkXNoPath:
                route_cost = -1
            total_cost += route_cost
        flat_waypoint_list.sort()
        frac_wp_clusered = np.sum(np.in1d(np.arange(11, 111),
                                          flat_waypoint_list)) / 100
        frac_wp_clustered_list.append(frac_wp_clusered)
        total_cost_list.append(total_cost)
        clustering_extent_list.append(cluster_check)

        np.savetxt(f"problem_{j}_dist_sp.dat",
                   np.array(dist_sparsity_list), fmt="%.3f")
        np.savetxt(f"problem_{j}_distpr_sp.dat",
                   np.array(dist_proxy_sparsity_list), fmt="%.3f")
        np.savetxt(f"problem_{j}_total_cost.dat",
                   np.array(total_cost_list), fmt="%.3f")
        np.savetxt(f"problem_{j}_frac_wp_clustered.dat",
                   np.array(frac_wp_clustered_list), fmt="%.3f")
        np.savetxt(f"problem_{j}_clustering_extent.dat",
                   np.array(clustering_extent_list), fmt="%.2f")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="quality_sparsity.py")
    parser.add_argument("prob_num", type=int, choices=list(range(15)))
    args = parser.parse_args()
    print(f"\n------------------\nProblem number: "
          f"{args.prob_num}\n------------------\n")
    main(j=args.prob_num)
