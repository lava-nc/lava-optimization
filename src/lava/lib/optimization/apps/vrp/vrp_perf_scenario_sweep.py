import logging
import os
import time
from dataclasses import dataclass
from pprint import pprint
from io import StringIO
import sys

import numpy as np
import numpy.typing as npty
import networkx as ntx
import typing as ty
import matplotlib.pyplot as plt

from lava.lib.optimization.apps.vrp.problems import VRP
from lava.lib.optimization.apps.vrp.solver import VRPSolver, VRPConfig, \
    CoreSolver
from lava.magma.core.callback_fx import NxSdkCallbackFx, NxBoard
from vrp.core.solver import VRPySolver
from vrp.demos.drone_swarm import DroneSwarmDemo


# Ref: https://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
class StdOutCapture(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout


def get_v_w_coords(num_v=10, num_w=100) -> ty.Tuple[ty.List[
    ty.Tuple], ty.List[ty.Tuple]]:
    dswm = DroneSwarmDemo(num_agents=num_v,
                          agent_distribution_func=np.random.uniform,
                          agent_distribution_func_args={'low': -4,
                                                        'high': 4,
                                                        'size': 2},
                          agent_velocity=1.,
                          num_waypoints=num_w,
                          waypoint_distribution_func=np.random.uniform,
                          waypoint_distribution_func_args={'low': -4,
                                                           'high': 4,
                                                           'size': 2},
                          waypoint_targets=[[0, 0]],
                          waypoint_velocity=0.5,
                          solver=VRPySolver())
    v_coords = [tuple(val.position) for val in dswm.agents.values()]
    w_coords = [tuple(val.position) for val in dswm.waypoints.values()]
    return v_coords, w_coords


def run_one_scenario(v_c: ty.List[ty.Tuple[int, int]],
                     w_c: ty.List[ty.Tuple[int, int]],
                     scfg: VRPConfig) -> \
        ty.Tuple[VRPSolver, npty.NDArray, ty.Dict[int, ty.List[int]]]:

    vrp_instance = VRP(node_coords=w_c, vehicle_coords=v_c)
    solver = VRPSolver(vrp=vrp_instance)
    clustering_solution, tsp_routes = solver.solve(scfg)
    # tmp = solver.clust_solver.solver_process.finders[
    #     0].cost_minimizer.coefficients_2nd_order.weights.get()
    # print(tmp)
    # solver.clust_solver.solver_process.finders[
    #     0].cost_minimizer.coefficients_2nd_order.weights.set(tmp//2)
    # solver.solve(scfg)
    # solver.clust_solver.solver_process.stop()
    # print(f"Iter {j}: {solver.clust_q_shape[0]}x{solver.clust_q_shape[1]} "
    #       f"clustering matrix: {solver.q_gen_time_clust} s")
    # for idx in range(len(solver.tsp_q_shapes)):
    #     print(f"Iter {j}: "
    #           f"{solver.tsp_q_shapes[idx][0]}x"
    #           f"{solver.tsp_q_shapes[idx][1]} "
    #           f"TSP matrix: {solver.q_gen_times_tsp[idx]} s")
    # # plt.scatter(npvc[:, 0], npvc[:, 1], s=20, c='r', marker='*')
    # plt.scatter(npwc[:, 0], npwc[:, 1], s=20, c='b', marker='o')
    # plt.show()
    return solver, clustering_solution, tsp_routes


def run_multiple_scenarios(scenario_list):
    solver_list = []
    scfg_list = []
    for scidx, scenario in enumerate(scenario_list):
        vc, wc = get_v_w_coords(num_v=scenario.num_vehicles,
                                num_w=scenario.num_waypoints)
        scfg = scenario.vrpcfg
        solver = run_one_scenario(v_c=vc, w_c=wc, scfg=scfg)

        scfg_list.append(scfg)
        solver_list.append(solver)

    # probe_time=False,
    # sparsify_dist=False,
    # sparsification_algo="cutoff",
    # max_cutoff_frac=1.0,
    # only_gen_q_mats=True,
    # profile_q_mat_clust=False,
    # profile_q_mat_tsp=False,
    # log_level=40):
    return solver_list, scfg_list


if __name__ == '__main__':
    cfg = VRPConfig(backend="Loihi2",
                    core_solver=CoreSolver.LAVA_QUBO,
                    hyperparameters={},
                    target_cost=-1000000,
                    timeout=10000,
                    probe_time=False,
                    do_distance_sparsification=False,
                    sparsification_algo="edge_prune",
                    max_dist_cutoff_fraction=0.52,
                    only_gen_q_mats=False,
                    only_cluster=True,
                    profile_q_mat_gen_clust=True,
                    profile_q_mat_gen_tsp=False,
                    log_level=20)
    vc, wc = get_v_w_coords(num_v=4,
                            num_w=10)
    # all_coords = np.loadtxt(f"vrp_instance_0.dat")
    # vc = [tuple(coords) for coords in all_coords[:10, :].tolist()]
    # wc = [tuple(coords) for coords in all_coords[10:, :].tolist()]
    plt.scatter(np.array(vc)[:, 0], np.array(vc)[:, 1], marker='*', s=20, c='r')
    plt.scatter(np.array(wc)[:, 0], np.array(wc)[:, 1], marker='o', s=20, c='b')

    solver, clustering_sol, tsp_rts = run_one_scenario(v_c=vc,
                                                       w_c=wc,
                                                       scfg=cfg)
# total_tts = np.round(np.sum(solver.clust_profiler.execution_time) * 1e6, 2)
# mean_tts = np.round(np.mean(solver.clust_profiler.execution_time) * 1e6, 2)
# std_tts = np.round(np.std(solver.clust_profiler.execution_time) * 1e6, 2)
# max_tts = np.round(np.max(solver.clust_profiler.execution_time) * 1e6, 2)
# min_tts = np.round(np.min(solver.clust_profiler.execution_time) * 1e6, 2)

    logger = logging.getLogger("VRP")
# logger.info(f"Exec time: "
#             f"{total_tts.item()}, {mean_tts.item()}, {std_tts.item()}, "
#             f"{max_tts.item()}, {min_tts.item()}")
    logger.info(f"Q Mat Gen Time Clust: {solver.q_gen_time_clust}")
    logger.info(f"Q Mat Size Clust: "
                f"{solver.clust_q_shape[0] * solver.clust_q_shape[1]}")
    logger.info(f"Dist sparsity: {solver.dist_sparsity}")
    logger.info(f"Dist proxy sparsity: {solver.dist_proxy_sparsity}")
    logger.info(f"Clustering solution check 1: "
                f"{np.sum(clustering_sol, axis=0)}")
    logger.info(f"Clustering solution check 2: "
                f"{np.sum(clustering_sol, axis=1)}, shape: "
                f"{np.sum(clustering_sol, axis=1).shape}")
    logger.info(f"TSP routes: \n {tsp_rts}")
    # all_wps = []
    # for rt in tsp_rts.values():
    #     all_wps.extend(rt)
    # all_wps.sort()
    # logger.info(f"Number of wayp"
    #             f"oints TSPed: {np.sum(np.in1d(np.arange(11, 111), all_wps))}")
    colours = ['g', 'c', 'm', 'y', 'k']
    for j, col in enumerate(clustering_sol.T):
        clust_indices = np.nonzero(col)
        v_idx = clust_indices[0][clust_indices[0] < 4]
        w_idx = clust_indices[0][clust_indices[0] >= 4] - 4
        plt.scatter(np.array(vc)[v_idx, 0], np.array(vc)[v_idx, 1],
                    marker='o', s=50, edgecolors=colours[j], facecolors="none")
        plt.scatter(np.array(wc)[w_idx, 0], np.array(wc)[w_idx, 1],
                    marker='o', s=50, edgecolors=colours[j], facecolors="none")
    plt.show()
    # for j, q_gen_time_tsp in enumerate(solver.q_gen_times_tsp):
    #     logger.info(f"Q Mat Gen Time TSP:: {j} :: {q_gen_time_tsp}")
    # for j, q_mat_tsp_shape in enumerate(solver.tsp_q_shapes):
    #     logger.info(f"Q Mat Size TSP:: {j} :: "
    #                 f"{q_mat_tsp_shape[0] * q_mat_tsp_shape[1]}")
