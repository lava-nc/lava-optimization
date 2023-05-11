import os
from pprint import pprint

import numpy as np
import networkx as ntx
import typing as ty
import matplotlib.pyplot as plt

from lava.lib.optimization.apps.vrp.problems import VRP
from lava.lib.optimization.apps.vrp.solver import VRPSolver
from vrp.core.solver import VRPySolver
from vrp.demos.drone_swarm import DroneSwarmDemo


def get_v_w_coords(num_v=10, num_w=100, ranseed=42313) -> ty.Tuple[ty.List[
    ty.Tuple], ty.List[ty.Tuple]]:
    np.random.seed(ranseed)
    dswm = DroneSwarmDemo(num_agents=num_v,
                          agent_distribution_func=np.random.uniform,
                          agent_distribution_func_args={'low': -7, 'high': 7,
                                                        'size': 2},
                          agent_velocity=1.,
                          num_waypoints=num_w,
                          waypoint_distribution_func=np.random.uniform,
                          waypoint_distribution_func_args={'low': -7, 'high':
                              7, 'size': 2},
                          waypoint_targets=[[0, 0]],
                          waypoint_velocity=0.5,
                          solver=VRPySolver())
    v_coords = [tuple(val.position) for val in dswm.agents.values()]
    w_coords = [tuple(val.position) for val in dswm.waypoints.values()]
    return v_coords, w_coords


def scenario_sweep():
    v_c, w_c = get_v_w_coords(num_v=40, num_w=200)
    npvc = np.array(v_c)
    npwc = np.array(w_c)
    vrp_instance = VRP(node_coords=w_c, vehicle_coords=v_c)
    solver = VRPSolver(vrp=vrp_instance)
    plt.scatter(npvc[:, 0], npvc[:, 1], s=20, c='r', marker='*')
    plt.scatter(npwc[:, 0], npwc[:, 1], s=20, c='b', marker='o')
    plt.show()


if __name__ == '__main__':
    scenario_sweep()
