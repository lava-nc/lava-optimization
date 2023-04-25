import os
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import numpy as np
import networkx as ntx
import matplotlib.pyplot as plt

from lava.utils.system import Loihi2
from lava.lib.optimization.problems.problems import QUBO
from lava.lib.optimization.solvers.generic.solver import OptimizationSolver, solve, SolverConfig
from lava.lib.optimization.utils.generators.mis import MISProblem


def is_request_reachable(view_coords, satellite_id, request_coords, view_height):
    return view_coords[satellite_id] <= request_coords[1] <= view_coords[satellite_id] + view_height


def is_same_satellite(n1, n2):
    return n1[1] == n2[1]


def is_movable(n1, n2):
    xdist = abs(n1[2] - n2[2])
    ydist = abs(n1[3] - n2[3])
    return move_rate * xdist >= ydist


def is_same_request(n1, n2):
    return (n1[2] == n2[2]) and (n1[3] == n2[3])


def is_feasible(n1, n2):
    return not is_same_request(n1, n2) and (not is_same_satellite(n1, n2) or is_movable(n1, n2))


def is_same_node(n1, n2):
    return n1[0] == n2[0]


def plot_problem(request_coords, graph, adjacency, view_coords, view_height):
    plt.figure(figsize=(12,4), dpi=120)
    plt.subplot(131)
    plt.scatter(request_coords[:,0], request_coords[:,1], s=2)
    for y in view_coords:
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        verts = [[-0.05, y + view_height / 2],
                [0.05, y + view_height],
                [0.05, y + 0.0],
                [-0.05, y + 0.0]]
        plt.gca().add_patch(PathPatch(Path(verts, codes), ec='none', alpha=0.3, fc='lightblue'))
        plt.scatter([-0.05], [y + view_height / 2], s=10, marker='s', c='gray')
        plt.plot([0, 1], [y + view_height / 2, y + view_height / 2], 'C1--', lw=0.75)
    plt.xticks([])
    plt.yticks([])
    plt.title(f'Schedule {num_satellites} satellites to observe {num_requests} targets.')
    plt.subplot(132)
    ntx.draw_networkx(graph, with_labels=False, node_size=2, width=0.5)
    plt.title(f'Infeasibility graph with {g.number_of_nodes()} nodes.')
    plt.subplot(133)
    plt.imshow(adjacency, aspect='auto')
    plt.title(f'Adjacency matrix has {adjacency.mean():.2%} connectivity.')
    plt.yticks([])
    plt.tight_layout()
    plt.show()


def plot_solutions(request_coords, satellite_ids, netx_solution, lava_solution, qubo_problem, solver_report):
    plt.figure(figsize=(15,5), dpi=120)

    plt.subplot(131)
    plt.scatter(request_coords[:,0], request_coords[:,1], s=2, c='r')
    for i in satellite_ids:
        sat_plan = netx_solution[:,1] == i
        plt.plot(netx_solution[sat_plan,2], netx_solution[sat_plan,3], 'bo-', markersize=2, lw=0.75)
    plt.title(f'The optimal solution satisfies {netx_solution.shape[0]} requests.')

    plt.subplot(132)
    plt.scatter(request_coords[:,0], request_coords[:,1], s=2, c='r')
    for i in satellite_ids:
        sat_plan = lava_solution[:,1] == i
        plt.plot(lava_solution[sat_plan,2], lava_solution[sat_plan,3], 'bo-', markersize=2, lw=0.75)
    plt.title(f'The lava solution satisfies {lava_solution.shape[0]} requests.')

    plt.subplot(233)
    plt.plot(solver_report.cost_timeseries.T, )
    plt.title(f'Optimal solution costs is {qubo_problem.compute_cost(solution_nodes)}\nQUBO Solution cost is {solver_report.best_cost}')

    plt.subplot(236)
    longest_plan = 1
    for i in satellite_ids:
        sat_plan = lava_solution[:,1] == i
        longest_plan = max(longest_plan, sat_plan.sum() - 1)
        x = lava_solution[sat_plan,2]
        y = lava_solution[sat_plan,3]
        plt.plot(abs(np.diff(y) / np.diff(x)))
    plt.plot([0, longest_plan], [move_rate, move_rate], '--')
    plt.title(f'Satellite move rates')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    num_satellites = 24
    view_height = 0.25
    view_coords = np.linspace(-0.1, 1.1 - view_height, num_satellites)
    num_requests = 400
    move_rate = 2
    target_cost = int(-0.99 * num_requests)
    np.random.seed(42)

    g = ntx.Graph()
    satellite_ids = range(num_satellites)
    request_coords = np.random.random((num_requests, 2))
    request_coords = request_coords[np.argsort(request_coords[:,0]),:]

    node_id = 0
    for i in satellite_ids:
        for j in range(num_requests):
            if is_request_reachable(view_coords, i, request_coords[j,:], view_height):
                g.add_node((node_id, i, request_coords[j,0], request_coords[j,1]))
                node_id += 1
    num_valid_nodes = node_id

    adjacency = np.zeros((num_valid_nodes, num_valid_nodes), dtype=int)
    for n1 in g.nodes:
        for n2 in g.nodes:
            if not is_same_node(n1, n2) and not is_feasible(n1, n2):
                adjacency[n1[0], n2[0]] = 1
                g.add_edge(n1, n2)

    adjacency = np.triu(adjacency)
    adjacency += adjacency.T - 2 * np.diag(adjacency.diagonal())

    plot_problem(request_coords, g, adjacency, view_coords, view_height)

    mis = MISProblem(num_vertices=num_satellites * num_requests, connection_prob=0.0, seed=44)
    # TODO: Edit MISProblem to accept an adjacency matrix in constructor
    mis._adjacency = adjacency
    solution = np.array([(0, 0, request_coords[0,0], request_coords[0,1])])
    solution_nodes = np.zeros(num_valid_nodes, dtype=int)
    solution_nodes[0] = 1

    q = mis.get_qubo_matrix(w_diag=1, w_off=8)
    qubo_problem = QUBO(q)
    solver = OptimizationSolver(qubo_problem)

    hyperparameters = {
        "temperature": int(8),
        "refract": np.random.randint(64, 127, qubo_problem.num_variables),
        "refract_counter": np.random.randint(0, 64, qubo_problem.num_variables),
    }

    if Loihi2.is_loihi2_available:
        backend = 'Loihi2'
        Loihi2.preferred_partition = "oheogulch"
    else:
        backend = 'CPU'

    solver_report = solver.solve(
        config=SolverConfig(
            timeout=1000,
            hyperparameters=hyperparameters,
            target_cost=int(target_cost),
            backend=backend,
            probe_cost=True
        )
    )

    qubo_state = solver_report.best_state
    lava_solution = np.array(g.nodes)[np.where(qubo_state)[0]]

    plot_solutions(request_coords, satellite_ids, solution, lava_solution, qubo_problem, solver_report)
