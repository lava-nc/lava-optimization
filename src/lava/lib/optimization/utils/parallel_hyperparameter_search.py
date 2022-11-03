#Copyright (C) 2022 Intel Corporation*<br>
#SPDX-License-Identifier: BSD-3-Clause*<br>
#See: https://spdx.org/licenses/*


# Quadratic Unconstrained Binary Optimization (QUBO) with Lava
#To solve QUBOs in Lava, we import the corresponding modules.

# Interface for QUBO problems
from lava.lib.optimization.problems.problems import QUBO
# Generic optimization solver
from lava.lib.optimization.solvers.generic.solver import OptimizationSolver, solve

#In addition, we import auxiliary modules to generate the workloads and run the solver.

import os
import numpy as np
import networkx as ntx
#If Loihi 2 hardware is available, we can take advantage of the large speed and energy efficiency of this chip to solve QUBOs. To access the chip, we must configure the following environment variables:

from lava.utils.system import Loihi2
Loihi2.preferred_partition = "oheogulch"
loihi2_is_available = Loihi2.is_loihi2_available

if loihi2_is_available:
    # Enable SLURM, the workload manager used to distribute Loihi2 resources to users
    os.environ['SLURM'] = '1'
    os.environ["PARTITION"] = "oheogulch"

example_workloads = {
    1: {'n_vert': 45, 'p_edge': 0.7, 'seed_graph': 5530, 'w_diag': 1, 'w_off': 4, 'cost_optimal': -5.0},
    2: {'n_vert': 45, 'p_edge': 0.5, 'seed_graph': 7865, 'w_diag': 1, 'w_off': 4, 'cost_optimal': -7.0, },
    3: {'n_vert': 45, 'p_edge': 0.3, 'seed_graph': 7079, 'w_diag': 3, 'w_off': 8, 'cost_optimal': -33.0},
    4: {'n_vert': 200, 'p_edge': 0.9, 'seed_graph': 6701, 'w_diag': 3, 'w_off': 8, 'cost_optimal': -12.0},
    5: {'n_vert': 200, 'p_edge': 0.7, 'seed_graph': 2999, 'w_diag': 3, 'w_off': 8, 'cost_optimal': -21.0},
    6: {'n_vert': 200, 'p_edge': 0.5, 'seed_graph': 6814, 'w_diag': 3, 'w_off': 8, 'cost_optimal': -33.0},
    7: {'n_vert': 500, 'p_edge': 0.9, 'seed_graph': 7757, 'w_diag': 3, 'w_off': 8, 'cost_optimal': -15.0},
    8: {'n_vert': 500, 'p_edge': 0.7, 'seed_graph': 7757, 'w_diag': 3, 'w_off': 8, 'cost_optimal': -24.0},
    9: {'n_vert': 700, 'p_edge': 0.9, 'seed_graph': 4155, 'w_diag': 1, 'w_off': 4, 'cost_optimal': -5.0},
    10: {'n_vert': 700, 'p_edge': 0.85, 'seed_graph': 4840, 'w_diag': 3, 'w_off': 7, 'cost_optimal': -18.0},
    11: {'n_vert': 1000, 'p_edge': 0.9, 'seed_graph': 4044, 'w_diag': 3, 'w_off': 7, 'cost_optimal': -27.0},
    }


workload = example_workloads[7]

# Import utility functions to create and analyze MIS workloads
from lava.lib.optimization.utils.generators.mis import MISProblem

# Create an undirected graph with 700 vertices and a 
# probability of 70% that any two vertices are randomly connected
mis = MISProblem(num_vertices=workload['n_vert'], connection_prob=workload['p_edge'], seed=workload['seed_graph'])

# Translate the MIS problem for this graph into a QUBO matrix
w_mult = 3 # CAN BE ADJUSTED
q = mis.get_qubo_matrix(w_diag=w_mult*workload['w_diag'], w_off=w_mult*workload['w_off'])

# Create the qubo problem
qubo_problem = QUBO(q)

# Find the optimal solution to the MIS problem
solution_opt = mis.find_maximum_independent_set()

# Calculate the QUBO cost of the optimal solution
cost_opt = qubo_problem.evaluate_cost(solution=solution_opt)

solver = OptimizationSolver(qubo_problem)

print(cost_opt)

#Solve the qubo problem with a set of hyperparameters

solver = OptimizationSolver(qubo_problem)

# Provide hyperparameters for the solver  # CAN BE ADJUSTED
# Guidance on the hyperparameter search will be provided in the deep dive tutorial
hyperparameters = {
    "steps_to_fire": 171,
    "noise_amplitude": 13,
    "noise_precision": 13,
    "step_size": 1110,}

if loihi2_is_available:
    backend = 'Loihi2'
else:
    backend = 'CPU'

# Solve the QUBO using Lava's OptimizationSolver on CPU
# Change "backend='Loihi2'" if your system has physical access to this chip
# solution_loihi = solver.solve(timeout=190000, # Todo: Set to rough number of time steps required to obtain optimum solution
#                               hyperparameters=hyperparameters,
#                               target_cost=cost_opt,
#                               backend=backend)

# print(f'\nSolution of the provided QUBO: {np.where(solution_loihi == 1.)[0]}.')

def stop(best_cost, best_to_sol):
    return best_cost <= cost_opt

#Stochastic search for a set of optimal hyperparameters with SolverTuner

from lava.lib.optimization.utils.solver_tuner import SolverTuner

params_grid = {
    "steps_to_fire": (1, 731),
    "noise_amplitude": (1, 17),
    "noise_precision": (1, 19),
    "step_size": (700, 2710),
}

solver_tuner = SolverTuner(params_grid=params_grid)

params = {"timeout": 10000000,
          "target_cost": int(cost_opt),
          "backend": "Loihi2"}
print(params)

hyperparams, success = solver_tuner.tune(solver=solver,
                                         solver_parameters=params,
                                         stopping_condition=stop)
print(params)


# Find the optimal solution to the MIS problem
solution_opt = mis.find_maximum_independent_set()

# Calculate the QUBO cost of the optimal solution
cost_opt = qubo_problem.evaluate_cost(solution=solution_opt)

# Calculate the QUBO cost of Lava's solution
cost_lava = qubo_problem.evaluate_cost(solution=solution_loihi)

print(f'QUBO cost of solution: {cost_lava} (Lava) vs. {cost_opt} (optimal)\n')

"""Pickl database for hyperparameters"""
class Hyperparameters:pass
selected_parameters = Hyperparameters()

selected_parameters.steps_to_fire = 
selected_parameters.noise_amplitude =
selected_parameters.noise_precision =
selected_parameters.step_size =


filehandler = open(b"hyperparameter_obj","wb")
pickle.dump(,filehandler)
