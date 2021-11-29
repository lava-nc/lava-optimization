# Neuromorphic Constraint Optimization Library

**A library of solvers that leverage neuromorphic hardware for constrained optimization.**

Constrained optimization searches for the values of input variables that minimize or maximize a given objective function, while the variables are subject to constraints. This kind of problem is ubiquitous throughout scientific domains and industries.
Constrained optimization is a promising application for neuromorphic computing as
it [naturally aligns with the dynamics of spiking neural networks](https://doi.org/10.1109/JPROC.2021.3067593). When individual neurons represent states of variables, the neuronal connections can directly encode constraints between the variables: in its simplest form, recurrent inhibitory synapses connect neurons that represent mutually exclusive variable states, while recurrent excitatory synapses link neurons representing reinforcing states. Implemented on massively parallel neuromorphic hardware, such a spiking neural network can simultaneously evaluate conflicts and cost functions involving many variables, and update all variables accordingly. This allows a quick convergence towards an optimal state. In addition, the fine-scale timing dynamics of SNNs allow them to readily escape from local minima.

This Lava repository currently provides constraint optimization solvers that leverage the benefits of neuromorphic computing for the following problems:

- Quadratic Programming (QP)

In the future, the library will be extended by solvers targeting further constraint optimization problems that are relevant for robotics and operations research.
The current focus lies on solvers for the following problems:

- Constraint Satisfaction Problems (CSP)
- Quadratic Unconstrained Binary Optimization (QUBO)
- Integer Linear Programming (ILP)
- Linear Programming (LP)
- Mixed-Integer Linear Programming (MILP)
- Mixed-Integer Quadratic Programming (MIQP)

 ![Overview_Solvers](https://user-images.githubusercontent.com/83413252/135428779-d128aaaa-54ed-4ae1-a5b1-8e0fcc08c96e.png?raw=true "Lava features a growing suite of constraint
	 optimization solvers")


## Tutorials

### QP Solver

- [Solving LASSO.](https://github.com/lava-nc/lava-optimization/tree/main/src/tutorials/qp/tutorial_01_solving_lasso.ipynb)


## Example

### QP Solver

```python
import numpy as np
from lava.lib.optimization.problems.problems import QP
from lava.lib.optimization.solvers.qp.solver import QPSolver

Q = np.array([[100, 0, 0], [0, 15, 0], [0, 0, 5]])
p = np.array([[1, 2, 1]]).T
A = -np.array([[1, 2, 2], [2, 100, 3]])
k = -np.array([[-50, 50]]).T

alpha, beta = 0.001, 1
alpha_d, beta_g = 10000, 10000
iterations = 400
problem = QP(Q, p, A, k)
solver = QPSolver(
                alpha=alpha,
                beta=beta,
                alpha_decay_schedule=alpha_d,
                beta_growth_schedule=beta_g,
                )
solver.solve(problem, iterations=iterations)
```

### Coming up next: CSPSolver
```python
from lava.lib.optimization import CspSolver

variables = ['var1', 'var2', 'var3']
domains = dict(var1 = {0, 1, 2}, var2 = {'a', 'b', 'c'}, var3 = {'red', 'blue', 'green'})
solver = CspSolver()
problem = CSP(variables, domains, constraints)
solution, t_sol = solver.solve(problem, timeout=5000, backend='Loihi2', profile=True)
print(solver.time_to_solution[-1], solver.energy_to_solution[-1])
```

## Requirements
- Working installation of Lava [see Lava installation
tutorial.](https://github.com/lava-nc/lava/blob/main/tutorials/in_depth/tutorial01_installing_lava.ipynb)

## Setup
- Clone the lava-optimization repository.
- Add lava-optimization/src to your PYTHONPATH variable. For example, from
lava-optimization/src/ run the following on your terminal:
 ```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```
You should do this only after having added Lava to your PYTHONPATH.
This order is required for namespaces to be well defined.
