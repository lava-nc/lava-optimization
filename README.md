# Neuromorphic Constraint Optimization Library

**A library of solvers that leverage neuromorphic hardware for constrained optimization.**

Constrained optimization searches for the values of input variables that minimize or maximize a given objective function, while the variables are subject to constraints. This kind of problem is ubiquitous throughout scientific domains and industries.
Constrained optimization is a promising application for neuromorphic computing as
it [naturally aligns with the dynamics of spiking neural networks](https://doi.org/10.1109/JPROC.2021.3067593). When individual neurons represent states of variables, the neuronal connections can directly encode constraints between the variables: in its simplest form, recurrent inhibitory synapses connect neurons that represent mutually exclusive variable states, while recurrent excitatory synapses link neurons representing reinforcing states. Implemented on massively parallel neuromorphic hardware, such a spiking neural network can simultaneously evaluate conflicts and cost functions involving many variables, and update all variables accordingly. This allows a quick convergence towards an optimal state. In addition, the fine-scale timing dynamics of SNNs allow them to readily escape from local minima.

This Lava repository currently supports the following constraint optimization problems:

- Quadratic Programming (QP)
- Quadratic Unconstrained Binary Optimization (QUBO)

As we continue development, the library will support more constraint optimization problems that are relevant for robotics and operations research.
We currently plan the following development order in such a way that new solvers build on the capabilities of existing ones:

- Constraint Satisfaction Problems (CSP)
- Integer Linear Programming (ILP)
- Mixed-Integer Linear Programming (MILP)
- Mixed-Integer Quadratic Programming (MIQP)
- Linear Programming (LP)

 ![Overview_Solvers](https://user-images.githubusercontent.com/83413252/135428779-d128aaaa-54ed-4ae1-a5b1-8e0fcc08c96e.png?raw=true "Lava features a growing suite of constraint
	 optimization solvers")


## Taxonomy of Optimization Problems
More formally, the general form of a constrained optimization problem is:

$$
\displaystyle{\min_{x} \lbrace f(x) | g_i(x)	\leq  b,	h_i(x)	= c.\rbrace}
$$

Where $f(x)$ is the obective function to be optimized while $g(x)$ and $h(x)$ 
constrain the validity of $f(x)$ to regions in the state space satisfying the 
respective equality and inequality constraints. The vector $x$ can be
 continuous, discrete or a mixture of both. We can then construct the following 
 taxonomy of optimization problems according to thecharacteristics of the 
 variable domain and of $f$, $g$ and $h$:

![image](https://user-images.githubusercontent.com/83413252/192852018-dbc08018-ddda-4571-8494-cd1fbfa8405f.png)

In the long run, lava-optimization aims to offer support to solve all of the problems in the figure with a neuromorphic backend. 

## OptimizationSolver and OptimizationProblem Classes

The figure below shows the general architecture of the library.  We harness the general definition of constraint optimization problems to create OptimizationProblem instances by compossing Constraints, Variables and Cost classes which describe the characteristics of every subproblem class. Note that while a quadratic problem (QP) will be described by linear equality and inequelity constraints with variables on the continuous domain and a quadratic function, a constraint satisfaction problem (CSP) will be described by discrete constraints, defined by variable subsets and a binary relation describing the mutually allowed values for such discrete variables and will have a costant cost function with the pure goal of satisfying constraints.

An API for every problem class can be created by inheriting from OptimizationSolver and compossing particular flavours of Constraints, Variables and Cost. 

![image](https://user-images.githubusercontent.com/83413252/192851930-919035a7-122d-4a82-8032-f1acc6da717b.png)

The instance of an Optimization problem is the valid input for instaintiating the generic OptimizationSolver class. In this way, the OptimizationSolver interface is left fixed and the OptimizationProblem allows the greatest flexibility for creating new APIs. Under the hood, the OptimizationSolver understands the compossed structure of the OptimizationProblem and will in turn compose the required solver components and Lava processes. 

## Tutorials

### QP Tutorial
- [Solving LASSO.](https://github.com/lava-nc/lava-optimization/tree/main/tutorials/qp/tutorial_01_solving_lasso.ipynb)

### Solving QP problems (After merging with the OptimizationSolver QPSolver will be deprecated)
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

### QUBO Tutorial
- [Solving MIS.](https://github.com/lava-nc/lava-optimization/tree/main/tutorials/qubo/tutorial_01_solving_mis.ipynb)

### Solving QUBO using the Generic OptimizationSolver
```python
from lava.lib.optimization.solvers.generic.solver import solve, OptimizationSolver
from lava.lib.optimization.problems.problems import QUBO

q1 = np.asarray([[-5, 2, 4, 0],
                 [ 2,-3, 1, 0],
                 [ 4, 1,-8, 5],
                 [ 0, 0, 5,-6]]))

q2 =-np.asarray([[ 1,-3,-3,-3],
                 [-3, 1, 0, 0],
                 [-3, 0, 1,-3],
                 [-3, 0,-3, 1]]))

# Define problems:
qubo1=QUBO(q1)
qubo2=QUBO(q2)

# solve using solve:
sol_qubo1 = solve(problem = qubo1, timeout=-1, backend=“Loihi2”)
sol_qubo2 = solve(problem = qubo2, timeout=-1, backend=“Loihi2”)

# Solve using OptimizationSolver:
solver = OptimizationSolver(problem=qubo1)

solutions = []
for trial in range(10):
	solution = solver.solve(timeout=3000, target_cost=-50, backend=“Loihi2”)
	solutions.append(solution)
![image](https://user-images.githubusercontent.com/83706504/191646177-bfb429ac-6555-4d37-8d0a-c6074db8e455.png)

# When exposing hyperparameters use more mathematical names than neuron ones.

step_size <- bias
```

## Requirements
- Working installation of Lava, installed automatically with poetry below. [ For custom installs see Lava installation
tutorial.](https://github.com/lava-nc/lava/blob/main/tutorials/in_depth/tutorial01_installing_lava.ipynb)

## Installation

#### [Linux/MacOS]
```bash
cd $HOME
git clone git@github.com:lava-nc/lava-optimization.git
cd lava-optimization
pip install "poetry>=1.1.13"
poetry config virtualenvs.in-project true
poetry install
source .venv/bin/activate
pytest
```
#### [Windows]
```powershell
# Commands using PowerShell
cd $HOME
git clone git@github.com:lava-nc/lava-optimization.git
cd lava-optimization
python3 -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install "poetry>=1.1.13"
poetry config virtualenvs.in-project true
poetry install
pytest
```
You should expect the following output after running the unit tests:
```
$ pytest
============================= test session starts ==============================
platform linux -- Python 3.8.10, pytest-7.1.2, pluggy-1.0.0
rootdir: /home/user/src/lava-optimization, configfile: pyproject.toml, testpaths: tests
plugins: cov-3.0.0
collected 14 items                                                                                                                                                            

tests/lava/lib/optimization/solvers/qp/test_models.py .......                                                                                                           [ 50%]
tests/lava/lib/optimization/solvers/qp/test_process.py .......                                                                                                          [100%]

---------- coverage: platform linux, python 3.8.10-final-0 -----------
Name                                                Stmts   Miss  Cover   Missing
---------------------------------------------------------------------------------
src/lava/lib/optimization/__init__.py                   0      0   100%
src/lava/lib/optimization/problems/__init__.py          0      0   100%
src/lava/lib/optimization/problems/problems.py         43     29    33%   48-107, 111, 115, 119, 123, 127
src/lava/lib/optimization/solvers/__init__.py           0      0   100%
src/lava/lib/optimization/solvers/qp/__init__.py        0      0   100%
src/lava/lib/optimization/solvers/qp/models.py        136      4    97%   97-98, 102-104
src/lava/lib/optimization/solvers/qp/processes.py      75      0   100%
src/lava/lib/optimization/solvers/qp/solver.py         26     18    31%   42-45, 62-104
---------------------------------------------------------------------------------
TOTAL                                                 280     51    82%

Required test coverage of 45.0% reached. Total coverage: 81.79%

=============== 14 passed in 8.95s ==============================================
```

### [Alternative] Installing Lava via Conda
If you use the Conda package manager, you can simply install the Lava package
via:
```bash
conda install lava-optimization -c conda-forge
```

Alternatively with intel numpy and scipy:

```bash
conda create -n lava-optimization python=3.9 -c intel
conda activate lava-optimization
conda install -n lava-optimization -c intel numpy scipy
conda install -n lava-optimization -c conda-forge lava-optimization --freeze-installed
```

