# Neuromorphic Constrained Optimization Library

**A library of solvers that leverage neuromorphic hardware for constrained optimization.**

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#taxonomy-of-optimization-problems">Taxonomy of Optimization Problems</a></li>
        <li><a href="#optimizationsolver-and-optimizationproblem-classes">OptimizationSolver and OptimizationProblem Classes</a></li>
      </ul>
    </li>
    <li>
      <a href="#tutorials">Tutorials</a>
      <ul>
        <li><a href="#quadratic-programming">Quadratic Programming</a></li>
        <li><a href="#quadratic-unconstrained-binary-optimization">Quadratic Unconstrained Binary Optimization</a></li>
      </ul>
    </li>
    <li>
      <a href="#examples">Examples</a>
      <ul>
        <li><a href="#solving-qp-problems">Solving QP</a></li>
        <li><a href="#solving-qubo">Solving QUBO</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#requirements">Requirements</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
  </ol>
</details>

## About the Project 

Constrained optimization searches for the values of input variables that minimize or maximize a given objective function, while the variables are subject to constraints. This kind of problem is ubiquitous throughout scientific domains and industries.
Constrained optimization is a promising application for neuromorphic computing as
it [naturally aligns with the dynamics of spiking neural networks](https://doi.org/10.1109/JPROC.2021.3067593). When individual neurons represent states of variables, the neuronal connections can directly encode constraints between the variables: in its simplest form, recurrent inhibitory synapses connect neurons that represent mutually exclusive variable states, while recurrent excitatory synapses link neurons representing reinforcing states. Implemented on massively parallel neuromorphic hardware, such a spiking neural network can simultaneously evaluate conflicts and cost functions involving many variables, and update all variables accordingly. This allows a quick convergence towards an optimal state. In addition, the fine-scale timing dynamics of SNNs allow them to readily escape from local minima.

This Lava repository currently supports solvers for the following constrained optimization problems:

- Quadratic Programming (QP)
- Quadratic Unconstrained Binary Optimization (QUBO)

As we continue development, the library will support more constrained optimization problems that are relevant for robotics and operations research.
We currently plan the following development order in such a way that new solvers build on the capabilities of existing ones:

- Constraint Satisfaction Problems (CSP) [problem interface already available]
- Integer Linear Programming (ILP)
- Mixed-Integer Linear Programming (MILP)
- Mixed-Integer Quadratic Programming (MIQP)
- Linear Programming (LP)

 ![Overview_Solvers](https://user-images.githubusercontent.com/83413252/135428779-d128aaaa-54ed-4ae1-a5b1-8e0fcc08c96e.png?raw=true "Lava features a growing suite of constrained optimization solvers")


### Taxonomy of Optimization Problems
More formally, the general form of a constrained optimization problem is:

$$
\displaystyle{\min_{x} \lbrace f(x) | g_i(x)	\leq  b,	h_i(x)	= c.\rbrace}
$$

Where $f(x)$ is the objective function to be optimized while $g(x)$ and $h(x)$ 
constrain the validity of $f(x)$ to regions in the state space satisfying the 
respective equality and inequality constraints. The vector $x$ can be
 continuous, discrete or a mixture of both. We can then construct the following 
 taxonomy of optimization problems according to  the characteristics of the 
 variable domain and of $f$, $g$, and $h$:

![image](https://user-images.githubusercontent.com/83413252/192852018-dbc08018-ddda-4571-8494-cd1fbfa8405f.png)

In the long run, lava-optimization aims to offer support to solve all of the problems in the figure with a neuromorphic backend. 

### OptimizationSolver and OptimizationProblem Classes

The figure below shows the general architecture of the library.  We harness the general definition of constraint optimization problems to create ``OptimizationProblem`` instances by composing  ``Constraints``, ``Variables``, and ``Cost`` classes which describe the characteristics of every subproblem class. Note that while a quadratic problem (QP) will be described by linear equality and inequality constraints with variables on the continuous domain and a quadratic function. A constraint satisfaction problem (CSP) will be described by discrete constraints, defined by variable subsets and a binary relation describing the mutually allowed values for such discrete variables and will have a constant cost function with the pure goal of satisfying constraints.

An API for every problem class can be created by inheriting from ``OptimizationSolver`` and composing particular flavors of ``Constraints``, ``Variables``, and ``Cost``. 

![image](https://user-images.githubusercontent.com/83413252/192851930-919035a7-122d-4a82-8032-f1acc6da717b.png)

The instance of an ``Optimization problem`` is the valid input for instantiating the generic ``OptimizationSolver`` class. In this way, the ``OptimizationSolver`` interface is left fixed and the ``OptimizationProblem`` allows the greatest flexibility for creating new APIs. Under the hood, the ``OptimizationSolver`` understands the composite structure of the ``OptimizationProblem`` and will in turn compose the required solver components and Lava processes.  

## Tutorials

### Quadratic Programming
- [Solving LASSO.](https://github.com/lava-nc/lava-optimization/blob/release/v0.2.0/tutorials/tutorial_01_solving_lasso.ipynb)

### Quadratic Unconstrained Binary Optimization
- [Solving Maximum Independent Set.](https://github.com/lava-nc/lava-optimization/blob/release/v0.2.0/tutorials/tutorial_02_solving_qubos.ipynb)

## Examples

### Solving QP problems 

```python
import numpy as np
from lava.lib.optimization.problems.problems import QP
from lava.lib.optimization.solvers.generic.solver import (
        SolverConfig,
        OptimizationSolver,
)

# Define QP problem
Q = np.array([[100, 0, 0], [0, 15, 0], [0, 0, 5]])
p = np.array([[1, 2, 1]]).T
A = -np.array([[1, 2, 2], [2, 100, 3]])
k = -np.array([[-50, 50]]).T

qp = QP(Q, p, A, k)

# Define hyper-parameters
hyperparameters = {
  "neuron_model": "qp-lp_pipg",
  "alpha_mantissa": 160,
  "alpha_exponent": -8,
  "beta_mantissa": 7,
  "beta_exponent": -10,
  "decay_schedule_parameters": (100, 100, 0),
  "growth_schedule_parameters": (3, 2),
}

# Solve using QPSolver
solver = OptimizationSolver(problem=qp)
config = SolverConfig(timeout=400, hyperparameters=hyperparameters, backend="Loihi2")
solver.solve(config=config)
```

### Solving QUBO
```python
import numpy as np
from lava.lib.optimization.problems.problems import QUBO
from lava.lib.optimization.solvers.generic.solver import (
        SolverConfig,
        OptimizationSolver,
)

# Define QUBO problem
q = np.array([[-5, 2, 4, 0],
              [ 2,-3, 1, 0],
              [ 4, 1,-8, 5],
              [ 0, 0, 5,-6]]))

qubo = QUBO(q)

# Solve using generic OptimizationSolver
solver = OptimizationSolver(problem=qubo)
config = SolverConfig(timeout=3000, target_cost=-50, backend="Loihi2")
solution = solver.solve(config=config)
```

## Getting Started

### Requirements
- Working installation of Lava, installed automatically with poetry below. [ For custom installs see Lava installation
tutorial.](https://github.com/lava-nc/lava/blob/main/tutorials/in_depth/tutorial01_installing_lava.ipynb)

### Installation

#### [Linux/MacOS]
```bash
cd $HOME
git clone git@github.com:lava-nc/lava-optimization.git
cd lava-optimization
curl -sSL https://install.python-poetry.org | python3 -
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
curl -sSL https://install.python-poetry.org | python3 -
poetry config virtualenvs.in-project true
poetry install
pytest
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

