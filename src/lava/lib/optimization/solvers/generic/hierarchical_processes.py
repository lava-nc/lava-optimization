# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
from lava.lib.optimization.problems.problems import OptimizationProblem
from numpy import typing as npty

import numpy as np
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var


class ContinuousVariablesProcess(AbstractProcess):
    """Process which implementation holds the evolution of continuous
    variables on the solver of an optimization problem."""

    def __init__(
        self,
        shape: ty.Tuple[int, ...],
        problem: OptimizationProblem,
        backend,
        hyperparameters: ty.Dict[str, ty.Union[int, npty.ArrayLike]] = None,
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
    ) -> None:
        super().__init__(
            shape=shape,
            name=name,
            problem=problem,
            log_config=log_config,
        )

        self.num_variables = np.prod(shape)
        self.backend = backend
        self.hyperparameters = hyperparameters
        self.problem = problem
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.variable_assignment = Var(shape=shape)
        self.cost = OutPort(shape=shape)


class DiscreteVariablesProcess(AbstractProcess):
    r"""Process which implementation holds the evolution of discrete variables
    on the solver of an optimization problem.

    Attributes
    ----------
    a_in: InPort
        The addition of all inputs (per dynamical system) at this timestep
        will be received by this port.
    s_out: OutPort
        The payload to be exchanged between the underlying dynamical systems
        when these fire.
    local_cost: OutPort
        The cost components per dynamical system underlying these
        variables, i.e., c_i = sum_j{Q_{ij} \cdot x_i}  will be sent through
        this port. The cost integrator will then complete the cost computation
        by adding all contributions, i.e., x^T \cdot Q \cdot x = sum_i{c_i}.
    variable_assignment: Var
        Holds the current value assigned to the variables by
        the solver network.
    """

    def __init__(
        self,
        shape: ty.Tuple[int, ...],
        cost_diagonal: npty.ArrayLike = None,
        cost_off_diagonal: npty.ArrayLike = None,
        hyperparameters: ty.Dict[str, ty.Union[int, npty.ArrayLike]] = None,
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
    ) -> None:
        """
        Parameters
        ----------
        shape: tuple
            A tuple of the form (number of variables, domain size).
        cost_diagonal: npty.ArrayLike
            The diagonal of the coefficient of the quadratic term on the cost
            function.
        cost_off_diagonal: npty.ArrayLike
            The off-diagonal of the coefficient of the quadratic term on the
            cost function.
        hyperparameters: dict, optional
        name: str, optional
            Name of the Process. Default is 'Process_ID', where ID is an integer
            value that is determined automatically.
        log_config: LogConfig, optional
            Configuration options for logging.z"""
        super().__init__(
            shape=shape,
            cost_diagonal=cost_diagonal,
            cost_off_diagonal=cost_off_diagonal,
            name=name,
            log_config=log_config,
        )
        self.num_variables = np.prod(shape)
        self.hyperparameters = hyperparameters
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.variable_assignment = Var(shape=shape)
        self.cost_diagonal = cost_diagonal
        self.local_cost = OutPort(shape=shape)


class CostConvergenceChecker(AbstractProcess):
    """Process that continuously monitors cost convergence.

    Attributes
    ----------
    cost_components: InPort
        Additive contributions to the total cost.
    cost_out_last_bytes: OutPort
        Notifies the next process about the detection of a better cost.
        Messages the last 3 byte of the new best cost.
        Total cost = cost_out_first_byte << 24 + cost_out_last_bytes.
    cost_out_first_byte: OutPort
        Notifies the next process about the detection of a better cost.
        Messages the first byte of the new best cost.
    cost_min_last_bytes
        Current minimum cost, i.e., the lowest reported cost so far.
        Saves the last 3 bytes.
        cost_min = cost_min_first_byte << 24 + cost_min_last_bytes
    cost_min_first_byte
        Current minimum cost, i.e., the lowest reported cost so far.
        Saves the first byte.
    """

    def __init__(
        self,
        shape: ty.Tuple[int, ...],
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
    ) -> None:
        """
        Constructor for CostConvergenceChecker class.

        Parameters
        ----------
        shape: tuple
            The expected shape of the input cost components.
        name: str, optional
            Name of the Process. Default is 'Process_ID', where ID is an
            integer value that is determined automatically.
        log_config: LogConfig, optional
            Configuration options for logging.
        """
        super().__init__(shape=shape, name=name, log_config=log_config)
        self.shape = shape
        self.cost_min_first_byte = Var(shape=(1,))
        self.cost_min_last_bytes = Var(shape=(1,))
        self.cost_first_byte = Var(shape=(1,))
        self.cost_last_bytes = Var(shape=(1,))
        self.cost_components = InPort(shape=shape)
        self.cost_out_last_bytes = OutPort(shape=(1,))
        self.cost_out_first_byte = OutPort(shape=(1,))


class SatConvergenceChecker(AbstractProcess):
    """Process that continuously monitors satisfiability convergence."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.satisfaction = None
        raise NotImplementedError


class AugmentedTermsProcess(AbstractProcess):
    """Process implementing cost coefficients as synapses."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError


class ContinuousConstraintsProcess(AbstractProcess):
    """Process implementing continuous constraints via neurons and synapses."""

    def __init__(
        self,
        shape_in: ty.Tuple[int, ...],
        shape_out: ty.Tuple[int, ...],
        problem: OptimizationProblem,
        backend,
        hyperparameters: ty.Dict[str, ty.Union[int, npty.ArrayLike]] = None,
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
    ) -> None:
        super().__init__(
            shape_in=shape_in,
            shape_out=shape_out,
            problem=problem,
            name=name,
            log_config=log_config,
        )

        self.num_constraints = np.prod(problem.constraint_biases_eq.shape)
        self.problem = problem
        self.backend = backend
        self.hyperparameters = hyperparameters
        self.a_in = InPort(shape=shape_in)
        self.s_out = OutPort(shape=shape_out)
        self.constraint_assignment = Var(
            shape=problem.constraint_biases_eq.shape
        )


class DiscreteConstraintsProcess(AbstractProcess):
    """Process implementing discrete constraints via synapses."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError


class MixedConstraintsProcess(AbstractProcess):
    """Process implementing continuous constraints via neurons and synapses."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError


class StochasticIntegrateAndFire(AbstractProcess):
    r"""Event-driven stochastic discrete dynamical system with two outputs.

    The main output is intended as input to other dynamical systems on
    the network, whilst the second output is to transfer local information to be
    integrated by an auxiliary dynamical system or circuit.

    Attributes
    ----------
    added_input: InPort
        The addition of all inputs (per dynamical system) at this
        timestep will be received by this port.
    replace_assignment: InPort
        Todo: deprecate
    messages: OutPort
        The payload to be sent to other dynamical systems when firing.
    local_cost: OutPort
        the cost component corresponding to this dynamical system, i.e.,
        c_i = sum_j{Q_{ij} \cdot x_i}  will be sent through this port. The cost
        integrator will then complete the cost computation  by adding all
        contributions, i.e., x^T \cdot Q \cdot x = sum_i{c_i}.

    """

    # This variable defines the time steps for a single algorithmic step
    # Used by the SolutionReadOut to extract variables from the spk_hist
    time_steps_per_algorithmic_step = 1

    def __init__(
        self,
        *,
        step_size: npty.ArrayLike,
        shape: ty.Tuple[int, ...] = (1,),
        init_state: npty.ArrayLike = 0,
        noise_amplitude: npty.ArrayLike = 1,
        noise_precision: npty.ArrayLike = 8,
        sustained_on_tau: npty.ArrayLike = -3,
        threshold: npty.ArrayLike = 10,
        cost_diagonal: npty.ArrayLike = 0,
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
        init_value: npty.ArrayLike = 0,
    ) -> None:
        """
        Parameters
        ----------
        shape: tuple
            The shape of the set of dynamical systems to be created.
        init_state: npty.ArrayLike, optional
            The starting value of the state variable.
        step_size: npty.ArrayLike, optional
            a value to be added to the state variable at each timestep.
        noise_amplitude: npty.ArrayLike, optional
            The width/range for the stochastic perturbation to the state
            variable. A random number within this range will be added to the
            state variable at each timestep.
        steps_to_fire: npty.ArrayLike, optional
            After how many timesteps would the dynamical system fire and reset
            without stochastic perturbation. Note that if noise_amplitude > 0,
            the system will stochastically deviate from this value.
        cost_diagonal: npty.ArrayLike, optional
            The linear coefficients on the cost function of the optimization
            problem where this system will be used.
        name: str, optional
            Name of the Process. Default is 'Process_ID', where ID is an
            integer value that is determined automatically.
        log_config: LogConfig, optional
            Configuration options for logging.
        init_value: int, optional
        """
        super().__init__(
            shape=shape,
            init_state=init_state,
            step_size=step_size,
            noise_amplitude=noise_amplitude,
            noise_precision=noise_precision,
            sustained_on_tau=sustained_on_tau,
            threshold=threshold,
            cost_diagonal=cost_diagonal,
            name=name,
            log_config=log_config,
            init_value=init_value,
        )
        self.added_input = InPort(shape=shape)
        self.messages = OutPort(shape=shape)
        self.local_cost = OutPort(shape=shape)

        self.integration = Var(shape=shape, init=0)
        self.step_size = Var(shape=shape, init=step_size)
        self.state = Var(shape=shape, init=init_state)
        self.noise_amplitude = Var(shape=shape, init=noise_amplitude)
        self.noise_precision = Var(shape=shape, init=noise_precision)
        self.sustained_on_tau = Var(shape=(1,), init=sustained_on_tau)
        self.threshold = Var(shape=shape, init=threshold)
        self.prev_assignment = Var(shape=shape, init=False)
        self.cost_diagonal = Var(shape=shape, init=cost_diagonal)
        self.assignment = Var(shape=shape, init=False)
        self.min_cost = Var(shape=shape, init=False)


class NEBMAbstract(AbstractProcess):
    r"""Event-driven stochastic discrete dynamical system with two outputs.

    The main output is intended as input to other dynamical systems on
    the network, whilst the second output is to transfer local information to be
    integrated by an auxiliary dynamical system or circuit.

    Attributes
    ----------
    added_input: InPort
        The addition of all inputs (per dynamical system) at this
        timestep will be received by this port.
    messages: OutPort
        The payload to be sent to other dynamical systems when firing.
    local_cost: OutPort
        the cost component corresponding to this dynamical system, i.e.,
        c_i = sum_j{Q_{ij} \cdot x_i}  will be sent through this port. The cost
        integrator will then complete the cost computation  by adding all
        contributions, i.e., x^T \cdot Q \cdot x = sum_i{c_i}.

    """

    # This variable defines the time steps for a single algorithmic step
    # Used by the SolutionReadOut to extract variables from the spk_hist
    time_steps_per_algorithmic_step = 1

    def __init__(
        self,
        *,
        temperature: npty.ArrayLike,
        refract: ty.Optional[ty.Union[int, npty.NDArray]],
        refract_counter: ty.Optional[ty.Union[int, npty.NDArray]],
        shape: ty.Tuple[int, ...] = (1,),
        init_state: npty.ArrayLike = 0,
        input_duration: npty.ArrayLike = 6,
        min_state: npty.ArrayLike = 1000,
        min_integration: npty.ArrayLike = -1000,
        cost_diagonal: npty.ArrayLike = 0,
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
        init_value: npty.ArrayLike = 0,
    ) -> None:
        """

        Parameters
        ----------
        shape: tuple
            The shape of the set of dynamical systems to be created.
        init_state: npty.ArrayLike, optional
            The starting value of the state variable.
        temperature: npty.ArrayLike, optional
            the temperature of the systems, defining the level of noise.
        input_duration: npty.ArrayLike, optional
            Number of timesteps by which each input should be preserved.
        min_state: npty.ArrayLike, optional
            The minimum value for the state variable. The state variable will be
            truncated at this value if updating results in a lower value.
        min_integration: npty.ArrayLike, optional
            The minimum value for the total input (addition of all valid inputs
            at a given timestep). The total input value will be truncated at
            this value if adding current and preserved inputs results in a lower
            value.
        refract: npty.ArrayLike, optional
            Number of timesteps to wait after firing and reset before resuming
            updating.
        refract_counter: npty.ArrayLike, optional
            Number of timesteps to initially suppress a unit firing.
        cost_diagonal: npty.ArrayLike, optional
            The linear coefficients on the cost function of the optimization
            problem where this system will be used.
        name: str, optional
            Name of the Process. Default is 'Process_ID', where ID is an
            integer value that is determined automatically.
        log_config: LogConfig, optional
            Configuration options for logging.
        init_value: int, optional
        """
        super().__init__(
            shape=shape,
            init_state=init_state,
            temperature=temperature,
            refract=refract,
            refract_counter=refract_counter,
            input_duration=input_duration,
            min_state=min_state,
            min_integration=min_integration,
            cost_diagonal=cost_diagonal,
            name=name,
            log_config=log_config,
            init_value=init_value,
        )
        self.added_input = InPort(shape=shape)
        self.messages = OutPort(shape=shape)
        self.local_cost = OutPort(shape=shape)
        self.integration = Var(shape=shape, init=0)
        self.temperature = Var(shape=shape, init=temperature)
        self.refract = Var(shape=shape, init=refract)
        self.refract_counter = Var(shape=shape, init=refract_counter)
        self.state = Var(shape=shape, init=init_state)
        self.input_duration = Var(shape=shape, init=input_duration)
        self.min_state = Var(shape=shape, init=min_state)
        self.firing = Var(shape=shape, init=init_value)
        self.prev_assignment = Var(shape=shape, init=False)
        self.cost_diagonal = Var(shape=shape, init=cost_diagonal)
        self.assignment = Var(shape=shape, init=False)
        self.min_cost = Var(shape=shape, init=False)


class SimulatedAnnealingLocalAbstract(AbstractProcess):
    r"""Event-driven stochastic discrete dynamical system with two outputs.

    The main output is intended as input to other dynamical systems on
    the network, whilst the second output is to transfer local information to be
    integrated by an auxiliary dynamical system or circuit.

    Attributes
    ----------
    added_input: InPort
        The addition of all inputs (per dynamical system) at this
        timestep will be received by this port.
    messages: OutPort
        The payload to be sent to other dynamical systems when firing.
    local_cost: OutPort
        the cost component corresponding to this dynamical system, i.e.,
        c_i = sum_j{Q_{ij} \cdot x_i}  will be sent through this port. The cost
        integrator will then complete the cost computation  by adding all
        contributions, i.e., x^T \cdot Q \cdot x = sum_i{c_i}.

    """

    # This variable defines the time steps for a single algorithmic step
    # Used by the SolutionReadOut to extract variables from the spk_hist
    time_steps_per_algorithmic_step = 1

    def __init__(
        self,
        *,
        cost_diagonal: npty.ArrayLike,
        max_temperature: int,
        min_temperature: int,
        delta_temperature: int,
        exp_temperature: int,
        steps_per_temperature: int,
        refract_scaling: int,
        refract_seed: int,
        annealing_schedule: str,
        shape: ty.Tuple[int, ...],
        init_state: npty.ArrayLike,
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
        init_value: npty.ArrayLike,
    ) -> None:
        """

        Parameters
        ----------
        shape: tuple
            The shape of the set of dynamical systems to be created.
        init_state: npty.ArrayLike, optional
            The starting value of the state variable.
        max_temperature: npty.ArrayLike, int
            The maximum/initial temperature of the systems. The temperature,
            defines the level of noise.
        min_temperature: npty.ArrayLike, int
            The minimum temperature of the systems. Once reached during
            annealing, the temperature will be reset to max_temperature.
        delta_temperature: npty.ArrayLike, int
            Defines the temperature annealing. For a linear annealing
            schedule, the temperature is changed in each annnealing step
            according to
                T -= delta_temperature
            In the geometrich annealin schedule, according to
                T *= delta_temperature * 2**(-exp_temperature)
        exp_temperature: npty.ArrayLike, int
            Defines the temperature annealing, together with delta_temperature.
            Must only be provided if annealing_schedule=='geometric'.
        steps_per_temperature: npty.ArrayLike
            Steps that the Boltzmann machine runs for before the next
            temperature annealing step takes place.
        annealing_schedule: npty.ArrayLike(str), str
            'linear' or 'geometric'.
        min_integration: npty.ArrayLike, optional
            The minimum value for the total input (addition of all valid inputs
            at a given timestep). The total input value will be truncated at
            this value if adding current and preserved inputs results in a lower
            value.
        refract_scaling : ArrayLike
            After a neuron has switched its binary variable, it remains in a
            refractory state that prevents any variable switching for a
            number of time steps. This number of time steps is determined by
                rand(0, 255) >> refract_scaling
            Refract_scaling thus denotes the order of magnitude of timesteps a
            neuron remains in a state after a transition.
        refract_seed : int
            Random seed to initialize the refractory periods. Allows
            repeatability.
        cost_diagonal: npty.ArrayLike, optional
            The linear coefficients on the cost function of the optimization
            problem where this system will be used.
        name: str, optional
            Name of the Process. Default is 'Process_ID', where ID is an
            integer value that is determined automatically.
        log_config: LogConfig, optional
            Configuration options for logging.
        init_value: int, optional
        """
        super().__init__(
            shape=shape,
            init_state=init_state,
            max_temperature=max_temperature,
            min_temperature=min_temperature,
            delta_temperature=delta_temperature,
            exp_temperature=exp_temperature,
            steps_per_temperature=steps_per_temperature,
            refract_scaling=refract_scaling,
            refract_seed=refract_seed,
            cost_diagonal=cost_diagonal,
            name=name,
            log_config=log_config,
            init_value=init_value,
            annealing_schedule=annealing_schedule,
        )
        self.added_input = InPort(shape=shape)
        self.messages = OutPort(shape=shape)
        self.local_cost = OutPort(shape=shape)

        self.integration = Var(shape=shape, init=0)
        self.max_temperature = Var(shape=shape, init=max_temperature)
        self.temperature = Var(shape=(1,), init=max_temperature)
        self.min_temperature = Var(shape=shape, init=min_temperature)
        self.delta_temperature = Var(shape=shape, init=delta_temperature)
        self.exp_temperature = Var(shape=shape, init=exp_temperature)
        self.steps_per_temperature = Var(
            shape=shape, init=steps_per_temperature
        )
        self.state = Var(shape=shape, init=init_state)
        self.firing = Var(shape=shape, init=init_value)
        self.prev_assignment = Var(shape=shape, init=False)
        self.cost_diagonal = Var(shape=shape, init=cost_diagonal)
        self.assignment = Var(shape=shape, init=False)
        self.min_cost = Var(shape=shape, init=False)


class SimulatedAnnealingAbstract(AbstractProcess):
    r"""Event-driven stochastic discrete dynamical system with two outputs.

    The main output is intended as input to other dynamical systems on
    the network, whilst the second output is to transfer local information to be
    integrated by an auxiliary dynamical system or circuit.

    Attributes
    ----------
    added_input: InPort
        The addition of all inputs (per dynamical system) at this
        timestep will be received by this port.
    messages: OutPort
        The payload to be sent to other dynamical systems when firing.
    local_cost: OutPort
        the cost component corresponding to this dynamical system, i.e.,
        c_i = sum_j{Q_{ij} \cdot x_i}  will be sent through this port. The cost
        integrator will then complete the cost computation  by adding all
        contributions, i.e., x^T \cdot Q \cdot x = sum_i{c_i}.

    """

    # This variable defines the time steps for a single algorithmic step
    # Used by the SolutionReadOut to extract variables from the spk_hist
    time_steps_per_algorithmic_step = 2

    def __init__(
        self,
        *,
        cost_diagonal: npty.ArrayLike,
        cost_off_diagonal: npty.ArrayLike,
        max_temperature: int,
        min_temperature: int,
        delta_temperature: int,
        exp_temperature: int,
        steps_per_temperature: int,
        shape: ty.Tuple[int, ...],
        init_state: npty.ArrayLike,
        init_value: npty.ArrayLike,
        annealing_schedule: str,
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
    ) -> None:
        """

        Parameters
        ----------
        shape: tuple
            The shape of the set of dynamical systems to be created.
        init_state: npty.ArrayLike, optional
            The starting value of the state variable.
        temperature: npty.ArrayLike, optional
            the temperature of the systems, defining the level of noise.
        refractory_period: npty.ArrayLike, optional
            Number of timesteps to wait after firing and reset before resuming
            updating.
        cost_diagonal: npty.ArrayLike, optional
            The linear coefficients on the cost function of the optimization
            problem where this system will be used.
        name: str, optional
            Name of the Process. Default is 'Process_ID', where ID is an
            integer value that is determined automatically.
        log_config: LogConfig, optional
            Configuration options for logging.
        init_value: int, optional
        """
        super().__init__(
            shape=shape,
            init_state=init_state,
            max_temperature=max_temperature,
            min_temperature=min_temperature,
            delta_temperature=delta_temperature,
            exp_temperature=exp_temperature,
            steps_per_temperature=steps_per_temperature,
            cost_diagonal=cost_diagonal,
            cost_off_diagonal=cost_off_diagonal,
            name=name,
            log_config=log_config,
            init_value=init_value,
            annealing_schedule=annealing_schedule,
        )
        self.added_input = InPort(shape=shape)
        self.messages = OutPort(shape=shape)
        self.local_cost = OutPort(shape=shape)

        self.integration = Var(shape=shape, init=0)
        self.max_temperature = Var(shape=shape, init=max_temperature)
        self.temperature = Var(shape=(1,), init=max_temperature)
        self.min_temperature = Var(shape=shape, init=min_temperature)
        self.delta_temperature = Var(shape=shape, init=delta_temperature)
        self.exp_temperature = Var(shape=shape, init=exp_temperature)
        self.steps_per_temperature = Var(
            shape=shape, init=steps_per_temperature
        )
        self.state = Var(shape=shape, init=init_state)
        self.firing = Var(shape=shape, init=init_value)
        self.prev_assignment = Var(shape=shape, init=False)
        self.cost_diagonal = Var(shape=shape, init=cost_diagonal)
        self.assignment = Var(shape=shape, init=False)
        self.min_cost = Var(shape=shape, init=False)
