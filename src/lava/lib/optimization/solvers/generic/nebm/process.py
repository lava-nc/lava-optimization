import numpy as np
import typing as ty
from numpy import typing as npty

from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var


class NEBM(AbstractProcess):
    """
    Non-equilibrium Boltzmann (NEBM) neuron model to solve QUBO problems.
    """

    def __init__(
        self,
        *,
        shape: ty.Tuple[int, ...],
        temperature: ty.Optional[ty.Union[int, npty.NDArray]] = 1,
        refract: ty.Optional[ty.Union[int, npty.NDArray]] = 1,
        refract_counter: ty.Optional[ty.Union[int, npty.NDArray]] = 0,
        init_value=0,
        init_state=0,
        neuron_model: str = 'nebm',
    ):
        """
        NEBM Process.

        Parameters
        ----------
        shape: Tuple
            Number of neurons. Default is (1,).
        temperature: ArrayLike
            Temperature of the system, defining the level of noise.
        refract : ArrayLike
            Minimum number of timesteps a neuron remains in a state after a
            transition.
        init_value : ArrayLike
            The spiking history with which the network is initialized
        init_state : ArrayLike
            The state of neurons with which the network is initialized
        """
        super().__init__(shape=shape)

        self.a_in = InPort(shape=shape)
        self.s_sig_out = OutPort(shape=shape)
        self.s_wta_out = OutPort(shape=shape)

        self.spk_hist = Var(
            shape=shape, init=(np.zeros(shape=shape) + init_value).astype(int)
        )
        #DELETE THIS AGAIN
        print("WRONG MODEL1")
        self.temperature = Var(shape=shape, init=int(temperature))
        self.refract = Var(shape=shape, init=refract)
        self.refract_counter = Var(shape=shape, init=refract_counter)

        # Initial state determined in DiscreteVariables
        self.state = Var(shape=shape, init=init_state.astype(int))

        @property
        def shape(self) -> ty.Tuple[int, ...]:
            return self.proc_params["shape"]


class NEBMSimulatedAnnealing(AbstractProcess):
    """
    Non-equilibrium Boltzmann (NEBM) neuron model to solve QUBO problems.
    """

    def __init__(
        self,
        *,
        shape: ty.Tuple[int, ...],
        cost_diagonal: npty.ArrayLike,
        max_temperature: int,
        refract_scaling: int,
        init_value=0,
        init_state=None,
        neuron_model: str,
    ):
        """
        SA Process.

        Parameters
        ----------
        shape: Tuple
            Number of neurons. Default is (1,).

        refract_scaling : ArrayLike
            After a neuron has switched its binary variable, it remains in a
            refractory state that prevents any variable switching for a
            number of time steps. This number of time steps is determined by
                rand(0, 255) >> refract_scaling
            Refract_scaling thus denotes the order of magnitude of timesteps a
            neuron remains in a state after a transition.
        init_value : ArrayLike
            The spiking history with which the network is initialized
        init_state : ArrayLike
            The state of neurons with which the network is initialized
        """

        self._validate_input(neuron_model)
        super().__init__(
            shape=shape,
            cost_diagonal=cost_diagonal,
            refract_scaling=refract_scaling,
        )

        self.a_in = InPort(shape=shape)
        self.delta_temperature_in = InPort(shape=shape)
        self.s_sig_out = OutPort(shape=shape)
        self.s_wta_out = OutPort(shape=shape)

        self.spk_hist = Var(
            shape=shape, init=(np.zeros(shape=shape) + init_value).astype(int)
        )

        self.temperature = Var(shape=shape, init=int(max_temperature))

        self.refract_counter = Var(
            shape=shape,
            init=0 + np.right_shift(
                np.random.randint(0, 2**8, size=shape), (refract_scaling or 0)
            ),
        )

        # Initial state determined in DiscreteVariables
        self.state = Var(
            shape=shape,
            init=init_state.astype(int)
            if init_state is not None
            else np.zeros(shape=shape, dtype=int),
        )

    def _validate_input(self, neuron_model: str) -> None:
        """Validates input. At the moment, it only checks that the user has
        chosen a supported neuron model is """

        if neuron_model == 'nebm-sa-refract':
            return
        elif neuron_model == 'nebm-sa':
            raise NotImplementedError(
                f"The model 'nebm-sa' has been deprecated. Instead, we "
                f"recommend switching to the new neuron model 'nebm-sa-refract'"
                f" for a better solver performance.")
        elif neuron_model == 'nebm-sa-balanced':
            raise NotImplementedError(
                f"The model 'nebm-sa-balanced' has been deprecated. Instead, we"
                f" recommend switching to the new neuron model "
                f"'nebm-sa-refract' for a better solver performance.")
        elif neuron_model == 'nebm-sa-refract-approx-unbalanced':
            raise NotImplementedError(
                f"Please note that the neuron model "
                f"'nebm-sa-refract_approx_unbalanced' has been removed. "
                f"Instead, we recommend switching to the new neuron model "
                f"'nebm-sa-refract' for a better solver performance.")
        elif neuron_model == 'nebm-sa-refract-approx':
            raise NotImplementedError(
                f"Please note that the neuron model "
                f"'nebm-sa-refract_approx_unbalanced' has been removed Instead,"
                f" we recommend switching to the neuron model 'nebm-sa-refract'"
                f" for a better solver performance.")
        else:
            raise ValueError(
                f"Please specify a correct neuron model as hyperparameter.")

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        return self.proc_params["shape"]
