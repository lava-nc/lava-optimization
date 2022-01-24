# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty

DType = ty.Union[ty.List[int], ty.List[ty.Tuple[ty.Any]]]


class Variable:
    """An entity to which a value can be assigned.

    Parameters
    ----------
    name: Optional name for the variable.
    """

    def __init__(self, name: str = None):
        self.name = name
        self._value = None

    @property
    def value(self):
        """Variable's current value."""
        return self._value

    @value.setter
    def value(self, value):
        self._value = value


class DiscreteVariables:
    """A set of variables which can only be assigned discrete values.

    Parameters
    ----------
    domains: List of tuples with values that each variable can take or List
    of domain sizes for each corresponding (by index) variable.
    """

    def __init__(self, domains: DType = None):
        self._domains = domains

    @property
    def domain_sizes(self):
        """Number of elements on the domain of each discrete variable."""
        return [len(d) for d in self.domains]

    @property
    def domains(self):
        """List of tuples containing the values that each variable can take."""
        if self._domains is None:
            return None
        if type(self._domains[0]) is int:
            return [range(d) for d in self._domains]
        elif type(self._domains[0]) is tuple:
            return self._domains
        else:
            raise ValueError("Domains format not recognized.")

    @property
    def variable_set(self):
        """List of discrete variables each an instance of the Variable class."""
        return [Variable(name=str(n)) for n in range(self.num_variables)]

    @property
    def num_variables(self):
        """Number of variables in this set."""
        return len(self.domains)

    @domains.setter
    def domains(self, value: DType):
        self._domains = value


class ContinuousVariables:
    def __init__(self, bounds: ty.List[ty.Tuple] = None):
        self._bounds = bounds

    @property
    def variable_set(self):
        return [Variable(name=str(n)) for n in range(self.num_variables)]

    @property
    def num_variables(self):
        return len(self._bounds)

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, value: ty.List[ty.Tuple]):
        self._bounds = value

