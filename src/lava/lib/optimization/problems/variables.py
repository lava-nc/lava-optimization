# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

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
