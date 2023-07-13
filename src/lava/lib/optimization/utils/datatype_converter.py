# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import numpy as np


def convert_to_fp(matrix, mantissa_bits):
    """Function that returns the exponent, mantissa representation for
    floating point numbers that need to be represented on Loihi. A global exp
    is calculated for the matrices based on the max absolute value in the
    matrix. This is then used to calculate the manstissae in the matrix.

    Args:
        matrix (np.float): The input floating point matrix that needs to be
        converted
        mantissa_bits (int): number of bits that the mantissa uses for it's
        representation (Including sign)

    Returns:
        mat_fp_man (np.int): The matrix in
    """
    if np.linalg.norm(matrix) == 0:
        return matrix.astype(int), 0
    else:
        exp = np.ceil(np.log2(np.max(np.abs(matrix)))) - mantissa_bits + 1
        mat_fp_man = (matrix // 2**exp).astype(int)
        return mat_fp_man.astype(int), exp.astype(int)
