#  Copyright (C) 2023 Battelle Memorial Institute
#  SPDX-License-Identifier: BSD-2-Clause
#  See: https://spdx.org/licenses/

import numpy as np


def sign_extend_24bit(x):
    """
    Sign extends a signed 24-bit numpy array into a signed 32-bit array.
    """
    x = x.astype(np.int32)
    mask = (np.right_shift(x, 23) > 0) * np.array([0xFF000000], dtype=np.int32)
    return np.bitwise_or(x, mask)


def apply_activation(voltage, threshold):
    """
    Applies a soft-threshold activation
    """
    return np.maximum(np.abs(voltage) - threshold, 0) * np.sign(voltage)


def get_1_layer_weights(dictionary, tau):
    """
    Returns the floating-pt weights for 1 Layer LCA.
    """
    return ((dictionary @ -dictionary.T) + np.eye(dictionary.shape[0])) * tau


def get_1_layer_bias(dictionary, tau, input):
    """
    Returns the floating-pt bias for 1 Layer LCA.
    """
    return (input @ dictionary.T) * tau


def get_fixed_pt_scale(sparse_coding):
    """
    Returns the optimal scale factor given a known or estimated sparse coding.
    The scale is the largest power of 2 such that the sparse_coding * scale does
    not exceed 2**24
    """
    return 2 ** (24 - np.ceil(np.log2(np.max(np.abs(sparse_coding)))))
