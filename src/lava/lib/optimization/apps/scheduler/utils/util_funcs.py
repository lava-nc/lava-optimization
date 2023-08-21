# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


def is_visible(satellite_y, request_y, view_height):
    """ Return whether the request is visible to the satellite. """
    return satellite_y <= request_y <= satellite_y + view_height


def is_same_satellite(n1, n2):
    """ Return whether nodes n1 and n2 reference the same satellite. """
    return n1[1] == n2[1]


def is_movable(n1, n2, turn_rate):
    """ Return whether a satellite can turn from n1 to n2 without
    exceeding turn_rate. """
    xdist = abs(n1[2] - n2[2])
    ydist = abs(n1[3] - n2[3])
    return turn_rate * xdist >= ydist


def is_same_request(n1, n2):
    """ Return whether n1 and n2 reference the same request. """
    return (n1[2] == n2[2]) and (n1[3] == n2[3])


def is_feasible(n1, n2, turn_rate):
    """ Return whether it is feasible to traverse from n1 to n2. """
    return not is_same_request(n1, n2) and \
        (not is_same_satellite(n1, n2) or is_movable(n1, n2, turn_rate))


def is_same_node(n1, n2):
    """ Return whether n1 and n2 are the same node. """
    return n1[0] == n2[0]
