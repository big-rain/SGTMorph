#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/26 20:07
# @Author  : ShengPengpeng
# @File    : augment_node_position.py
# @Description :

import scipy.sparse as sp
import math
import numbers
import random
from itertools import repeat
from typing import Sequence, Union, Tuple
from tran.function import neighbors_to_adjacency_pos, adjacency_to_neighbors

import numpy as np
import torch
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.spatial.transform import Rotation as R


def rotate_graph(pos_matrix):
    ''' Randomly rotate graph in xyz-direction.

    Args:
        pos_matrix: Matrix with xyz-node positions (N x 3).
        axis: Axis around which to rotate. Defaults to `None`,
            in which case no rotation is performed.
    '''

    axis = random.choice([0, 1, 2])

    rotation_matrix = R.random().as_matrix()

    if axis == 0: # x
        rotation_matrix[0, 1] = 0
        rotation_matrix[0, 2] = 0
        rotation_matrix[0, 0] = 1
        rotation_matrix[1, 0] = 0
        rotation_matrix[2, 0] = 0
    elif axis == 1:  # y
        rotation_matrix[0, 1] = 0
        rotation_matrix[1, 0] = 0
        rotation_matrix[1, 1] = 1
        rotation_matrix[1, 2] = 0
        rotation_matrix[2, 1] = 0
    elif axis == 2:  # z
        rotation_matrix[0, 2] = 0
        rotation_matrix[1, 2] = 0
        rotation_matrix[2, 2] = 1
        rotation_matrix[2, 0] = 0
        rotation_matrix[2, 1] = 0

    rot_pos_matrix = pos_matrix @ rotation_matrix
    return rot_pos_matrix


def random_filp(pos, axis=None):
    if random.random() < 0.5:
        filp_pos = pos.clone()
        pos[..., axis] = -filp_pos[..., axis]
        return pos


# 随机抖动
def jitter_node_pos(node_positions, scale=0.1):
    """
    Randomly jitter nodes in xyz-direction.

    Args:
        node_positions: Matrix with xyz-node positions (N x 3).
        scale: Scale factor for jittering.
    """
    (n, dim), t = node_positions.size(), scale
    if isinstance(t, numbers.Number):
        t = list(repeat(t, times=n))
    assert len(t) == n
    pos = node_positions.clone()
    ts = []
    for d in range(n):
        ts.append(pos.new_empty(dim).uniform_(-abs(t[d]), abs(t[d])))

    for i in range(n):
        # if random.randint(0, 1):
        #     pos[i] = node_positions[i] + ts[i]
        pos[i] = node_positions[i] + ts[i]
    # else:
    #     pos[i] = node_positions[i]
    return pos


# def jitter_node(node_positions, scale=0.1):
#
#     return node_positions + (torch.randn(*node_positions.shape).numpy() * scale)


def translate_soma_pos(node_positions, scale=1):
    """
    Randomly translate the position of the entire grpah.

    Args:
        node_positions: Matrix with xyz-node positions (N x 3).
        scale: Scale factor for jittering.
    """
    new_node_features = node_positions.clone()
    # new_node_features = node_positions.copy()
    jitter = torch.randn(3).numpy() * scale
    new_node_features[:, :3] += jitter
    return new_node_features




class Augment_node_position:
    def __init__(self, trans_scale, jitter_scale, rota=True):
        self.trans_scale = trans_scale
        self.scale = jitter_scale
        self.rota = rota

    def __call__(self, data):

        adj = to_scipy_sparse_matrix(data.edge_index)
        neighbors = adjacency_to_neighbors(adj_matrix=adj)

        features = data.x

        pos = features[:, :3]
        node_radius = features[:, 3]


        if self.rota:
            rot_pos = rotate_graph(pos)
        else:
            rot_pos = pos
        jitter_pos = jitter_node_pos(rot_pos, scale=self.scale)
        # jitter_pos = jitter_node(rot_pos, scale=self.scale)
        #
        translate_pos = translate_soma_pos(jitter_pos, scale=self.trans_scale)
        # translate_pos = tensor.cpu().numpy()

        features[:, :3] = translate_pos

        # 节点位置
        translate_pos = (translate_pos - torch.min(translate_pos))/(torch.max(translate_pos) - torch.min(translate_pos))


        subsampling_adj = neighbors_to_adjacency_pos(neighbors, neighbors, translate_pos, node_radius)
        edge_index_temp = sp.coo_matrix(subsampling_adj)
        weight_values = edge_index_temp.data
        edge_weight = torch.Tensor(weight_values)


        data.weight = edge_weight
        # data.index  边链接关系 不变
        data.pos = translate_pos
        # y  图类别 不变
        data.x = features
        # pe  随机游走位置编码 不变
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}({self.degree})'


