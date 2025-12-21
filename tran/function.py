#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/4 10:57
# @Author  : ShengPengpeng
# @File    : function.py
# @Description :

import numpy as np


def eucliDist(A,B):
    return sum([(a - b)**2 for (a,b) in zip(A, B)])

def euclidean_distance(a, b):
    """计算两个点之间的欧式距离"""
    return np.sqrt(np.sum((a - b) ** 2))

def gaussian_distance_encoding(points, num_bins, sigma=1.0):
    """
    使用高斯函数编码欧式距离
    :param points: 形状为 (num_points, dim) 的张量，包含点的坐标
    :param num_bins: 编码的数量
    :param sigma: 高斯函数的标准差
    :return: 高斯编码矩阵，形状为 (num_points, num_points, num_bins)
    """
    num_points = points.shape[0]
    encoding = np.zeros((num_points, num_points, num_bins))

    # 计算每对点之间的欧式距离并编码
    for i in range(num_points):
        for j in range(num_points):
            dist = euclidean_distance(points[i], points[j])
            for k in range(num_bins):
                # 计算离散的距离
                bin_center = k - num_bins // 2
                # 使用高斯函数计算编码
                encoding[i, j, k] = np.exp(-0.5 * ((dist - bin_center) / sigma) ** 2)

    return encoding


def neighbors_to_adjacency_pos(neighbors, not_deleted, pos, raduis):
    """
    Create adjacency matrix from list of non-empty neighbors.

    Args:
        neighbors: Dict of neighbors per node.
        not_deleted: List of nodes, who did not get deleted in previous processing steps.
    """
    node_map = {n: i for i, n in enumerate(not_deleted)}

    n_nodes = len(not_deleted)
    # print('nnnnn', n_nodes)
    # print(raduis)
    new_adj_matrix = np.zeros((n_nodes, n_nodes))
    for ii in neighbors.keys():
        for jj in neighbors[ii]:
            i, j = node_map[ii], node_map[jj]
            new_adj_matrix[i, i] = 0 # diagonal if needed
            new_adj_matrix[i, j] = eucliDist(pos[i], pos[j]) * (raduis[i] + raduis[j]) + 1e-5
            new_adj_matrix[j, i] = eucliDist(pos[j], pos[i]) * (raduis[i] + raduis[j]) + 1e-5

    return new_adj_matrix


def neighbors_to_adjacency_at(neighbors, not_deleted, node_type, pos):
    """
    Create adjacency matrix from list of non-empty neighbors.

    Args:
        neighbors: Dict of neighbors per node.
        not_deleted: List of nodes, who did not get deleted in previous processing steps.
    """
    node_map = {n: i for i, n in enumerate(not_deleted)}

    n_nodes = len(not_deleted)
    # print('bbbb', n_nodes)
    node_type = node_type.numpy()

    new_adj_matrix = np.zeros((n_nodes, n_nodes))
    for ii in neighbors.keys():
        for jj in neighbors[ii]:
            i, j = node_map[ii], node_map[jj]
            # node_type.dtype = np.uint8
            # print(node_type.shape)

            type_i = ''.join(str(i) for i in node_type[i].astype(int))
            type_j = ''.join(str(i) for i in node_type[j].astype(int))

            new_adj_matrix[i, i] = 0
            new_adj_matrix[i, j] = (int(type_i, 2) + int(type_j, 2) - 1) * (eucliDist(pos[i], pos[j]))
            new_adj_matrix[j, i] = (int(type_j, 2) + int(type_i, 2) - 1) * (eucliDist(pos[i], pos[j]))

    return new_adj_matrix

def neighbors_to_adjacency_attr(neighbors, not_deleted, node_type):
    """
    Create adjacency matrix from list of non-empty neighbors.

    Args:
        neighbors: Dict of neighbors per node.
        not_deleted: List of nodes, who did not get deleted in previous processing steps.
    """
    node_map = {n: i for i, n in enumerate(not_deleted)}

    n_nodes = len(not_deleted)
    # print('bbbb', n_nodes)
    # node_type = node_type.numpy()

    new_adj_matrix = np.zeros((n_nodes, n_nodes))
    for ii in neighbors.keys():
        for jj in neighbors[ii]:
            i, j = node_map[ii], node_map[jj]
            # node_type.dtype = np.uint8
            # print(node_type.shape)

            type_i = ''.join(str(i) for i in node_type[i].astype(int))
            type_j = ''.join(str(i) for i in node_type[j].astype(int))

            new_adj_matrix[i, i] = 0
            new_adj_matrix[i, j] = int(type_i, 2) + int(type_j, 2) - 1
            new_adj_matrix[j, i] = int(type_j, 2) + int(type_i, 2) - 1

    return new_adj_matrix


def neighbors_to_adjacency_attr_train(neighbors, not_deleted, node_type):
    """
    Create adjacency matrix from list of non-empty neighbors.

    Args:
        neighbors: Dict of neighbors per node.
        not_deleted: List of nodes, who did not get deleted in previous processing steps.
    """
    node_map = {n: i for i, n in enumerate(not_deleted)}

    n_nodes = len(not_deleted)
    # print('bbbb', n_nodes)
    node_type = node_type.numpy()

    new_adj_matrix = np.zeros((n_nodes, n_nodes))
    for ii in neighbors.keys():
        for jj in neighbors[ii]:
            i, j = node_map[ii], node_map[jj]
            # node_type.dtype = np.uint8
            # print(node_type.shape)

            type_i = ''.join(str(i) for i in node_type[i].astype(int))
            type_j = ''.join(str(i) for i in node_type[j].astype(int))

            new_adj_matrix[i, i] = 0
            new_adj_matrix[i, j] = int(type_i, 2) + int(type_j, 2) - 1
            new_adj_matrix[j, i] = int(type_j, 2) + int(type_i, 2) - 1

    return new_adj_matrix


def adjacency_to_neighbors(adj_matrix):
    """
    Create list of non-empty neighbors from adjacancy matrix.
    Args:
        adj_matrix: adjacency matrix (N x N)
    """
    # Remove diagonal to avoid self-neighbors.
    a, b = np.where(adj_matrix - np.eye(adj_matrix.shape[0]) == 1)
    neigh = dict()
    for _a, _b in zip(a, b):
        if _a not in neigh:
            neigh[_a] = set()
        neigh[_a].add(_b)
    return neigh
