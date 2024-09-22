#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/26 22:58
# @Author  : ShengPengpeng
# @File    : Neuron_utils.py
# @Description :

# from torch_geometric.datasets import ZINC

from typing import Callable, List, Optional
import torch
import pandas as pd

import numpy as np

import math
def eucliDist(A,B):
    return math.sqrt(sum([(a - b)**2 for (a,b) in zip(A,B)]))


def neighbors_to_adjacency_1(neighbors, not_deleted):
    """
    Create adjacency matrix from list of non-empty neighbors.

    Args:
        neighbors: Dict of neighbors per node.
        not_deleted: List of nodes, who did not get deleted in previous processing steps.
    """
    node_map = {n: i for i, n in enumerate(not_deleted)}

    n_nodes = len(not_deleted)

    new_adj_matrix = np.zeros((n_nodes, n_nodes))
    for ii in neighbors.keys():
        for jj in neighbors[ii]:
            i, j = node_map[ii], node_map[jj]
            new_adj_matrix[i, i] = True  # diagonal if needed
            new_adj_matrix[i, j] = True
            new_adj_matrix[j, i] = True

    return new_adj_matrix


def neighbors_to_adjacency(neighbors, not_deleted, pos):
    """
    Create adjacency matrix from list of non-empty neighbors.

    Args:
        neighbors: Dict of neighbors per node.
        not_deleted: List of nodes, who did not get deleted in previous processing steps.
    """
    node_map = {n: i for i, n in enumerate(not_deleted)}

    # n_nodes = len(not_deleted)

    new_adj_matrix = np.zeros((1000, 1000))
    for ii in neighbors.keys():
        for jj in neighbors[ii]:
            i, j = node_map[ii], node_map[jj]
            new_adj_matrix[i, i] = eucliDist(pos[i], pos[i])  # diagonal if needed
            new_adj_matrix[i, j] = eucliDist(pos[i], pos[j])
            new_adj_matrix[j, i] = eucliDist(pos[j], pos[i])

    return new_adj_matrix

def subsample_graph(neighbors=None, not_deleted=None, keep_nodes=200, protected=[0]):
    """
    Subsample graph.

    Args:
        neighbors: dict of neighbors per node
        not_deleted: list of nodes, who did not get deleted in previous processing steps
        keep_nodes: number of nodes to keep in graph
        protected: nodes to be excluded from subsampling
    """
    if neighbors is not None:
        k_nodes = len(neighbors)
    else:
        raise ValueError('neighbors must be provided')

    # protect soma node from being removed
    protected = set(protected)

    # indices as set in random order
    perm = torch.randperm(k_nodes).tolist()
    all_indices = np.array(list(not_deleted))[perm].tolist()
    deleted = set()

    while len(deleted) < k_nodes - keep_nodes:

        while True:
            if len(all_indices) == 0:
                assert len(not_deleted) > keep_nodes, len(not_deleted)
                remaining = list(not_deleted - deleted)
                perm = torch.randperm(len(remaining)).tolist()
                all_indices = np.array(remaining)[perm].tolist()

            idx = all_indices.pop()

            if idx not in deleted and len(neighbors[idx]) < 3 and idx not in protected:
                break

        if len(neighbors[idx]) == 2:
            n1, n2 = neighbors[idx]
            neighbors[n1].remove(idx)
            neighbors[n2].remove(idx)
            neighbors[n1].add(n2)
            neighbors[n2].add(n1)
        elif len(neighbors[idx]) == 1:
            n1 = neighbors[idx].pop()
            neighbors[n1].remove(idx)

        del neighbors[idx]
        deleted.add(idx)

    not_deleted = list(not_deleted - deleted)
    return neighbors, not_deleted

def remap_neighbors(x):
    """
    Remap node indices to be between 0 and the number of nodes.

    Args:
        x: Dict of node id mapping to the node's neighbors.
    Returns:
        ordered_x: Dict with neighbors with new node ids.
        subsampled2new: Mapping between old and new indices (dict).
    """
    # Create maps between new and old indices.
    subsampled2new = {k: i for i, k in enumerate(sorted(x))}

    # Re-map indices to 1..N.
    ordered_x = {i: x[k] for i, k in enumerate(sorted(x))}

    # Re-map keys of neighbors
    for k in ordered_x:
        ordered_x[k] = {subsampled2new[x] for x in ordered_x[k]}

    return ordered_x, subsampled2new


# ======================================================================================================================
