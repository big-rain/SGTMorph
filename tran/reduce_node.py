#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/24 10:14
# @Author  : ShengPengpeng
# @File    : reduce_node.py
# @Description :

import math
import torch
import networkx as nx
import scipy.sparse as sp
import numpy as np
from typing import Sequence, Union, Tuple
from torch_geometric.utils import to_scipy_sparse_matrix

from tran.function import neighbors_to_adjacency_attr, neighbors_to_adjacency_pos, adjacency_to_neighbors, neighbors_to_adjacency_attr_train


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

# ================================================================================

def get_leaf_branch_nodes(neighbors):
    """"
    Create list of candidates for leaf and branching nodes.
    Args:
        neighbors: dict of neighbors per node
    """
    all_nodes = list(neighbors.keys())
    leafs = [i for i in all_nodes if len(neighbors[i]) == 1]

    candidates = leafs
    next_nodes = []
    for l in leafs:
        next_nodes += [n for n in neighbors[l] if len(neighbors[n]) == 2]

    while next_nodes:
        s = next_nodes.pop(0)
        candidates.append(s)
        next_nodes += [n for n in neighbors[s] if
                       len(neighbors[n]) == 2 and n not in candidates and n not in next_nodes]

    return candidates


def compute_node_distances(idx, neighbors):
    """"
    Computation of node degree.
    Args:
        idx: index of node
        neighbors: dict of neighbors per node
    """
    queue = []
    queue.append(idx)

    degree = dict()
    degree[idx] = 0

    while queue:
        s = queue.pop(0)
        prev_dist = degree[s]

        for neighbor in neighbors[s]:
              if neighbor not in degree:
                queue.append(neighbor)
                degree[neighbor] = prev_dist + 1
    return degree


def drop_random_branch(nodes, neighbors, distances, keep_nodes=200):
    """
    Removes a terminal branch. Starting nodes should be between
    branching node and leaf (see leaf_branch_nodes)

    Args:
        nodes: List of nodes of the graph
        neighbors: Dict of neighbors per node
        distances: Dict of distances of nodes to origin
        keep_nodes: Number of nodes to keep in graph
    """
    start = list(nodes)[torch.randint(len(nodes), (1,)).item()]
    to = list(neighbors[start])[0]

    # print(len(nodes))
    # print('node', nodes)
    #
    # print('start', start)
    # print('to', to)
    # print(len(distances))
    # print('distance[start]', distances[start])
    # print('distance[to]', distances[to])

    if distances[start] > distances[to]:
        start, to = to, start
    #
    # print('len node', len(nodes))
    # print('len neighbor', len(neighbors))
    # print('len distance',len(distances))

    drop_nodes = [to]
    next_nodes = [n for n in neighbors[to] if n != start]

    while next_nodes:
        s = next_nodes.pop(0)
        drop_nodes.append(s)
        next_nodes += [n for n in neighbors[s] if n not in drop_nodes]

    if len(neighbors) - len(drop_nodes) < keep_nodes:
        return neighbors, set()
    else:
        # Delete nodes.
        for key in drop_nodes:
            if key in neighbors:
                for k in neighbors[key]:
                    neighbors[k].remove(key)
                del neighbors[key]

        return neighbors, set(drop_nodes)

def jitter_random_branch(nodes, neighbors, distances):
    """
    Removes a terminal branch. Starting nodes should be between
    branching node and leaf (see leaf_branch_nodes)

    Args:
        nodes: List of nodes of the graph
        neighbors: Dict of neighbors per node
        distances: Dict of distances of nodes to origin
    """
    start = list(nodes)[torch.randint(len(nodes), (1,)).item()]
    to = list(neighbors[start])[0]

    # print(len(nodes))
    # print('node', nodes)
    #
    # print('start', start)
    # print('to', to)
    # print(len(distances))
    # print('distance[start]', distances[start])
    # print('distance[to]', distances[to])

    if distances[start] > distances[to]:
        start, to = to, start
    #
    # print('len node', len(nodes))
    # print('len neighbor', len(neighbors))
    # print('len distance',len(distances))

    drop_nodes = [to]
    next_nodes = [n for n in neighbors[to] if n != start]

    while next_nodes:
        s = next_nodes.pop(0)
        drop_nodes.append(s)
        next_nodes += [n for n in neighbors[s] if n not in drop_nodes]

    # if len(neighbors) - len(drop_nodes) < keep_nodes:
    #     return neighbors, set()
    # else:
        # Delete nodes.
    for key in drop_nodes:
        if key in neighbors:
            for k in neighbors[key]:
                neighbors[k]
                # del neighbors[key]

    return neighbors, set(drop_nodes)



class RanDomReduceNodes:
    def __init__(self, keep_node: Union[int], n_branch: int):
        self.keep_node = keep_node
        self.n_branch = n_branch
        self.soma_id = 0

    def __call__(self, data):
        # print(data)

        # for i in range(batch_size):
        #
        adj = to_scipy_sparse_matrix(data.edge_index)

        # G = nx.Graph(adj)
        #
        # print('aaaaa')

        neighbors = adjacency_to_neighbors(adj_matrix=adj)
        # print('neighbor', len(neighbors))

        leaf_branch_nodes = get_leaf_branch_nodes(neighbors)
        # print('leaf branch nodes', leaf_branch_nodes)
        # Using the distances we can infer the direction of an edge.

        # distances = compute_node_distances(self.soma_id, neighbors)
        distances = data.distance
        # print('distance', len(distances))
        leaf_branch_nodes = set(leaf_branch_nodes)
        not_deleted = set(range(len(neighbors)))

        for i in range(self.n_branch):
            neighbors, drop_nodes = drop_random_branch(leaf_branch_nodes,
                                                       neighbors,
                                                       distances,
                                                       keep_nodes=self.keep_node)

            not_deleted -= drop_nodes
            leaf_branch_nodes -= drop_nodes

            if len(leaf_branch_nodes) == 0:
                break


        neighbors, not_deleted = subsample_graph(neighbors=neighbors,
                                                 not_deleted=not_deleted,
                                                 keep_nodes=self.keep_node,
                                                 protected=[0])
        neighbors, subsampled2new = remap_neighbors(neighbors)

        # print(subsampled2new)


        features = data.x[list(subsampled2new.keys())]
        pos = data.pos[list(subsampled2new.keys())]
        node_radius = features[:, 3]
        node_types = features[:, 4:]


        subsampling_adj = neighbors_to_adjacency_pos(neighbors, neighbors, pos, node_radius)
        edge_index_temp = sp.coo_matrix(subsampling_adj)
        weight_values = edge_index_temp.data
        edge_weight = torch.Tensor(weight_values)

        subsampling_adj_atr = neighbors_to_adjacency_attr_train(neighbors, neighbors, node_types)
        edge_index_temp_atr = sp.coo_matrix(subsampling_adj_atr)
        attr_value = edge_index_temp_atr.data
        edge_attr = torch.LongTensor(attr_value)

        indices = np.vstack((edge_index_temp.row, edge_index_temp.col))  # 我们真正需要的coo形式
        edge_index = torch.LongTensor(indices)  # 我们真正需要的coo形式


        data.x = features
        data.edge_index = edge_index
        data.pos = pos
        data.edge_attr = edge_attr
        data.edge_weight = edge_weight

        return data

    def __repr__(self):
        return f'{self.__class__.__name__}({self.keep_node})'