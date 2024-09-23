#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/31 21:41
# @Author  : ShengPengpeng
# @File    : swc_utils.py
# @Description :

import torch
from rdp import rdp
import numpy as np
from collections import Counter




def get_sample_leaf_branch_nodes(neighbors, coords):
    """"
    Create list of candidates for leaf and branching nodes.
    Args:
        neighbors: dict of neighbors per node
    """
    all_nodes = list(neighbors.keys())
    leafs = [[i] for i in all_nodes if len(neighbors[i]) == 1]
    all_path = []
    retime_node = []
    c = 0
    ss = 0

    for next_nodes in leafs:
        path = []
        while next_nodes:
            s = next_nodes.pop(0)
            # print(s)
            next_nodes += [n for n in neighbors[s] if
                           len(neighbors[n]) == 2 and n not in path and n not in next_nodes]
            path.append(s)
            all_path.append(s)
            ss = ss + 1

        # print(path)
        simplified = rdp(coords[path], epsilon=0.5)
        # print(simplified)
        for i in simplified:
            idx = np.where(coords == i)
            retime_node.append(Counter(idx[0]).most_common(1)[0][0])
            c = c + 1
        # a.append(b)

    deleted = set(all_path) - set(retime_node)
    # all_path.append(b)

    return all_path, deleted, c, ss




def rdb_graph(neighbors=None, deleted=None):
    # print(deleted)
    if neighbors is not None:
        k_nodes = len(neighbors)
    else:
        raise ValueError('neighbors must be provided')

    # indices as set in random order
    not_deleted = set(range(len(neighbors)))
    # print(all_deleted)
    deleted = set(deleted)
    # print(len(deleted))
    # deleted = all_deleted - deleted

    while deleted:
        idx = deleted.pop()
        # print(idx)
        if len(neighbors[idx]) == 2:
            # print('bbb')
            n1, n2 = neighbors[idx]
            neighbors[n1].remove(idx)
            neighbors[n2].remove(idx)
            neighbors[n1].add(n2)
            neighbors[n2].add(n1)
        elif len(neighbors[idx]) == 1:
            n1 = neighbors[idx].pop()
            neighbors[n1].remove(idx)
        del neighbors[idx]

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
