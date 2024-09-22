#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/20 22:02
# @Author  : ShengPengpeng
# @File    : neuron.py
# @Description :

from pathlib import Path
import pickle
from tqdm import tqdm
import networkx as nx
from swc import read_swc
from Neuron_utils import neighbors_to_adjacency_1
# from process import connect_graph
import os
import numpy as np
import random


def connect_graph(adj_matrix, neighbors, features, verbose=False):
    """
    Check if graph consists of only one connected component. If not iterativly connect two points of two unconnected components with shortest distance between them.
    Args:
        adj_matrix: adjacency matrix of graph (N x N)
        neighbors: dict of neighbors per node
        features: features per node (N x D)
    """
    neighbors2 = {k: set(v) for k, v in neighbors.items()}
    G = nx.Graph(adj_matrix)
    num_comp = nx.number_connected_components(G)
    count = 1
    while num_comp > 1:
        components = {i: l for i, l in enumerate(list(nx.connected_components(G)))}
        components_ids = list(components.keys())
        for i, c_id in enumerate(components_ids):
            nodes = components[c_id]
            leaf_nodes = [n for n in nodes if len(neighbors2[n]) == 1]

            if len(leaf_nodes) > 0:

                min_comp_dist = np.inf
                min_comp_dist_id = -1
                min_comp_dist_node = -1
                for i, l in enumerate(leaf_nodes):
                    ne = neighbors2[l]
                    ne = list(ne)[0]

                    node_pos = features[l][:3]

                    nodes_pos_diff = ((features[:, :3] - node_pos) ** 2).sum(axis=1)
                    nodes_pos_diff[ne] = np.inf
                    nodes_pos_diff[l] = np.inf
                    nodes_pos_diff[list(nodes)] = np.inf

                    min_dist_id = np.argmin(nodes_pos_diff)
                    min_dist = np.min(nodes_pos_diff)

                    if min_comp_dist > min_dist:
                        min_comp_dist = min_dist
                        min_comp_dist_id = min_dist_id
                        min_comp_dist_node = l

                if min_comp_dist_id != -1 and min_comp_dist_node != -1:
                    neighbors2[min_comp_dist_id].add(min_comp_dist_node)
                    neighbors2[min_comp_dist_node].add(min_comp_dist_id)

        adj_matrix = neighbors_to_adjacency_1(neighbors2, range(len(neighbors2)))
        G = nx.Graph(adj_matrix)
        num_comp = nx.number_connected_components(G)
        if verbose:
            print(count, num_comp)
        count += 1

    return adj_matrix, neighbors2


def data_process(data_path, save_path):
    # with open(file, 'r', encoding='utf-8') as csvfile:
    #     reader = csv.reader(csvfile)
    #     fname = [row[2] for row in reader]

    # csv_path = "../data/info/bli.csv"
    # data = pd.read_csv(file)
    # fname = np.array(data['swc__fname'])
    fname = []
    # fname.append(data_path)
    for files in os.listdir(data_path):
        fname.append(files)

    for fname_eswc in tqdm(fname):
    # for fname_eswc in fname:
        # print(fname_eswc)

        path = Path(save_path, 'skeletons/', fname_eswc)
        path.mkdir(parents=True, exist_ok=True)

        morphology = read_swc(os.path.join(data_path, fname_eswc))

        morphology.strip_type(2)
        # SOMA = 1
        # AXON = 2
        # DENDRITE = 3
        # BASAL_DENDRITE = 3
        # APICAL_DENDRITE = 4

        # Get soma coordinates.
        soma = morphology.soma
        soma_pos = np.array([soma['x'], soma['y'], soma['z']])
        soma_id = soma['id']

        # process graph
        neighbors = {}
        idx2node = {}

        for i, item in enumerate(morphology.compartment_list):

            # get node feature
            sec_type = [0, 0, 0, 0, 0]

            if item['type'] - 1 > 3:
                sec_type = [0, 0, 0, 0, 1]
            else:
                # print(item['type'] - 1)
                sec_type[item['type'] - 1] = 1
            feat = tuple([item['x'], item['y'], item['z'], item['radius']]) + tuple(sec_type)
            idx2node[i] = feat

            # get neighbors
            neighbors[i] = set(item['children'])
            if item['parent'] != -1:
                neighbors[i].add(item['parent'])

        features = np.array(list(idx2node.values()))
        #
        assert ~np.any(np.isnan(features))

        ## Normalize soma position to origin.
        norm_features = features.copy()
        # print('before', norm_features)

        # plot_neuron(neighbors, norm_features)
        # plt.show()

        norm_features[:, :3] = norm_features[:, :3] - soma_pos

        ### 归一化处理，在后续处理
        # mean_x = norm_features[:, 0].mean()
        # mean_y = norm_features[:, 1].mean()
        # mean_z = norm_features[:, 2].mean()
        # std_x = norm_features[:, 0].std()
        # std_y = norm_features[:, 1].std()
        # std_z = norm_features[:, 2].std()
        # norm_features[:, :3] = (norm_features[:, :3] - [mean_x, mean_y, mean_z]) / [std_x, std_y, std_z]

        # ## 最大联通
        adj_matrix = neighbors_to_adjacency_1(neighbors, range(len(neighbors)))
        G = nx.Graph(adj_matrix)
        #
        if nx.number_connected_components(G) > 1:
            adj_matrix, neighbors = connect_graph(adj_matrix, neighbors, features)
        # #
        assert len(neighbors) == len(adj_matrix)
        #

        # #
        np.save(Path(path, 'features'), norm_features)
        with open(Path(path, 'neighbors.pkl'), 'wb') as f:
            pickle.dump(dict(neighbors), f, pickle.HIGHEST_PROTOCOL)



def creat_train_ids(data_path, file_path):
    file = os.listdir(data_path)
    np.save(Path(file_path, 'all_ids.npy'), file)

    cell_ids = list(np.load(Path(file_path, 'all_ids.npy')))
    print(len(cell_ids))

    train_ids = random.sample(cell_ids, int(0.9 * len(cell_ids)))
    print(len(train_ids))
    np.save(Path(file_path, 'train_ids.npy'), train_ids)

    val_ids = [i for i in cell_ids if i not in train_ids]
    print(len(val_ids))
    np.save(Path(file_path, 'val_ids.npy'), val_ids)



if __name__ == '__main__':
    # swc 文件处理
    data_path = '../data/allen_cell_type/swc'
    save_path = '../data/allen_cell_type/'
    data_process(data_path, save_path)
    #
    # file_path = '../data/allen_cell_type/'
    # creat_train_ids(data_path, file_path)