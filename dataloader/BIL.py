#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/14 16:00
# @Author  : ShengPengpeng
# @File    : BIL.py
# @Description :


import pickle
import shutil
from typing import Callable, List, Optional
import numpy as np
from pathlib import Path
import torch
import os

from torch_geometric.data.data import BaseData
from tqdm import tqdm
import math
import pandas as pd
import networkx as nx
import scipy.sparse as sp


from dataloader.swc_utils import remap_neighbors, rdb_graph, get_sample_leaf_branch_nodes,subsample_graph
from tran.reduce_node import compute_node_distances
from torch_geometric import transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import Linear
from tran.function import neighbors_to_adjacency_pos, neighbors_to_adjacency_attr


BIL_7_classes = {
    "Isocortex_layer23": 0,
    "Isocortex_layer4": 1,
    "Isocortex_layer5": 2,
    "Isocortex_layer6": 3,
    'CP': 4,
    'VPM': 5,
    # 'VPL': 6
}


class BILdataset(InMemoryDataset):

    def __init__(
            self,
            root: str,
            subset: bool = False,
            split: str = 'train',
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
    ):
        assert split in ['train', 'val', 'all', 'cla']
        super().__init__(root, transform, pre_transform, pre_filter)
        self.subset = subset
        path = os.path.join(self.processed_dir, f'{split}.pt')

        # print(path)

        self.data, self.slices = torch.load(path)
        # print(self.slices)

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'train.pickle', 'val.pickle', 'test.pickle',
            'train.index', 'val.index', 'test.index'
        ]

    @property
    def processed_file_names(self) -> List[str]:
        return ['train.pt', 'val.pt', 'all.pt']

    def download(self):
        pass

    def process(self):

        meta_data = pd.read_csv('../data/BIL/BIL_info_swc_10folds.csv')

        for split in ['train', 'val', 'all']:
            cell_ids = list(np.load(Path('../data/BIL/raw/', f'{split}_ids.npy')))

            pbar = tqdm(total=len(cell_ids))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for cell_id in cell_ids:

                soma_id = 0
                features = np.load(Path(self.root, 'skeletons', str(cell_id), 'features.npy'))
                with open(Path(self.root, 'skeletons', str(cell_id), 'neighbors.pkl'), 'rb') as f:
                    neighbors = pickle.load(f)


                # print(cell_id)
                scpecimen_id = cell_id[11:-8]
                labels = meta_data[meta_data["specimen__id"] == scpecimen_id]["structure_merge__acronym"].values[0]

                # scpecimen_id = cell_id[11:-8]
                # swc__name = cell_id
                # labels = meta_data[meta_data["swc__fname"] == swc__name]["structure_merge__acronym"].values[0]
                # labels = meta_data[meta_data["specimen__id"] == scpecimen_id]["structure_merge__acronym"].values[0]
                if labels == 'Isocortex_layer2/3':
                    labels = 'Isocortex_layer23'

                if labels in BIL_7_classes.keys():
                    label = BIL_7_classes[labels]
                else:
                    label = -1

                if label != -1:
                    y = torch.tensor([label])

                    # 下采样
                    # all_index, deleted_index, c, ss = get_sample_leaf_branch_nodes(neighbors, coords=features[:, :3])
                    # neighbors, not_deleted = rdb_graph(neighbors, deleted_index)

                    neighbors, not_deleted = subsample_graph(neighbors=neighbors,
                                                             not_deleted=set(range(len(neighbors))),
                                                             keep_nodes=1000,
                                                             protected=[soma_id])
                    # Remap neighbor indices to 0..999.

                    # Remap neighbor indices to 0..999.
                    neighbors, subsampled2new = remap_neighbors(neighbors)

                    # Accumulate features of subsampled nodes.
                    features = features[list(subsampled2new.keys())]

                    node_distances = compute_node_distances(0, neighbors)
                    a = np.zeros(len(node_distances))
                    for i in range(len(node_distances)):
                        # print(distance[i])
                        a[i] = node_distances[i]
                    distances = torch.Tensor(a)

                    x = features
                    x = torch.Tensor(x)
                    pos = features[:, :3]
                    node_radius = features[:, 3]
                    node_types = features[:, 4:]
                    pos = (pos - np.min(pos))/(np.max(pos) - np.min(pos))
                    pos = torch.Tensor(pos)


                    adj_attr = neighbors_to_adjacency_attr(neighbors, neighbors,  node_type=node_types)

                    G = nx.Graph(adj_attr)
                    if nx.number_connected_components(G) > 1:
                        print('attr')

                    edge_index_temp_attr = sp.coo_matrix(adj_attr)
                    values_attr = edge_index_temp_attr.data
                    edge_attr = torch.LongTensor(values_attr)

                    indices_attr = np.vstack((edge_index_temp_attr.row, edge_index_temp_attr.col))  # 我们真正需要的coo形式
                    edge_index_attr = torch.LongTensor(indices_attr)  # 我们真正需要的coo形式

####################################################################################################

                    adj_pos = neighbors_to_adjacency_pos(neighbors, neighbors, pos=pos, raduis=node_radius)
                    edge_index_temp_pos = sp.coo_matrix(adj_pos)
                    values_pos = edge_index_temp_pos.data
                    edge_weight = torch.Tensor(values_pos)

                    data = Data(x=x, edge_index=edge_index_attr, edge_attr=edge_attr,
                                edge_weight=edge_weight, pos=pos,  y=y, distance=distances, cell_id=cell_id[:-4])

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)
                pbar.update(1)
            pbar.close()

            torch.save(self.collate(data_list), os.path.join(self.processed_dir, f'{split}.pt'))


if __name__ == '__main__':

    from tran.transformer import *
    import torch_geometric.transforms as T
    from tran.transform_t2 import Augmentor_Transform, MyAug_Identity

    path = '../data/BIL'

    transform_view_2 = T.Compose([
        Augmentor_Transform['nodeDrop'](prob=0.1),
        T.AddRandomWalkPE(walk_length=20, attr_name='pe')
    ])

    # transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')

    train_dataset = BILdataset(path, split='all')
    print(len(train_dataset))
    # val_dataset = BILdataset(path, split='val', transform=transform_view_2)
    # for data in train_dataset:
    #     print(data)