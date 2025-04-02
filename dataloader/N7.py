#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/23 15:23
# @Author  : ShengPengpeng
# @File    : N7.py
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

from dataloader.swc_utils import subsample_graph, rdb_graph, remap_neighbors
from tran.reduce_node import compute_node_distances
from torch_geometric import transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import Linear
from tran.function import neighbors_to_adjacency_pos, neighbors_to_adjacency_attr



n_7_classes = {
    'amacrine': 0,
    'aspiny': 1,
    'basket': 2,
    'bipolar': 3,
    'pyramidal': 4,
    'spiny': 5,
    'stellate': 6
}




class N7dataset(InMemoryDataset):

    def __init__(
            self,
            root: str,
            subset: bool = False,
            split: str = 'train',
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
    ):
        assert split in ['train', 'val', 'all',]
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

        meta_data = pd.read_csv('../data/neuron7/neuron7_info_swc_10folds.csv')

        for split in ['train', 'val', 'all']:
        # for split in ['train']:
            # indices = range(len(mols))
            # with open(osp.join(self.raw_dir, f'{split}_ids.npy'), 'r') as f:
            #     indices = [int(x) for x in f.read()[:-1].split(',')]

            cell_ids = list(np.load(Path('../data/neuron7/raw/', f'{split}_ids.npy')))

            pbar = tqdm(total=len(cell_ids))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for cell_id in cell_ids:
                if cell_id == 'desktop.ini':
                    continue

                soma_id = 0
                features = np.load(Path(self.root, 'skeletons', str(cell_id), 'features.npy'))
                with open(Path(self.root, 'skeletons', str(cell_id), 'neighbors.pkl'), 'rb') as f:
                    neighbors = pickle.load(f)


                # print(cell_id)
                scpecimen_id = cell_id
                labels = meta_data[meta_data["specimen__id"] == scpecimen_id]["structure_merge__acronym"].values[0]

                if labels in n_7_classes.keys():
                    label = n_7_classes[labels]
                else:
                    label = -1

                if label != -1:
                    y = torch.tensor([label])

                    neighbors, not_deleted = subsample_graph(neighbors=neighbors,
                                                             not_deleted=set(range(len(neighbors))),
                                                             keep_nodes=1000,
                                                             protected=[soma_id])


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

    path = '../data/neuron7/swc/'


    name = 'Rbp4-Cre_KL100_Ai14-204744-03-02-01_496123859_m.CNG.swc'

    # transform_view_2 = T.Compose([
    #     Augmentor_Transform['nodeDrop'](prob=0.1),
    #     T.AddRandomWalkPE(walk_length=20, attr_name='pe')
    # ])

    # transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')

    train_dataset = N7dataset(path, split='train')
    # val_dataset = BILdataset(path, split='val', transform=transform_view_2)
    for data in train_dataset:
        print(data)