#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/14 16:00
# @Author  : ShengPengpeng
# @File    : ACT.py
# @Description :


from typing import Callable, List, Optional
import torch
import pandas as pd

import numpy as np
from tqdm import tqdm
from pathlib import Path
import pickle
import os.path as osp
from torch_geometric.data import InMemoryDataset, Data
import scipy.sparse as sp
import torch_geometric.transforms as T
import networkx as nx


from dataloader.Neuron_utils import remap_neighbors, neighbors_to_adjacency, eucliDist, subsample_graph

ACT_6_classes = {
    "Isocortex_layer23": 0,
    "Isocortex_layer4": 1,
    "Isocortex_layer5": 2,
    "Isocortex_layer6": 3,
}


class ACTDatset(InMemoryDataset):

    def __init__(
            self,
            root: str,
            subset: bool = False,
            split: str = 'train',
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
    ):
        assert split in ['train', 'val']
        super().__init__(root, transform, pre_transform, pre_filter)
        self.subset = subset
        # print(self.processed_paths[0])
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'train.pickle', 'val.pickle', 'test.pickle',
            'train.index', 'val.index', 'test.index'
        ]

    @property
    def processed_file_names(self) -> List[str]:
        return ['train.pt', 'val.pt']

    def download(self):
        pass

    def process(self):

        meta_data = pd.read_csv('../data/allen_cell_type/ACT_info_swc_10folds.csv')

        for split in ['train', 'val']:

            # indices = range(len(mols))
            # with open(osp.join(self.raw_dir, f'{split}_ids.npy'), 'r') as f:
            #     indices = [int(x) for x in f.read()[:-1].split(',')]

            cell_ids = list(np.load(Path('../data/allen_cell_type/raw/', f'{split}_ids.npy')))

            pbar = tqdm(total=len(cell_ids))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for cell_id in cell_ids:

                soma_id = 0
                features = np.load(Path(self.root, 'skeletons', str(cell_id), 'features.npy'))
                with open(Path(self.root, 'skeletons', str(cell_id), 'neighbors.pkl'), 'rb') as f:
                    neighbors = pickle.load(f)

                # get graph labels
                # scpecimen_id = cell_id[11:-8]
                swc__name = cell_id
                labels = meta_data[meta_data["swc__fname"] == swc__name]["structure_merge__acronym"].values[0]
                # labels = meta_data[meta_data["specimen__id"] == scpecimen_id]["structure_merge__acronym"].values[0]
                if labels == 'Isocortex_layer2/3':
                    labels = 'Isocortex_layer23'

                if labels in ACT_6_classes.keys():
                    label = ACT_6_classes[labels]
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

                    pos = features[:, :3]
                    # pos = 2 *( (pos - np.min(pos) )/ (np.max(pos) - np.min(pos)) ) - 1
                    pos = (pos - np.min(pos)) / (np.max(pos) - np.min(pos))

                    adj = neighbors_to_adjacency(neighbors, neighbors, pos=pos)

                    pos = torch.Tensor(pos)
                    # x = torch.Tensor(features)

                    # graph = nx.from_numpy_array(adj)
                    # attn_bias = nx.floyd_warshall_numpy(graph, weight='weight')
                    # attn_bias = torch.Tensor(attn_bias)

                    edge_index_temp = sp.coo_matrix(adj)
                    values = edge_index_temp.data
                    edge_weight = torch.Tensor(values) * 100
                    edge_attr = torch.LongTensor(values)
                    # 边上对应权重值weight
                    indices = np.vstack((edge_index_temp.row, edge_index_temp.col))  # 我们真正需要的coo形式
                    edge_index = torch.LongTensor(indices)  # 我们真正需要的coo形式
                    # data = Data(x=pos, edge_index=edge_index, edge_attr=edge_attr, pos=pos, y=y)
                    data = Data(x=pos, edge_index=edge_index, edge_attr=edge_attr, edge_weight=edge_weight, pos=pos,  y=y)

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)
                pbar.update(1)
            pbar.close()
            # self.data, self.slices = self.collate(data_list)
            torch.save(self.collate(data_list), osp.join(self.processed_dir, f'{split}.pt'))
            # torch.save((self.data, self.slices), osp.join(self.processed_dir[0], f'{split}.pt'))


if __name__ == '__main__':

    from tran.transformer import *
    import torch_geometric.transforms as T
    from tran.t1 import Augmentor_Transform, MyAug_Identity

    path = '../data/allen_cell_type'

    transform_view_2 = T.Compose([
        Augmentor_Transform['nodeDrop'](prob=0.1),
        T.AddRandomWalkPE(walk_length=20, attr_name='pe')
    ])

    transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')

    # train_dataset = Neuron(path, split='train', pre_transform=transform)
    val_dataset = ACTDatset(path, split='val', transform=transform_view_2)
    # print(val_dataset)
    for data in val_dataset:
        print(data)
        # print('x',data.x.dtype)              # float64
        # print('edge_index',data.edge_index.dtype)     # torch.int64
        # print('edge_attr ',data.edge_attr.dtype)      # torch.float32
        # print('y', data.y.dtype)              # torch.int64
        # print('pos', data.pos.dtype)            # float32
        # print('pe', data.pe.dtype)             # torch.float32


    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=64)