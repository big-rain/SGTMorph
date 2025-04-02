#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/14 16:00
# @Author  : ShengPengpeng
# @File    : ACT.py
# @Description :


from typing import Callable, List, Optional
import torch
import pandas as pd

import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pickle
import os.path as osp
from torch_geometric.data import InMemoryDataset, Data
import scipy.sparse as sp
from dataloader.swc_utils import rdb_graph, remap_neighbors, get_sample_leaf_branch_nodes, subsample_graph
import torch_geometric.transforms as T
import networkx as nx
from tran.function import neighbors_to_adjacency_pos, neighbors_to_adjacency_attr
from tran.reduce_node import compute_node_distances


ACT_4_classes = {
    "Isocortex_layer23": 0,
    "Isocortex_layer4": 1,
    "Isocortex_layer5": 2,
    "Isocortex_layer6": 3,
}


ACT_3_classes = {
    "spiny": 0,
    "aspiny": 1,
    "sparsely spiny": 2,
    # "Isocortex_layer6": 3,
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
        assert split in ['val', 'train', 'all']
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
        return ['train.pt', 'val.pt', 'all.pt']

    def download(self):
        pass

    def process(self):

        meta_data = pd.read_csv('../data/ACT/ACT_info_swc_10folds.csv')

        for split in [ 'train', 'val', 'all']:
            # indices = range(len(mols))
            # with open(osp.join(self.raw_dir, f'{split}_ids.npy'), 'r') as f:
            #     indices = [int(x) for x in f.read()[:-1].split(',')]

            cell_ids = list(np.load(Path('../data/ACT/raw/', f'{split}_ids.npy')))
            pbar = tqdm(total=len(cell_ids))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for cell_id in cell_ids:
                # print(cell_id)
                soma_id = 0
                features = np.load(Path(self.root, 'skeletons', str(cell_id), 'features.npy'))
                with open(Path(self.root, 'skeletons', str(cell_id), 'neighbors.pkl'), 'rb') as f:
                    neighbors = pickle.load(f)

                # get graph labels
                # scpecimen_id = cell_id[11:-8]
                specimen_id = int(cell_id)
                # print(meta_data[meta_data["specimen_id"] ==specimen_id ])

                # labels = meta_data[meta_data["specimen_id"] == specimen_id]["dendrite_type"].values[0]
                #
                labels = meta_data[meta_data["specimen__id"] == specimen_id]["structure_merge__acronym"].values[0]
                if labels == 'Isocortex_layer2/3':
                    labels = 'Isocortex_layer23'

                if labels in ACT_4_classes.keys():
                    label = ACT_4_classes[labels]
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


                    pos = (pos - np.min(pos)) / (np.max(pos) - np.min(pos))
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


# ===================================================================================================

                    adj_pos = neighbors_to_adjacency_pos(neighbors, neighbors, pos=pos, raduis=node_radius)
                    edge_index_temp_pos = sp.coo_matrix(adj_pos)
                    values_pos = edge_index_temp_pos.data
                    edge_weight = torch.Tensor(values_pos)

                    data = Data(x=x, edge_index=edge_index_attr, edge_attr=edge_attr,
                                edge_weight=edge_weight, pos=pos, y=y, distance=distances, cell_id=cell_id)

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)
                pbar.update(1)
            pbar.close()

            torch.save(self.collate(data_list), os.path.join(self.processed_dir, f'{split}.pt'))


if __name__ == '__main__':

    # from tran.transformer import *
    # from BIL import BILdataset
    import torch_geometric.transforms as T
    # from tran.transform_t2 import Augmentor_Transform, MyAug_Identity

    path = '../data/PFC'

    # transform_view_2 = T.Compose([
    #     Augmentor_Transform['nodeDrop'](prob=0.1),
    #     T.AddRandomWalkPE(walk_length=20, attr_name='pe')
    # ])

    transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')

    # train_dataset = Neuron(path, split='train', pre_transform=transform)
    val_dataset = ACTDatset(path, split='train')
    print(len(val_dataset))
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