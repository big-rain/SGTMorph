#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/7 22:33
# @Author  : ShengPengpeng
# @File    : transformer.py
# @Description :
import copy
from typing import Sequence, Union, Tuple

from torch_geometric.transforms import Compose

from tran.reduce_node import RanDomReduceNodes
from tran.augment_node_position import Augment_node_position
import torch_geometric.transforms as T


# # 抖动
# class RanDomJitter:
#     def __init__(self, translate: Union[float, int, Sequence]):
#         self.translate = translate
#         # self.trans = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
#
#     def __call__(self, data):
#         (n, dim), t = data.pos.size(), self.translate
#         if isinstance(t, numbers.Number):
#             t = list(repeat(t, times=dim))
#         assert len(t) == dim
#
#         ts = []
#         for d in range(dim):
#             ts.append(data.pos.new_empty(n).uniform_(-abs(t[d]), abs(t[d])))
#
#         data.pos = data.pos + torch.stack(ts, dim=-1)
#         for i in range(len(data.edge_index)):
#             data.edge_attr[i] = eucliDist(data.pos[data.edge_index[i][0]], data.pos[data.edge_index[i][1]])
#             data.edge_weight[i] = eucliDist(data.pos[data.edge_index[i][0]], data.pos[data.edge_index[i][1]])
#         # self.trans(data)
#         return data
#
#     def __repr__(self):
#         return f'{self.__class__.__name__}({self.translate})'
#
#
# # 旋转
# class RandomRotate:
#     def __init__(self, degrees: Union[Tuple[float, float], float], axis: int = 0):
#         if isinstance(degrees, numbers.Number):
#             degrees = (-abs(degrees), abs(degrees))
#         assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
#         self.degrees = degrees
#         self.axis = axis
#
#     def __call__(self, data):
#         degree = math.pi * random.uniform(*self.degrees) / 180.0
#         sin, cos = math.sin(degree), math.cos(degree)
#
#         if data.pos.size(-1) == 2:
#             matrix = [[cos, sin],
#                       [-sin, cos]]
#         else:
#             if self.axis == 0:
#                 matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
#
#             elif self.axis == 1:
#                 matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
#             else:
#                 matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
#
#         return LinearTransformation(torch.tensor(matrix))(data)
#
#
#     def __repr__(self) -> str:
#         return (f'{self.__class__.__name__}({self.degrees}, axis={self.axis})')
# #
# # 翻转
#
# class RandomFlip:
#     def __init__(self, axis: int, p: float = 0.5):
#         self.axis = axis
#         self.p = p
#
#     def __call__(self, data: Data) -> Data:
#         if random.random() < self.p:
#             pos = data.pos.clone()
#             pos[..., self.axis] = -pos[..., self.axis]
#             data.pos = pos
#         return data
#
#     def __repr__(self) -> str:
#         return f'{self.__class__.__name__}(axis={self.axis}, p={self.p})'

#
# # 剪枝
#
# class DropEdges:
#     r"""Drops edges with probability p."""
#     def __init__(self, p, force_undirected=False):
#         assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p
#
#         self.p = p
#         self.force_undirected = force_undirected
#
#     def __call__(self, data):
#         edge_index = data.edge_index
#         edge_attr = data.edge_attr if 'edge_attr' in data else None
#
#         edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=self.p, force_undirected=self.force_undirected)
#
#         data.edge_index = edge_index
#         if edge_attr is not None:
#             data.edge_attr = edge_attr
#         return data
#
#     def __repr__(self):
#         return '{}(p={}, force_undirected={})'.format(self.__class__.__name__, self.p, self.force_undirected)


##########################################################################################



def get_graph_transformer(keep_nodes: Union[float, int, Sequence],
                          drop_branch: Union[float, int, Sequence],
                          jitter_scale: Union[Tuple[float, float], float],
                          trans_scale: Union[Tuple[float, float], float],
                          rota = True,
                          ):
    transforms = list()

    # make copy of graph
    transforms.append(copy.deepcopy)

    transforms.append(RanDomReduceNodes(keep_node=keep_nodes, n_branch=drop_branch))
    #
    transforms.append(Augment_node_position(jitter_scale=jitter_scale, trans_scale=trans_scale, rota=rota))

    transforms.append(T.AddRandomWalkPE(walk_length=20, attr_name='pe'))

    return Compose(transforms)

class MultiViewDataInjector(object):
    def __init__(self, *args):
        self.transforms = args[0]

    def __call__(self, sample):
        output = [transform(sample) for transform in self.transforms]

        return output

if __name__ == '__main__':
    import os.path as osp
    from torch_geometric.loader import DataLoader
    import torch_geometric.transforms as T
    from model.class_model import CGNN
    from dataloader.BIL import BILdataset
    import numpy as np
    path = osp.join('..', 'data', 'BIL')
    # pre_transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
    transformer_view_1 = get_graph_transformer(keep_nodes=500,
                                               drop_branch=10,
                                               degrees=15)

    train_dataset = BILdataset(path, subset=True, split='train', transform=MultiViewDataInjector([transformer_view_1, transformer_view_1]))
    # val_dataset = BILdataset(path, subset=True, split='val')

    # data = transformer_view_1(train_dataset[0])
    for data in train_dataset:
        print(data)



