#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/16 20:12
# @Author  : ShengPengpeng
# @File    : class_model.py
# @Description :


import torch.nn as nn
import inspect
from typing import Any, Dict, Optional
import argparse
import os.path as osp
import sklearn.metrics as metrics
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
import numpy as np
from dataloader.PFC import PFCdataset
from dataloader.N7 import N7dataset
from dataloader.JML import JMLdataset
from dataloader.BIL import BILdataset
from dataloader.ACT import ACTDatset

from torch.nn import (
    BatchNorm1d,
    Embedding,
    Dropout,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)

# from layer.gatagcn_layer import GatedGCNLayer

from torch_geometric.loader import DataLoader
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)

from torch_geometric.datasets import ZINC
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_geometric.transforms as T
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GINEConv, GPSConv, GATConv, ResGatedGraphConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

from tran.lr_scheduler import cosine_with_warmup_scheduler
from loss.weighted_cross_entropy import weight_CE_loss

import warnings
warnings.filterwarnings("ignore")
# warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')

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
            print(dist)
            for k in range(num_bins):
                # 计算离散的距离
                bin_center = k - num_bins // 2
                # 使用高斯函数计算编码
                encoding[i, j, k] = np.exp(-0.5 * ((dist - bin_center) / sigma) ** 2)

    return encoding

class LGConv(torch.nn.Module):
    def __init__(self,

                 channels: int,
                 conv: Optional[MessagePassing],

                 heads: int = 1,
                 dropout: float = 0.0,
                 act: str = 'relu',
                 act_kwargs: Optional[Dict[str, Any]] = None,
                 norm: Optional[str] = 'batch_norm',
                 norm_kwargs: Optional[Dict[str, Any]] = None,
                 attn_type: str = 'multihead',
                 attn_kwargs: Optional[Dict[str, Any]] = None,
                 ):

        super().__init__()

        self.channels = channels
        self.conv = conv
        self.heads = heads
        self.dropout = dropout
        # print(dropout)
        self.attn_type = attn_type

        attn_kwargs = attn_kwargs or {}
        if attn_type == 'multihead':
            self.attn = torch.nn.MultiheadAttention(
                channels,
                heads,
                batch_first=True,
                **attn_kwargs,
            )
        elif attn_type == 'performer':
            self.attn = PerformerAttention(
                channels=channels,
                heads=heads,
                **attn_kwargs,
            )
        else:
            # TODO: Support BigBird
            raise ValueError(f'{attn_type} is not supported')

        self.input_norm = torch.nn.LayerNorm(channels)

        # We follow the paper in that all hidden dims are equal to the embedding dim
        self.mlp = Sequential(
            Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = 'batch' in signature.parameters

    def forward(self,
                x: Tensor,
                edge_index: Adj,
                # attn_bias,
                batch: Optional[torch.Tensor] = None,
                **kwargs,
                ) -> Tensor:

        # local
        hs = []
        if self.conv is not None:  # Local MPNN.
            h = self.conv(x, edge_index, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h = self.norm1(h, batch=batch)
                else:
                    h = self.norm1(h)
            hs.append(h)

        # Global attention transformer-style model.
        # print(x.shape)
        # print(batch.shape)
        # ========
        h, mask = to_dense_batch(x, batch)
        # print(h.shape)
        if isinstance(self.attn, torch.nn.MultiheadAttention):
            h, _ = self.attn(h, h, h, key_padding_mask=~mask,
                             need_weights=False)
        elif isinstance(self.attn, PerformerAttention):
            h = self.attn(h, mask=mask)

        h = h[mask]
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x  # Residual connection.

        if self.norm2 is not None:
            if self.norm_with_batch:
                h = self.norm2(h, batch=batch)
            else:
                h = self.norm2(h)
        # print('norm 2 h', h)
        hs.append(h)
        out = sum(hs)  # Combine local and global outputs.

        out = out + self.mlp(out)
        if self.norm3 is not None:
            if self.norm_with_batch:

                out = self.norm3(out, batch=batch)
            else:
                out = self.norm3(out)
        # print('norm 3 out', out)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'conv={self.conv}, heads={self.heads})')

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.conv is not None:
            self.conv.reset_parameters()
        self.attn._reset_parameters()
        reset(self.mlp)
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.norm3 is not None:
            self.norm3.reset_parameters()


class TransGraph(nn.Module):
    def __init__(self,
                 channels: int,
                 pe_dim: int,
                 num_layer: int,
                 attn_type: str,
                 attn_kwargs: Dict[str, Any]):
        super().__init__()

        # 预处理 修改ing
        self.node_emb = Linear(3, channels - pe_dim)
        # self.node_emb = Linear(3, channels)
        self.pe_lin = Linear(20, pe_dim)
        self.pe_norm = BatchNorm1d(20)
        self.edge_emb = Embedding(64, channels)
        # self.edge_emb = Linear(1, channels)
        self.convs = ModuleList()

        for _ in range(num_layer):
            nn = ResGatedGraphConv(in_channels=channels, out_channels=channels, act=torch.nn.ReLU(), edge_dim=channels)
            # nn = ResGatedGraphConv(in_channels=channels, out_channels=channels, act=torch.nn.ReLU())
            # nn = None
            conv = LGConv(channels, nn, heads=8, attn_type=attn_type, attn_kwargs=attn_kwargs)
            self.convs.append(conv)

    def forward(self,
                x,
                pe,
                edge_index,
                edge_attr,
                batch,
                ):

        # print(x.shape)
        x_pe = self.pe_norm(pe)
        node_emb = self.node_emb(x.squeeze(-1))
        x = node_emb
        pe_lin = self.pe_lin(x_pe)
        x = torch.cat((node_emb, pe_lin), dim=1)


        edge_attr = self.edge_emb(edge_attr)
        # print('x', x.shape)

        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
            # x = conv(x, edge_index, batch)
        # x = global_max_pool(x, batch)
        x = global_mean_pool(x, batch)
        return x



class CGNN(nn.Module):
    def __init__(self, channels, attn_type, **attn_kwargs):
        super().__init__()

        self.encoder = TransGraph(channels=128, pe_dim=16, num_layer=3, attn_type=attn_type, attn_kwargs=attn_kwargs)

        self.mlp = Sequential(
            Linear(channels, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, 7),
        )


    def forward(self, x, pe, edge_index, edge_attr, batch):

        x = self.encoder(x, pe, edge_index, edge_attr, batch)
        x = self.mlp(x)
        return x


def train(train_loader):
    model.train()
    correct = 0
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.pos, data.pe, data.edge_index, data.edge_attr, data.batch)
        pred = out.argmax(dim=1)  # 使用概率最高的类别
        correct += int((pred == data.y).sum())  # 检查真实标签

        loss = criterion(out, data.y)

        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_loader.dataset), correct / len(train_loader.dataset)



@torch.no_grad()
def test(loader):
    model.eval()
    test_true = []
    test_pred = []
    correct = 0
    for data in loader:  # 批遍历测试集数据集。
        data = data.to(device)
        out = model(data.pos, data.pe, data.edge_index, data.edge_attr, data.batch)  # 一次前向传播
        # print(out.shape)
        pred = out.argmax(dim=1)  # 使用概率最高的类别
        # correct += int((pred == data.y).sum())  # 检查真实标签

        test_true.append(data.y.cpu().numpy())
        # print('test: ',test_true)
        test_pred.append(pred.detach().cpu().numpy())

        # print(test_true)

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)

    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    return test_acc, avg_per_class_acc


if __name__ == '__main__':

    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PFC')
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'neuron7')
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'BIL')
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ACT')
    # pre_reansform_1 = T.A
    pre_transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
    transform = T.Compose([
        T.RandomJitter(0.01),
        T.RandomRotate(15, axis=0),
        T.RandomRotate(15, axis=1),
        T.RandomRotate(15, axis=2),
        T.AddRandomWalkPE(walk_length=20, attr_name='pe')
    ])

    # train_dataset = BILdataset(path, subset=True, split='train',  transform=transform)
    # val_dataset = BILdataset(path, subset=True, split='val', transform=pre_transform)
    # train_dataset = ACTDatset(path, subset=True, split='train',  transform=transform)
    # val_dataset = ACTDatset(path, subset=True, split='val', transform=pre_transform)
    # train_dataset = JMLdataset(path, subset=True, split='train',  transform=transform)
    # val_dataset = JMLdataset(path, subset=True, split='val', transform=pre_transform)
    train_dataset = N7dataset(path, subset=True, split='train',  transform=transform)
    val_dataset = N7dataset(path, subset=True, split='val', transform=pre_transform)

    # train_dataset = PFCdataset(path, subset=True, split='train',  transform=transform)
    # val_dataset = PFCdataset(path, subset=True, split='val', transform=pre_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    parser = argparse.ArgumentParser()
    parser.add_argument('--attn_type', default='multihead',
                        help="Global attention type such as 'multihead' or 'performer'.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attn_kwargs = {'dropout': 0.5}
    model = CGNN(channels=128, attn_type=args.attn_type, **attn_kwargs).to(device)

    # pkl_path = '../ckgs/BIL/BIL_transformer_gatedgcn_0.9213.pt'
    # state_dict = torch.load(pkl_path)
    # model.load_state_dict(state_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    scheduler = cosine_with_warmup_scheduler(optimizer, num_warmup_epochs=50, max_epoch=150)

    for epoch in range(1, 1000):
        max_acc = 0
        loss, train_acc = train(train_loader)
        val_acc, val_avg_acc = test(val_loader)
        scheduler.step()
        if val_acc > max_acc:
            max_acc = val_acc
            torch.save(model.state_dict(), osp.join('../ckgs/N7_ablation', 'N7_all_{:.4f}_{:.4f}.pt'.format(max_acc, val_avg_acc)))
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, train: {train_acc:.4f}, Val: {val_acc:.4f}, Val_acc: {val_avg_acc:.4f}')



