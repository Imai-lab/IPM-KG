import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from torch_geometric.data import DataLoader as GeometricDataLoader
from torch_geometric.data import DataLoader
import random
from tqdm import tqdm
import pickle
import torch
import torch.distributed as dist
import os
from node2vec import Node2Vec
import numpy as np

class SampledSAGEConv(SAGEConv):
    def __init__(self, in_channels, out_channels, sampling_ratio, **kwargs):
        super(SampledSAGEConv, self).__init__(in_channels, out_channels, **kwargs)
        self.sampling_ratio = sampling_ratio
    def forward(self, x, edge_index):
        # Get the number of neighbors to sample
        num_neighbors = int(self.sampling_ratio * edge_index.size(1))
        # Randomly sample a subset of the edges
        sampled_edges = random.sample(range(edge_index.size(1)), num_neighbors)
        sampled_edge_index = edge_index[:, sampled_edges]
        return super(SampledSAGEConv, self).forward(x, sampled_edge_index)


# GNN defenition for edge label classification
class EdgeClassifier(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, num_layers,dropout_prob):
        super(EdgeClassifier, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SampledSAGEConv(num_features, hidden_channels,sampling_ratio))
        for _ in range(num_layers - 1):
            self.convs.append(SampledSAGEConv(hidden_channels, hidden_channels,sampling_ratio))
        self.combine = torch.nn.Linear(2 * hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(2 * hidden_channels, num_classes)
        self.dropout_prob = dropout_prob
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x_old = x
            x = conv(x, edge_index)
            x = F.relu(x)
            x = torch.cat([x_old, x], dim=-1)
            x = self.combine(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_prob, training=self.training)
        # Construct edge features by concatenating node features
        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        edge_logits = self.linear(edge_features)
        return edge_logits

def load_edges(file_path):
    with open(file_path, 'r') as f:
        edges = [tuple(line.strip().split("\t")) for line in f.readlines()]
    return edges


def load_edge_labels(file_path):
    with open(file_path, 'r') as f:
        edge_labels = [int(line.strip()) for line in f.readlines()]
    return edge_labels


def node2vec_embedding(graph, dimensions=128, walk_length=20, num_walks=20, window_size=5, p=1, q=1, seed=42):
   node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, p=p, q=q, seed=seed)
   model = node2vec.fit(window=window_size, min_count=0, sg=1)
   return model


def create_data_object(x, edges, edge_labels):
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    num_edges = len(edges)
    edge_attr = torch.randn(num_edges, 100)  # Create a random 100-dimensional feature matrix for edges
    data = Data(x=x, edge_index=edge_index, edge_labels=edge_labels, edge_attr=edge_attr)  # Add edge_attr
    return data
