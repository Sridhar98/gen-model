import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from dgl import DGLGraph
import dgl
import numpy as np
import json
from params import *

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)
        self.conv3 = torch.nn.Conv2d(in_channels=1,out_channels=1,kernel_size=1)
        self.linear = nn.Linear(parameters['parts']*4,2*parameters['latent_size'])

    def forward(self, g, inputs,adj):

        #pass through the 2 gcn layers
        inputs = torch.mm(adj,inputs)
        batched_inputs = np.reshape(inputs,[-1,1,parameters['parts'],5])
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = torch.mm(adj,h)
        h = self.conv2(g, h)
        h = torch.relu(h)

        #skip connection
        skip = self.conv3(batched_inputs[:,:,:,:4])
        print('skip.shape: ',skip.shape)
        skip = torch.reshape(skip,(-1,4*parameters['parts']))
        #skip = torch.flatten(skip)
        skip = self.linear(skip)

        return h,skip

def add_features(G,parts,X):
    #feature matrix
    embed = nn.Embedding(parts, 5)  # p nodes with embedding dim equal to 5
    embed.weight = torch.nn.Parameter(torch.from_numpy(X))
    G.ndata['feat'] = embed.weight
    return embed
