import torch
import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class MLP(nn.Module):
    """
    Graph-free baseline exactly matching GCN capacity. 
    Mathematically equivalent to GCN where adj = Identity matrix.
    """
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nclass)
        self.dropout = dropout
        
        # Match GCN weight initialization (Uniform)
        self.reset_parameters()
        
    def reset_parameters(self):
        import math
        stdv1 = 1. / math.sqrt(self.fc1.weight.size(1))
        self.fc1.weight.data.uniform_(-stdv1, stdv1)
        self.fc1.bias.data.uniform_(-stdv1, stdv1)
        
        stdv2 = 1. / math.sqrt(self.fc2.weight.size(1))
        self.fc2.weight.data.uniform_(-stdv2, stdv2)
        self.fc2.bias.data.uniform_(-stdv2, stdv2)

    def forward(self, x, adj=None):
        # adj is ignored, proving performance comes strictly from node features
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
