import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv 
import argparse



class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))

    def sample_from_p(self, p):
        return F.relu(torch.sign(p - torch.rand(p.size())))

    def v_to_h(self, v):
        p_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return p_h, self.sample_from_p(p_h)

    def h_to_v(self, h):
        p_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return p_v, self.sample_from_p(p_v)

    def forward(self, v):
        if v.dim() > 2:
            v = v.view(v.size(0), -1)  
        _, h = self.v_to_h(v)
        return h

    def free_energy(self, v):
        v_term = torch.matmul(v, self.v_bias)
        wx_b = F.linear(v, self.W, self.h_bias)
        h_term = torch.sum(F.softplus(wx_b), dim=1)
        return -v_term - h_term



class DBN(nn.Module):
    def __init__(self, n_visible, n_hidden, n_classes, n_layers=2):
        super(DBN, self).__init__()
        self.rbm_layers = nn.ModuleList()
        #first RBM layer
        self.rbm_layers.append(RBM(n_visible, n_hidden))

        #subsequent RBM layers
        for i in range(1, n_layers):
            self.rbm_layers.append(RBM(n_hidden, n_hidden))
        #fully connected layer
        self.fc = nn.Linear(n_hidden, n_classes)

    

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # Flatten input

        #pass through each RBM layer
        for rbm in self.rbm_layers:
            x = rbm(x)

        #pass through fully connected layer
        return self.fc(x)




class MV_GRU(torch.nn.Module):
    def __init__(self, device, n_features, seq_length, hidden_dim=100, n_layers=2, num_class=2):
        super(MV_GRU, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.device = device
        self.n_hidden = hidden_dim  
        self.n_layers = n_layers  # number of GRU layers 
        
        self.gru = torch.nn.GRU(input_size=n_features, 
                                hidden_size=self.n_hidden, 
                                num_layers=self.n_layers, 
                                batch_first=True)
        
        self.linear = torch.nn.Linear(self.n_hidden * self.seq_len, num_class)

    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden, device=self.device)
        return hidden_state

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        h0 = self.init_hidden(batch_size)
        
        gru_out, _ = self.gru(x, h0)
        
        x = gru_out.contiguous().view(batch_size, -1)
        return self.linear(x)

class MV_RNN(torch.nn.Module):
    def __init__(self, device, n_features, seq_length, hidden_dim=100, n_layers=2, num_class=2):
        super(MV_RNN, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.device = device
        self.n_hidden = hidden_dim  
        self.n_layers = n_layers
        
        self.rnn = torch.nn.RNN(input_size=n_features, 
                                hidden_size=self.n_hidden, 
                                num_layers=self.n_layers, 
                                batch_first=True)
        
        
        self.linear = torch.nn.Linear(self.n_hidden * self.seq_len, num_class)

    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden, device=self.device)
        return hidden_state

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        h0 = self.init_hidden(batch_size)
        
        rnn_out, _ = self.rnn(x, h0)
        
        x = rnn_out.contiguous().view(batch_size, -1)
        
        return self.linear(x)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class MV_GCN(nn.Module):
    def __init__(self, n_features, seq_length, hidden_dim=64, num_class=2):
        super(MV_GCN, self).__init__()
        self.seq_length = seq_length
        self.n_features = n_features
        
        self.register_buffer('adj', self._create_adj_matrix(seq_length))
        
        self.gc1 = GraphConvolution(n_features, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * seq_length, num_class)

    def _create_adj_matrix(self, seq_length, window_size=5):
        adj = torch.zeros(seq_length, seq_length)
        for i in range(seq_length):
            for j in range(max(0, i-window_size), min(seq_length, i+window_size+1)):
                adj[i, j] = 1
        adj = adj / adj.sum(1, keepdim=True)
        return adj

    def forward(self, x):
        batch_size = x.size(0)
        
        x = F.relu(self.gc1(x, self.adj))
        x = self.dropout(x)
        x = F.relu(self.gc2(x, self.adj))
        
        x = x.view(batch_size, -1)
        x = self.fc(x)
        
        return x

# class MV_GCN(nn.Module):
#     def __init__(self, n_features, seq_length, hidden_dim=64, num_class=2, num_layers=2):
#         super(MV_GCN, self).__init__()
#         self.seq_length = seq_length
#         self.n_features = n_features
#         self.register_buffer('adj', self._create_adj_matrix(seq_length))
        
#         self.layers = nn.ModuleList()
#         self.layers.append(GraphConvolution(n_features, hidden_dim))
#         for _ in range(num_layers - 2):
#             self.layers.append(GraphConvolution(hidden_dim, hidden_dim))
#         self.layers.append(GraphConvolution(hidden_dim, hidden_dim))
        
#         self.dropout = nn.Dropout(0.5)
#         self.fc = nn.Linear(hidden_dim * seq_length, num_class)

#     def forward(self, x):
#         batch_size = x.size(0)
#         for layer in self.layers[:-1]:
#             x = F.relu(layer(x, self.adj))
#             x = self.dropout(x)
#         x = F.relu(self.layers[-1](x, self.adj))
#         x = x.view(batch_size, -1)
#         x = self.fc(x)
#         return x