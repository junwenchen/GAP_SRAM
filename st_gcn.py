import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from utils import *
#from net.utils.tgcn import ConvTemporalGraphical
#from net.utils.graph import Graph
from tgcn import ConvTemporalGraphical

import pdb

import math

from torch.nn.parameter import Parameter

class ConvGraphicalDecoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 device=None):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(
            in_channels,
            in_channels * kernel_size,
            kernel_size=1,
            padding=0,
            stride=1,
            dilation=1,
            bias=bias)

        self.conv2 = nn.Conv1d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=1,
            padding=0,
            stride=1,
            dilation=1,
            bias=bias)

    def forward(self, latent, x, ra):
        x = self.conv1((x+latent).permute(0,2,1))
        x = torch.matmul(ra, x.permute(0,2,1)) #+ torch.matmul(rp, x.permute(0,2,1))
        x = self.conv2(x.permute(0,2,1))

        return x.contiguous()

class ConvGraphicalEncoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 device=None):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=1,
            padding=0,
            stride=1,
            dilation=1,
            bias=bias)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels * kernel_size,
            kernel_size=1,
            padding=0,
            stride=1,
            dilation=1,
            bias=bias)

    def forward(self, x, ra):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.matmul(ra, x.permute(0,2,1)) #+ torch.matmul(rp, x.permute(0,2,1))

        return x.contiguous()

class st_gcn_decoder(nn.Module):

    def __init__(self,
                 cfg,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.cfg = cfg
        NG = 1
        NFG = 1024
        NFR = 1024
        NFG_ONE = 1024
        N = 12
        self.fc_rn_theta_list=torch.nn.ModuleList([ nn.Linear(NFG,NFR) for i in range(NG) ])
        self.fc_rn_phi_list=torch.nn.ModuleList([ nn.Linear(NFG,NFR) for i in range(NG) ])
        self.fc_gcn_list=torch.nn.ModuleList([ nn.Linear(NFG,NFG_ONE,bias=False) for i in range(NG) ])
        # self.pos_fc_gcn_list=torch.nn.ModuleList([ nn.Linear(NFG,NFG_ONE,bias=False) for i in range(NG) ])
        self.nl_gcn_list=torch.nn.ModuleList([ nn.LayerNorm([N,NFG_ONE]) for i in range(NG) ])
        # self.pos_nl_gcn_list=torch.nn.ModuleList([ nn.LayerNorm([N,NFG_ONE]) for i in range(NG) ])

    def forward(self, latent_t, latent_x, relation_graph):
        graph_boxes_features = latent_t+latent_x
        # one_pos_graph_boxes_features=self.pos_fc_gcn_list[0](torch.matmul(pos_relation_graph,graph_boxes_features))  #B, N, NFG_ONE
        # one_pos_graph_boxes_features=self.pos_nl_gcn_list[0](one_pos_graph_boxes_features)
        # one_pos_graph_boxes_features=F.relu(one_pos_graph_boxes_features)
        # Graph convolution
        one_sim_graph_boxes_features=self.fc_gcn_list[0](torch.matmul(relation_graph,graph_boxes_features))  #B, N, NFG_ONE
        one_sim_graph_boxes_features=self.nl_gcn_list[0](one_sim_graph_boxes_features)
        one_sim_graph_boxes_features=F.relu(one_sim_graph_boxes_features)

        # graph_boxes_features=one_sim_graph_boxes_features

        return one_sim_graph_boxes_features.permute(0,2,1)

class st_gcn_short(nn.Module):

    def __init__(self,
                 cfg,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.cfg = cfg

        self.relu = nn.ReLU(inplace=True)
        NG = 1
        NFG = 1024
        NFR = 1024
        NFG_ONE = 1024
        N = 12
        self.fc_rn_theta_list=torch.nn.ModuleList([ nn.Linear(NFG,NFR) for i in range(NG) ])
        self.fc_rn_phi_list=torch.nn.ModuleList([ nn.Linear(NFG,NFR) for i in range(NG) ])
        
        self.fc_gcn_list=torch.nn.ModuleList([ nn.Linear(NFG,NFG_ONE,bias=False) for i in range(NG) ])
        # self.pos_fc_gcn_list=torch.nn.ModuleList([ nn.Linear(NFG,NFG_ONE,bias=False) for i in range(NG) ])

        self.nl_gcn_list=torch.nn.ModuleList([ nn.LayerNorm([N,NFG_ONE]) for i in range(NG) ])
        # self.pos_nl_gcn_list=torch.nn.ModuleList([ nn.LayerNorm([N,NFG_ONE]) for i in range(NG) ])

    def forward(self, x, A):
        # GCN graph modeling
        # Prepare boxes similarity relation
        boxes_in_flat = A
        B, C, T, N = x.shape
        graph_boxes_positions=boxes_in_flat.contiguous().view(-1,4)  #B*T*N, 4
        graph_boxes_positions[:,0]=(graph_boxes_positions[:,0] + graph_boxes_positions[:,2]) / 2
        graph_boxes_positions[:,1]=(graph_boxes_positions[:,1] + graph_boxes_positions[:,3]) / 2
        graph_boxes_positions=graph_boxes_positions[:,:2].reshape(B,T,N,2)  #B,T, N, 2
        graph_boxes_distances=calc_pairwise_distance_3d(graph_boxes_positions.reshape(B*T,N,2),graph_boxes_positions.reshape(B*T,N,2))  #B*T, N, N
        OH, OW=self.cfg.out_size
        pos_threshold=self.cfg.pos_threshold
        position_mask=( graph_boxes_distances > (pos_threshold*OW) )
        graph_boxes_features = x

        graph_boxes_features_theta=self.fc_rn_theta_list[0](graph_boxes_features.permute(0,2,3,1).contiguous().view(B*T,N,-1))  #B,N,NFR
        graph_boxes_features_phi=self.fc_rn_phi_list[0](graph_boxes_features.permute(0,2,3,1).contiguous().view(B*T,N,-1))  #B,N,NFR
        similarity_relation_graph=torch.matmul(graph_boxes_features_theta,graph_boxes_features_phi.transpose(1,2))  #B,N,N
        NFR = graph_boxes_features_phi.shape[-1]
        similarity_relation_graph=similarity_relation_graph/np.sqrt(NFR)
        similarity_relation_graph=similarity_relation_graph.reshape(-1,1)  #B*N*N, 1
        # Build relation graph for similarity 
        relation_graph=similarity_relation_graph
        relation_graph = relation_graph.reshape(B*T,N,N)
        relation_graph[position_mask]=-float('inf')
        relation_graph = torch.softmax(relation_graph,dim=2)

        # # # Build relation graph for position
        # pos_relation_graph_sum = torch.sum(graph_boxes_distances, dim=2)
        # pos_relation_graph = pos_relation_graph_sum.unsqueeze(2)-graph_boxes_distances
        # # pos_relation_graph[position_mask]=-float('inf')
        # pos_relation_graph = pos_relation_graph/(pos_relation_graph.sum(2).unsqueeze(2))
        # # pos_relation_graph = torch.softmax(pos_relation_graph,dim=2)

        # Graph convolution
        one_sim_graph_boxes_features=self.fc_gcn_list[0](torch.matmul(relation_graph,graph_boxes_features.permute(0,2,3,1).contiguous().view(B*T,N,-1)))  #B, N, NFG_ONE
        one_sim_graph_boxes_features=self.nl_gcn_list[0](one_sim_graph_boxes_features)
        one_sim_graph_boxes_features=F.relu(one_sim_graph_boxes_features)

        # one_pos_graph_boxes_features=self.pos_fc_gcn_list[0](torch.matmul(pos_relation_graph,graph_boxes_features.permute(0,2,3,1).contiguous().view(B*T,N,-1)))  #B, N, NFG_ONE
        # one_pos_graph_boxes_features=self.pos_nl_gcn_list[0](one_pos_graph_boxes_features)
        # one_pos_graph_boxes_features=F.relu(one_pos_graph_boxes_features)

        # one_graph_boxes_features = (one_sim_graph_boxes_features + one_pos_graph_boxes_features)/2
        one_graph_boxes_features = one_sim_graph_boxes_features

        # graph_boxes_features=torch.sum(torch.stack(graph_boxes_features_list),dim=0) #B*T, N, NFG=256
        return one_graph_boxes_features, relation_graph

class st_gcn(nn.Module):

    def __init__(self,
                 cfg,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.cfg = cfg

        self.tcn = nn.Sequential(
           nn.ReLU(inplace=True),
           nn.Conv2d(
               in_channels,
               in_channels,
               (kernel_size[0], 1),
               (stride, 1),
               padding,
           ),
           nn.Dropout(dropout, inplace=True),
        )

        self.relu = nn.ReLU(inplace=True)
        NG = 1
        NFG = 1024
        NFR = 1024
        NFG_ONE = 1024
        N = 12
        self.fc_rn_theta_list=torch.nn.ModuleList([ nn.Linear(NFG,NFR) for i in range(NG) ])
        self.fc_rn_phi_list=torch.nn.ModuleList([ nn.Linear(NFG,NFR) for i in range(NG) ])
        
        self.fc_gcn_list=torch.nn.ModuleList([ nn.Linear(NFG,NFG_ONE,bias=False) for i in range(NG) ])
        self.pos_fc_gcn_list=torch.nn.ModuleList([ nn.Linear(NFG,NFG_ONE,bias=False) for i in range(NG) ])

        self.nl_gcn_list=torch.nn.ModuleList([ nn.LayerNorm([N,NFG_ONE]) for i in range(NG) ])
        self.pos_nl_gcn_list=torch.nn.ModuleList([ nn.LayerNorm([N,NFG_ONE]) for i in range(NG) ])

    def forward(self, x, A):
        # GCN graph modeling
        # Prepare boxes similarity relation
        boxes_in_flat = A
        B, C, T, N = x.shape
        graph_boxes_positions=boxes_in_flat.contiguous().view(-1,4)  #B*T*N, 4
        graph_boxes_positions[:,0]=(graph_boxes_positions[:,0] + graph_boxes_positions[:,2]) / 2
        graph_boxes_positions[:,1]=(graph_boxes_positions[:,1] + graph_boxes_positions[:,3]) / 2
        graph_boxes_positions=graph_boxes_positions[:,:2].reshape(B,T,N,2)  #B,T, N, 2
        graph_boxes_distances=calc_pairwise_distance_3d(graph_boxes_positions.reshape(B*T,N,2),graph_boxes_positions.reshape(B*T,N,2))  #B*T, N, N

        OH, OW=self.cfg.out_size
        pos_threshold=self.cfg.pos_threshold
        position_mask=( graph_boxes_distances > (pos_threshold*OW) )

        graph_boxes_features = x
        graph_boxes_features_theta=self.fc_rn_theta_list[0](graph_boxes_features.permute(0,2,3,1).contiguous().view(B*T,N,-1))  #B,N,NFR
        graph_boxes_features_phi=self.fc_rn_phi_list[0](graph_boxes_features.permute(0,2,3,1).contiguous().view(B*T,N,-1))  #B,N,NFR
        similarity_relation_graph=torch.matmul(graph_boxes_features_theta,graph_boxes_features_phi.transpose(1,2))  #B,N,N
        NFR = graph_boxes_features_phi.shape[-1]
        similarity_relation_graph=similarity_relation_graph/np.sqrt(NFR)
        similarity_relation_graph=similarity_relation_graph.reshape(-1,1)  #B*N*N, 1
        # Build relation graph
        relation_graph=similarity_relation_graph
        relation_graph = relation_graph.reshape(B*T,N,N)
        relation_graph[position_mask]=-float('inf')
        relation_graph = torch.softmax(relation_graph,dim=2)

        # # # Build relation graph
        # pos_relation_graph_sum = torch.sum(graph_boxes_distances, dim=2)
        # pos_relation_graph = pos_relation_graph_sum.unsqueeze(2)-graph_boxes_distances
        # pos_relation_graph = pos_relation_graph/(pos_relation_graph.sum(2).unsqueeze(2))
        # one_pos_graph_boxes_features=self.pos_fc_gcn_list[0](torch.matmul(pos_relation_graph,graph_boxes_features.permute(0,2,3,1).contiguous().view(B*T,N,-1)))  #B, N, NFG_ONE
        # one_pos_graph_boxes_features=self.pos_nl_gcn_list[0](one_pos_graph_boxes_features)
        # one_pos_graph_boxes_features=F.relu(one_pos_graph_boxes_features)
        

        # Graph convolution
        one_sim_graph_boxes_features=self.fc_gcn_list[0](torch.matmul(relation_graph,graph_boxes_features.permute(0,2,3,1).contiguous().view(B*T,N,-1)))  #B, N, NFG_ONE
        one_sim_graph_boxes_features=self.nl_gcn_list[0](one_sim_graph_boxes_features)
        one_sim_graph_boxes_features=F.relu(one_sim_graph_boxes_features)

        one_graph_boxes_features = one_sim_graph_boxes_features
        # one_graph_boxes_features = one_sim_graph_boxes_features  +one_pos_graph_boxes_features
        one_graph_boxes_features = one_graph_boxes_features.view(B,T,N,-1).permute(0,3,1,2)
        one_graph_boxes_features = self.tcn(one_graph_boxes_features)  #.permute(0,2,3,1)

        return graph_boxes_features.mean(2).permute(0,2,1)




