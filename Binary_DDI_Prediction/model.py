import math
from typing import Hashable
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utils import get_adj_mat,load_feat,normalize_adj,preprocess_adj,get_train_test_set
import sys
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from time import gmtime, strftime,localtime
import numpy as np
from torch_geometric.nn import GCNConv,GATConv
from utils import get_adj_mat

'''Model definition'''


# Graph Attention network
class GAT(torch.nn.Module):
    def __init__(self,num_nodes,num_outputs,num_hidden,init_feat=64):
        super(GAT,self).__init__()
        self.X = nn.Parameter(torch.randn((num_nodes,init_feat)),requires_grad=True) 
        nn.init.xavier_uniform_(self.X) # Initialize node feature matrix with xavier initialization
        #Define two conv layers
        self.conv1 = GATConv(init_feat,num_hidden,dropout=0.5,heads=2,concat=False)
        self.conv2 = GATConv(num_hidden,num_outputs,dropout=0.5,heads=2,concat=False)
        
    def forward(self,edge_index):
        x = self.conv1(self.X,edge_index)
        
        x = self.conv2(x,edge_index)
       
        return x

#Generator definition with WGAN style
class Generator(nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super(Generator,self).__init__()
        self.l1 = nn.Linear(num_inputs,100)
        self.l2 = nn.Linear(100,num_outputs)
        self.relu = F.relu
        self.dropout = nn.Dropout()
        
    def forward(self,x):
        x = self.l1(x)
        
        x = self.l2(x)
      
        return x


class MLP(nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super(Generator,self).__init__()
        self.l1 = nn.Linear(num_inputs,100)
        self.l2 = nn.Linear(100,num_outputs)
        self.relu = F.relu
        self.dropout = nn.Dropout()
        
    def forward(self,x):
        x = self.l1(x)
        
        x = self.l2(x)
      
        return x

#WGAN Style discriminator
class Discriminator(nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super(Discriminator,self).__init__()
        self.l1 = nn.Linear(num_inputs,100)
        self.l2 = nn.Linear(100,num_outputs)
        self.relu = F.relu
        
    def forward(self,x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        
        y = x.mean(0)
        return y




class DNN(nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super(DNN,self).__init__()
        
        self.layers = nn.Sequential(nn.Linear(num_inputs,512),nn.ReLU(),nn.BatchNorm1d(512),nn.Dropout(0.3),
                      nn.Linear(512,256),nn.ReLU(),nn.BatchNorm1d(256),nn.Dropout(0.3),
                      nn.Linear(256,num_outputs)
                      )
    def forward(self,x):
        output = torch.sigmoid(self.layers(x))
        return output


class Model(nn.Module):
    def __init__(self,num_nodes,gcn_outputs,gcn_hidden,num_outputs,attr_dim):
        super(Model,self).__init__()
        # self.GCN = GCN(num_nodes,gcn_outputs,gcn_hidden)
        self.encoder = GAT(num_nodes,gcn_outputs,gcn_hidden) #Topology structure encoder with GAT       
        self.g_t = MLP(gcn_outputs,num_outputs) #Projecting structural embedding to the common space
        self.g_t2a = Generator(num_outputs,attr_dim) #Generator for structure to attribute
        self.g_a = MLP(attr_dim,num_outputs) #MLP to encoding drug attribute
        self.g_a2t = Generator(num_outputs,gcn_outputs) #Generator for attribute to structure
        self.classifier = DNN(num_outputs*4,1)
        
    def forward(self,edge_index,attr_mtx,x,*args):
        
        gcn_out = self.encoder(edge_index)
        topo_emd = self.g_t(gcn_out)
        t2a_mtx = self.g_t2a(topo_emd)
        attr_emd = self.g_a(attr_mtx)
        a2t_mtx = self.g_a2t(attr_emd)
     
        
        #Concatnation
        embedding = torch.hstack([topo_emd,attr_emd])
        self.h_t = topo_emd
        self.h_a = attr_emd
        self.topo_emb = gcn_out
        self.attr_emb = attr_mtx
        self.false_topo = a2t_mtx
        self.false_attr = t2a_mtx
        
        x = x.long()
       

        X = torch.hstack([embedding[x[:,0]],embedding[x[:,1]]])
        output = self.classifier(X)
        return output,gcn_out,t2a_mtx,a2t_mtx



def get_edge_index(edge_list):
    edge_list_image = edge_list[:,[1,0]]
    edge_list_image
    edge_index = np.vstack([edge_list,edge_list_image])
    edge_index = torch.tensor(edge_index,dtype=torch.long)

    return edge_index.t().contiguous()
