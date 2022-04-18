import math
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
import os

from model import Discriminator,Model,Generator,get_edge_index
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,auc,precision_recall_curve
from functools import reduce
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_topo_feats', type=int, default=80, help='dim of topology features')
parser.add_argument('--n_hid', type=int, default=256, help='num of hidden features')
parser.add_argument('--n_out_feat', type=int, default=128, help='num of output features')
parser.add_argument('--n_epochs', type=int, default=100, help='num of epochs')
parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
args = parser.parse_args()


n_topo_feats = args.n_topo_feats
n_hid = args.n_hid
n_out_feat = args.n_out_feat
n_epochs = args.n_epochs
batch_size = args.batch_size

print(os.path.basename(sys.argv[0]))
print("embedding size: ",n_out_feat)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)    
    

edges_all,num_nodes = get_adj_mat()
adj_mat, train_edges_true, train_edges_false, test_edges_true, test_edges_false=get_train_test_set(edges_all[:],num_nodes,0.2)
train_true_labels = np.ones(train_edges_true.shape[0],dtype=np.float32)
train_false_labels = np.zeros(train_edges_false.shape[0],dtype=np.float32)
test_true_labels = np.ones(test_edges_true.shape[0],dtype=np.float32)
test_false_labels = np.zeros(test_edges_false.shape[0],dtype=np.float32)
adj_mat = preprocess_adj(adj_mat)

adj_mat = sparse_mx_to_torch_sparse_tensor(adj_mat)
feat_mat = load_feat('./drug_sim.csv')

print('shape of initial feature matrix',feat_mat.shape)


train_feat_mat = feat_mat
train_feat_mat = torch.FloatTensor(train_feat_mat)
train_feat_mat = train_feat_mat.to(device)
train_edges = np.vstack([train_edges_true,train_edges_false])
y_train = np.concatenate([train_true_labels,train_false_labels])
test_edges = np.vstack([test_edges_true,test_edges_false])
y_test = np.concatenate([test_true_labels,test_false_labels])
print('number of training samples:',y_train.shape)
print('number of test samples:',y_test.shape)
class DDIDataset(Dataset):
    def __init__(self,x,y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.n_samples = self.x.shape[0]
    
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    
    def __len__(self):
        return self.n_samples
train_dataset = DDIDataset(train_edges,y_train)
test_dataset = DDIDataset(test_edges,y_test)
train_loader = DataLoader(dataset=train_dataset,shuffle=True,batch_size=batch_size)
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size)

def update_E(net_E,edge_list,feat_mat,batch_edges,labels,D_t2a,D_a2t,loss,trainer_E,device):
    edge_index = get_edge_index(edge_list).to(device)

    y_pred,gcn_out,t2a_mtx,a2t_mtx = net_E(edge_index,feat_mat,batch_edges)
    # y_pred = y_pred.reshape(-1)
   
    trainer_E.zero_grad()

    # ones = torch.ones(gcn_out.shape[0]).to(device)

    # zeros = torch.zeros(a2t_mtx.shape[0]).to(device)    
    one = torch.FloatTensor([1]).to(device)
    # one = torch.FloatTensor([1])
    mone = (one * -1).to(device)
    D_a2t.eval()
    D_t2a.eval()
    fake_y_a2t = D_a2t(a2t_mtx)
    fake_y_a2t.backward(one)
    # trainer_E.step()

    # trainer_E.zero_grad()
    fake_y_t2a = D_t2a(t2a_mtx)
    fake_y_t2a.backward(one)
    trainer_E.step()
    trainer_E.zero_grad()
    # trainer_E.step()
    # model_loss = loss(y_pred,labels)+loss(fake_y_a2t,ones.reshape(fake_y_a2t.shape))+\
    #              loss(fake_y_t2a,ones.reshape(fake_y_t2a.shape))
    y_pred,gcn_out,t2a_mtx,a2t_mtx = net_E(edge_index,feat_mat,batch_edges)
    y_pred = y_pred.reshape(-1)
    model_loss = loss(y_pred,labels)
    model_loss.backward(retain_graph=True)


   
    # trainer_E.step()
    trainer_E.step()
    return y_pred,model_loss

def update_D_t2a(net_E,edge_list,feat_mat,batch_edges,D_t2a,loss,trainer_D_t2a,device):
    edge_index = get_edge_index(edge_list).to(device)

    # _,_,t2a_mtx,_ = net_E(edge_index,feat_mat,batch_edges,False)
    clamp_lower = -0.01
    clamp_upper = 0.01
    for p in D_t2a.parameters():
        p.data.clamp_(clamp_lower, clamp_upper)
    trainer_D_t2a.zero_grad()
    # ones = torch.ones(feat_mat.shape[0]).to(device)
    # zeros = torch.zeros(t2a_mtx.shape[0]).to(device)
    one = torch.FloatTensor([1]).to(device)
    mone = (one * -1).to(device)
    net_E.eval()
    _,_,t2a_mtx,_ = net_E(edge_index,feat_mat,batch_edges)
    fake_y = D_t2a(t2a_mtx)
    fake_y.backward(mone)

    real_y = D_t2a(feat_mat)
    real_y.backward(one)
    # loss_D = (nn.BCELoss()(real_y, ones) +
    #           nn.BCELoss()(fake_y, zeros)) / 2

    # loss_D.backward()
    trainer_D_t2a.step()
    return 

def update_D_a2t(net_E,edge_list,feat_mat,batch_edges,D_a2t,loss,trainer_D_a2t,device):
    edge_index = get_edge_index(edge_list).to(device)
    # _,gcn_out,_,a2t_mtx = net_E(edge_index,feat_mat,batch_edges,False)
    clamp_lower = -0.01
    clamp_upper = 0.01
    for p in D_a2t.parameters():
        p.data.clamp_(clamp_lower, clamp_upper)

    trainer_D_a2t.zero_grad()
    # ones = torch.ones(a2t_mtx.shape[0]).to(device)
    # zeros = torch.zeros(feat_mat.shape[0]).to(device)
    one = torch.FloatTensor([1]).to(device)
    mone = (one * -1).to(device)
    net_E.eval()
    _,gcn_out,_,a2t_mtx = net_E(edge_index,feat_mat,batch_edges)
    fake_y = D_a2t(a2t_mtx)
    # fake_y = fake_y.view(-1)
    fake_y.backward(mone)

    real_y = D_a2t(gcn_out)
    # real_y = real_y.view(-1)
    real_y.backward(one)
    # loss_D = (nn.BCELoss()(real_y, ones) +
            #   nn.BCELoss()(fake_y, zeros)) / 2

    # loss_D.backward()
    trainer_D_a2t.step()
    return 


model = Model(num_nodes,n_topo_feats,n_hid,n_out_feat,feat_mat.shape[1])
# list(model.named_parameters())
D_t2a = Discriminator(feat_mat.shape[1],1)
D_a2t = Discriminator(n_topo_feats,1)
model.to(device)
D_t2a.to(device)
D_a2t.to(device)
criterion = nn.BCELoss().to(device)

trainer_E = torch.optim.RMSprop(model.parameters(),lr=1e-3)
trainer_D_t2a = torch.optim.RMSprop(D_t2a.parameters(),lr=1e-3)
trainer_D_a2t = torch.optim.RMSprop(D_a2t.parameters(),lr=1e-3)
# step_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=2,gamma=0.5)

n_iterations = len(train_loader)
num_epochs = n_epochs
start = time.time()
running_loss = 0.0
running_correct = 0.0
for epoch in range(num_epochs):
    model.train()
    true_labels,pred_labels = [],[]
    running_loss = 0.0
    running_correct = 0.0
    total_samples = 0
    for i,(edges,labels) in enumerate(train_loader):
#         print(train_edges_true.shape)
        edges,labels = edges.to(device),labels.to(device)
        y_pred,loss = update_E(model,train_edges_true,train_feat_mat,edges,labels,D_t2a,D_a2t,criterion,trainer_E,device)
        update_D_t2a(model,train_edges_true,train_feat_mat,edges,D_t2a,loss,trainer_D_t2a,device)
        update_D_a2t(model,train_edges_true,train_feat_mat,edges,D_a2t,loss,trainer_D_a2t,device)
        pred_labels.append(list(y_pred.cpu().detach().numpy().reshape(-1)))
        y_pred = y_pred.cpu().detach().numpy().reshape(-1).round()
        labels = labels.cpu().numpy()
        total_samples += labels.shape[0]
        true_labels.append(list(labels))
        running_loss += loss.item()
        running_correct += (y_pred == labels).sum().item()

    print(f"epoch {epoch+1}/{num_epochs};trainging loss: {running_loss/n_iterations:.4f}")
    print(f"epoch {epoch+1}/{num_epochs};training set acc: {running_correct/total_samples:.4f}")
    def merge(x,y):
        return x + y
    
    true_labels = reduce(merge,true_labels)
    pred_labels = reduce(merge,pred_labels)
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    # print(true_labels.shape)
    lr_precision, lr_recall, _ = precision_recall_curve(true_labels, pred_labels)
    aupr = auc(lr_recall, lr_precision)
    auroc = roc_auc_score(true_labels,pred_labels)

end = time.time()
elapsed = end-start
print(f"Training completed in {elapsed//60}m: {elapsed%60:.2f}s.")

n_test_samples = 0
n_correct = 0
total_labels = []
total_pred = []
with torch.no_grad():
    model.eval()
    for edges,labels in test_loader:

        edge_index = get_edge_index(train_edges_true).to(device)
        edges,labels = edges.to(device),labels.to(device)
        y_pred,_,_,_ = model(edge_index,train_feat_mat,edges,False)
        total_pred.append(y_pred.cpu().reshape(-1))
        y_pred = y_pred.cpu().numpy().reshape(-1).round()
        total_labels.append(labels.cpu())
        labels = labels.cpu().numpy()

        n_test_samples += edges.shape[0]
        n_correct += (y_pred == labels).sum()
#         print((y_pred == labels).sum())
    acc = 100.0 * n_correct/n_test_samples
    total_pred = torch.cat(total_pred)
    total_labels = torch.cat(total_labels)
    lr_precision, lr_recall, _ = precision_recall_curve(total_labels,total_pred)
    aupr = auc(lr_recall, lr_precision)
    auroc = roc_auc_score(total_labels,total_pred )
    print(f"test set accuracy: {acc}")
    print(f"AUPR: {aupr}")
    print(f"AUROC: {auroc}")


