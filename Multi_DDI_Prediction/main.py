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
import warnings
from model import Discriminator,Model,Generator,get_edge_index
import random
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,auc,precision_recall_curve,f1_score,precision_score,recall_score,accuracy_score
from functools import reduce
import pandas as pd
import argparse
from collections import defaultdict
import os

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

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)    
    

edges_all,num_nodes = get_adj_mat()
train_edges_true, train_edges_false, test_edges_true, test_edges_false=get_train_test_set(edges_all[:],num_nodes)
train_true_labels = np.ones(train_edges_true.shape[0],dtype=np.float32)
train_false_labels = np.zeros(train_edges_false.shape[0],dtype=np.float32)
test_true_labels = np.ones(test_edges_true.shape[0],dtype=np.float32)
test_false_labels = np.zeros(test_edges_false.shape[0],dtype=np.float32)
# adj_mat = preprocess_adj(adj_mat)

# adj_mat = sparse_mx_to_torch_sparse_tensor(adj_mat)
def get_train_test_set(ratio=0.2):
    # df_data = pd.read_csv('DDI.csv')
    df_data = pd.read_csv('DDI.csv')
    all_tup = [(h, t, r) for h, t, r in zip(df_data['drug_one'], df_data['drug_two'], df_data['type'])]
    np.random.shuffle(all_tup)
    all_edges = []
    all_labels = []
    for h,t,r in all_tup:
        all_edges.append((h,t))
        label = r-1
        all_labels.append(label)
    all_edges = np.array(all_edges)
    all_labels = np.array(all_labels)

    test_size = int(all_edges.shape[0]*ratio)
    test_edges = all_edges[0:test_size]
    test_labels = all_labels[0:test_size]    
    train_edges = all_edges[test_size:]
    train_labels = all_labels[test_size:]

    return train_edges,train_labels,test_edges,test_labels

def get_straitified_data(ratio=0.2):
    df_data = pd.read_csv('DDI.csv')
    all_tup = [(h, t, r) for h, t, r in zip(df_data['drug_one'], df_data['drug_two'], df_data['type'])]
    np.random.shuffle(all_tup)
    tuple_by_type = defaultdict(list)
    for h,t,r in all_tup:
        # all_edges.append((h,t))
        # label = r-1
        # all_labels.append(label)
        tuple_by_type[r-1].append((h,t,r-1))
    tuple_by_type.keys().__len__()
    train_edges = []
    test_edges = []
    for r in tuple_by_type.keys():
        test_edges.append(tuple_by_type[r][:int(len(tuple_by_type[r])*ratio)])
        train_edges.append(tuple_by_type[r][int(len(tuple_by_type[r])*ratio):])
    # len(test_edges)
    # print(len(train_edges))
    def merge(x,y):
            return x + y        
    test_tups = np.array(reduce(merge,test_edges))
    # len(test_edges)/len(all_tup)
    train_tups = np.array(reduce(merge,train_edges))
    # len(train_edges)/len(all_tup)
    train_edges = train_tups[:,:2]
    train_labels = train_tups[:,-1]
    test_edges = test_tups[:,:2]
    test_labels = test_tups[:,-1]
    return train_edges,train_labels,test_edges,test_labels


testset_ratio = 0.2
train_edges, train_labels, test_edges, test_labels = get_straitified_data(testset_ratio)
feat_mat = load_feat('./drug_sim.csv')


train_feat_mat = feat_mat
train_feat_mat = torch.FloatTensor(train_feat_mat)
train_feat_mat = train_feat_mat.to(device)


class DDIDataset(Dataset):
    def __init__(self,x,y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.n_samples = self.x.shape[0]
    
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    
    def __len__(self):
        return self.n_samples

train_dataset = DDIDataset(train_edges,train_labels)
test_dataset = DDIDataset(test_edges,test_labels)
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
  
    y_pred,gcn_out,t2a_mtx,a2t_mtx = net_E(edge_index,feat_mat,batch_edges)
    # y_pred = y_pred.reshape(-1)
    model_loss = loss(y_pred,labels)

    model_loss.backward()
   
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



model = Model(num_nodes,n_topo_feats,n_hid,n_out_feat,feat_mat.shape[1],n_rels=86)
# list(model.named_parameters())
D_t2a = Discriminator(feat_mat.shape[1],1)
D_a2t = Discriminator(n_topo_feats,1)
model.to(device)
D_t2a.to(device)
D_a2t.to(device)
# criterion = nn.BCELoss().to(device)
criterion = nn.CrossEntropyLoss().to(device)

trainer_E = torch.optim.RMSprop(model.parameters(),lr=1e-3)
trainer_D_t2a = torch.optim.RMSprop(D_t2a.parameters(),lr=1e-3)
trainer_D_a2t = torch.optim.RMSprop(D_a2t.parameters(),lr=1e-3)
# step_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=2,gamma=0.5)

n_iterations = len(train_loader)
num_epochs = 1
start = time.time()
writer = SummaryWriter("runs/drugbank")
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
        y_pred,loss = update_E(model,edges_all,train_feat_mat,edges,labels,D_t2a,D_a2t,criterion,trainer_E,device)
        update_D_t2a(model,edges_all,train_feat_mat,edges,D_t2a,loss,trainer_D_t2a,device)
        update_D_a2t(model,edges_all,train_feat_mat,edges,D_a2t,loss,trainer_D_a2t,device)
        running_correct += torch.sum((torch.argmax(y_pred, dim=1).type(torch.FloatTensor) == labels.cpu()).detach()).float()
        pred_labels.append(list(y_pred.cpu().detach().numpy().reshape(-1)))
        # y_pred = y_pred.cpu().detach().numpy().reshape(-1).round()
        labels = labels.cpu().numpy()
        total_samples += labels.shape[0]
        true_labels.append(list(labels))
        running_loss += loss.item()
        # running_correct += (y_pred == labels).sum().item()

    print(f"epoch {epoch+1}/{num_epochs};trainging loss: {running_loss/n_iterations:.4f}")
    print(f"epoch {epoch+1}/{num_epochs};training set acc: {running_correct/total_samples:.4f}")
    writer.add_scalar('training loss',running_loss/n_iterations,epoch)
    writer.add_scalar('training accuracy',running_correct/total_samples,epoch)
    def merge(x,y):
        return x + y
    
  

end = time.time()
elapsed = end-start
print(f"Training completed in {elapsed//60}m: {elapsed%60:.2f}s.")

def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        #order = np.lexsort((recall, precision))
        #precision1,recall1 = precision[order], recall[order]
        return auc(recall, precision)

    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)

    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)

event_num = 86
n_test_samples = 0
n_correct = 0
total_labels = []
total_pred = []
with torch.no_grad():
    model.eval()
    for edges,labels in test_loader:

        edge_index = get_edge_index(edges_all).to(device)
        edges,labels = edges.to(device),labels.to(device)
        y_pred,_,_,_ = model(edge_index,train_feat_mat,edges,False)
        y_hat = F.softmax(y_pred,dim=1)
        total_pred.append(y_hat.cpu().numpy())
        # total_pred.append(y_pred.cpu().reshape(-1))
        # y_pred = y_pred.cpu().numpy().reshape(-1).round()
        total_labels.append(labels.cpu().numpy())
        # labels = labels.cpu().numpy()

        n_test_samples += edges.shape[0]
        # n_correct += (y_pred == labels).sum()
        n_correct += torch.sum((torch.argmax(y_pred,dim=1).type(torch.FloatTensor) == labels.cpu()).detach()).float()
#         print((y_pred == labels).sum())
    acc = 100.0 * n_correct/n_test_samples
  
    total_pred = np.vstack(total_pred)
    total_labels = np.concatenate(total_labels)
    pred_type = np.argmax(total_pred,axis=1)
    y_one_hot = label_binarize(total_labels, np.arange(event_num))
    # pred_one_hot = label_binarize(pred_type, np.arange(event_num))
    # result_all['accuracy'] = accuracy_score(y_test, pred_type)
    aupr = roc_aupr_score(y_one_hot, total_pred, average='micro')
    auroc = roc_auc_score(y_one_hot, total_pred, average='micro')
    f1 = f1_score(total_labels,pred_type , average='macro')
    precision = precision_score(total_labels, pred_type, average='macro',zero_division=0)
    recall = recall_score(total_labels, pred_type, average='macro',zero_division=0)
    
    print(f"test set accuracy: {acc}")
    print(f"AUPR: {aupr}")
    print(f"AUROC: {auroc}")
    print(f"F1: {f1}")
    print(f"Precison: {precision}")
    print(f"Recall: {recall}")
   
  
