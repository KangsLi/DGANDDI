
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.optim import lr_scheduler
from utils import get_adj_mat,load_feat,normalize_adj,preprocess_adj,get_train_test_set,get_full_dataset
import sys
import time
import os
# from model import DDIGCN,Discriminator,Model,Generator,get_edge_index,GCN
from model_concat import Discriminator,Model,Generator,get_edge_index,GCN
# from model_gcn import Discriminator,Model,Generator,get_edge_index,GCN
from sklearn.neighbors import KernelDensity
import random
from sklearn.manifold import TSNE

from sklearn.metrics import roc_auc_score,auc,precision_recall_curve
from functools import reduce
import pandas as pd
import argparse
from sklearn.model_selection import StratifiedKFold


'''Code for 5 fold cross-validation'''
parser = argparse.ArgumentParser()
parser.add_argument('--n_topo_feats', type=int, default=80, help='dim of topology features')
parser.add_argument('--n_hid', type=int, default=256, help='dim of hidden features in ')
parser.add_argument('--n_out_feat', type=int, default=128, help='num of output features')
parser.add_argument('--n_epochs', type=int, default=100, help='num of epochs')
parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
parser.add_argument('--gpu_id', type=str,default='0', help='if -1, use cpu')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

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
    

true_edges,num_nodes = get_adj_mat()


all_edges,all_labels = get_full_dataset() #Get all DDIs in dataset


feat_mat = load_feat('./drug_sim.csv') #Load drug attribute file

print('shape of initial feature matrix',feat_mat.shape)


train_feat_mat = feat_mat
train_feat_mat = torch.FloatTensor(train_feat_mat) 
train_feat_mat = train_feat_mat.to(device)

class DDIDataset(Dataset):
    '''Customized dataset processing class'''
    def __init__(self,x,y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.n_samples = self.x.shape[0]
    
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    
    def __len__(self):
        return self.n_samples


def update_E(net_E,edge_list,feat_mat,batch_edges,labels,D_t2a,D_a2t,loss,trainer_E,device):

    '''This function mainly used to optimize parameters of encoder'''
    edge_index = get_edge_index(edge_list).to(device)

    y_pred,gcn_out,t2a_mtx,a2t_mtx = net_E(edge_index,feat_mat,batch_edges,False)
   
    trainer_E.zero_grad()
   
    one = torch.FloatTensor([1]).to(device)  
    mone = (one * -1).to(device)
    D_a2t.eval()
    D_t2a.eval()
    fake_y_a2t = D_a2t(a2t_mtx)
    fake_y_a2t.backward(one)
    

   
    fake_y_t2a = D_t2a(t2a_mtx)
    fake_y_t2a.backward(one)
    trainer_E.step()
    trainer_E.zero_grad()
  
    y_pred,gcn_out,t2a_mtx,a2t_mtx = net_E(edge_index,feat_mat,batch_edges,False)
    y_pred = y_pred.reshape(-1)
    model_loss = loss(y_pred,labels)
    model_loss.backward()

    trainer_E.step()
    return y_pred,model_loss

def update_D_t2a(net_E,edge_list,feat_mat,batch_edges,D_t2a,loss,trainer_D_t2a,device):


    '''This function mainly used to optimize parameters of discriminator for topology-to-attribute'''
    edge_index = get_edge_index(edge_list).to(device)

   
    clamp_lower = -0.01
    clamp_upper = 0.01
    for p in D_t2a.parameters():
        p.data.clamp_(clamp_lower, clamp_upper)
    trainer_D_t2a.zero_grad()
    
    one = torch.FloatTensor([1]).to(device)
    mone = (one * -1).to(device)
    net_E.eval()
    _,_,t2a_mtx,_ = net_E(edge_index,feat_mat,batch_edges,False)
    fake_y = D_t2a(t2a_mtx)
    fake_y.backward(mone)

    real_y = D_t2a(feat_mat)
    real_y.backward(one)

    trainer_D_t2a.step()
    return 

def update_D_a2t(net_E,edge_list,feat_mat,batch_edges,D_a2t,loss,trainer_D_a2t,device):

    '''This function mainly used to optimize parameters of discriminator for attribute-to-topology'''
    edge_index = get_edge_index(edge_list).to(device)
   
    clamp_lower = -0.01
    clamp_upper = 0.01
    for p in D_a2t.parameters():
        p.data.clamp_(clamp_lower, clamp_upper)

    trainer_D_a2t.zero_grad()

    one = torch.FloatTensor([1]).to(device)
    mone = (one * -1).to(device)
    net_E.eval()
    _,gcn_out,_,a2t_mtx = net_E(edge_index,feat_mat,batch_edges,False)
    fake_y = D_a2t(a2t_mtx)
   
    fake_y.backward(mone)

    real_y = D_a2t(gcn_out)
  
    real_y.backward(one)
   
    trainer_D_a2t.step()
    return 

results = []  #Record the prediction results for each fold


skf = StratifiedKFold(n_splits=5) #Stratified split policy is adopted

'''Perform 5 fold cross-validation'''
for k,(train_index,test_index) in enumerate(skf.split(all_edges,all_labels)):
    edges_train,edges_test = all_edges[train_index],all_edges[test_index]
    y_train,y_test = all_labels[train_index],all_labels[test_index]
    train_edges_true = edges_train[y_train==1]
   
    train_dataset = DDIDataset(edges_train,y_train.astype(np.float32))
    test_dataset = DDIDataset(edges_test,y_test)
    train_loader = DataLoader(dataset=train_dataset,shuffle=True,batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size)
    model = Model(num_nodes,n_topo_feats,n_hid,n_out_feat,feat_mat.shape[1],alpha=0.6)
    
    D_t2a = Discriminator(feat_mat.shape[1],1)
    D_a2t = Discriminator(n_topo_feats,1)
    model.to(device)
    D_t2a.to(device)
    D_a2t.to(device)
    criterion = nn.BCELoss().to(device)

    trainer_E = torch.optim.RMSprop(model.parameters(),lr=1e-3)
    trainer_D_t2a = torch.optim.RMSprop(D_t2a.parameters(),lr=1e-3)
    trainer_D_a2t = torch.optim.RMSprop(D_a2t.parameters(),lr=1e-3)
    

    n_iterations = len(train_loader)
    num_epochs = 30
    start = time.time()

    running_loss = 0.0
    running_correct = 0.0
    # Training phase
    for epoch in range(num_epochs):
        model.train()
        true_labels,pred_labels = [],[]
        running_loss = 0.0
        running_correct = 0.0
        total_samples = 0
        for i,(edges,labels) in enumerate(train_loader):
 
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
    fold_results = []
    # Testing phase
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

        # Calculate evaluation indexes
        acc = 100.0 * n_correct/n_test_samples
        total_pred = torch.cat(total_pred)
        total_labels = torch.cat(total_labels)       
        lr_precision, lr_recall, _ = precision_recall_curve(total_labels,total_pred)
        aupr = auc(lr_recall, lr_precision)
        auroc = roc_auc_score(total_labels,total_pred )
        fold_results.append(k)
        fold_results.append(acc)
        fold_results.append(aupr)
        fold_results.append(auroc)
        print(f"test set accuracy: {acc}")
        print(f"AUPR: {aupr}")
        print(f"AUROC: {auroc}")
    results.append(fold_results)
  

#Write cross-validation results to file
pd.DataFrame(results,columns=['fold','acc','aupr','auroc']).to_csv('cross_validation.csv')