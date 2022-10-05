import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.optim import lr_scheduler
import numpy as np
from utils import get_adj_mat,load_feat,normalize_adj,preprocess_adj,get_train_test_set
import sys
import time
import warnings
from model import Discriminator,Model,Generator,get_edge_index,GCN
from sklearn.neighbors import KernelDensity
import random
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize

from sklearn.metrics import roc_auc_score,auc,precision_recall_curve,f1_score,precision_score,recall_score,accuracy_score
from functools import reduce
import pandas as pd
import argparse
from collections import defaultdict
import os


'''Code for 5 fold cross-validation'''

parser = argparse.ArgumentParser()
parser.add_argument('--n_topo_feats', type=int, default=80, help='dim of topology features')
parser.add_argument('--n_hid', type=int, default=256, help='num of hidden features')
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

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)    
    

edges_all,num_nodes = get_adj_mat() #Get all DDIs in dataset



def get_straitified_data(ratio=0.2):
    df_data = pd.read_csv('KnownDDI.csv')
    all_tup = [(h, t, r) for h, t, r in zip(df_data['drug_one'], df_data['drug_two'], df_data['type'])]
    np.random.shuffle(all_tup)
    tuple_by_type = defaultdict(list)
    for h,t,r in all_tup:
      
        tuple_by_type[r-1].append((h,t,r-1))
    tuple_by_type.keys().__len__()
    train_edges = []
    test_edges = []
    splits = []
 
    for k in range(0,(int)(1/ratio)):
        edges = []
        for r in tuple_by_type.keys():
            test_set_size = int(len(tuple_by_type[r])*ratio)
            if k<4:
                edges.append(tuple_by_type[r][k*test_set_size:(k+1)*test_set_size])
            else:
                edges.append(tuple_by_type[r][k*test_set_size:])
        splits.append(edges)
    return splits        
   
#Generate training and test set for each fold
def make_data(splits,fold_k):
    test_edges = splits[fold_k]
    train_edges =  []
    for i in range(0,len(splits)):
        if i == fold_k:
            continue
        train_edges += splits[i]
    def merge(x,y):
            return x + y        
    test_tups = np.array(reduce(merge,test_edges))
 
    train_tups = np.array(reduce(merge,train_edges))
    
    train_edges = train_tups[:,:2]
    train_labels = train_tups[:,-1]
    test_edges = test_tups[:,:2]
    test_labels = test_tups[:,-1]
    return train_edges,train_labels,test_edges,test_labels

testset_ratio = 0.2
splits = get_straitified_data(testset_ratio)

feat_mat = load_feat('./drug_sim.csv')
print(os.path.basename(sys.argv[0]))
print("embedding size: ",n_out_feat)
print("training set ratio:",(1-testset_ratio))
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


results = [] #Record the prediction results for each fold

'''Perform 5 fold cross-validation'''
for k in range(0,5):
    
   
    print("training on fold ",k)
    train_edges, train_labels, test_edges, test_labels = make_data(splits,k)
    train_dataset = DDIDataset(train_edges,train_labels)
    test_dataset = DDIDataset(test_edges,test_labels)
    train_loader = DataLoader(dataset=train_dataset,shuffle=True,batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size)  


    n_iterations = len(train_loader)
    num_epochs = n_epochs
    start = time.time()

    running_loss = 0.0
    running_correct = 0.0

    model = Model(num_nodes,n_topo_feats,n_hid,n_out_feat,feat_mat.shape[1],n_rels=86,alpha=0.6)
    D_t2a = Discriminator(feat_mat.shape[1],1)
    D_a2t = Discriminator(n_topo_feats,1)
    model.to(device)
    D_t2a.to(device)
    D_a2t.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    trainer_E = torch.optim.RMSprop(model.parameters(),lr=1e-3)
    trainer_D_t2a = torch.optim.RMSprop(D_t2a.parameters(),lr=1e-3)
    trainer_D_a2t = torch.optim.RMSprop(D_a2t.parameters(),lr=1e-3)
 

    n_iterations = len(train_loader)
    num_epochs = 1
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
            y_pred,loss = update_E(model,edges_all,train_feat_mat,edges,labels,D_t2a,D_a2t,criterion,trainer_E,device)
            update_D_t2a(model,edges_all,train_feat_mat,edges,D_t2a,loss,trainer_D_t2a,device)
            update_D_a2t(model,edges_all,train_feat_mat,edges,D_a2t,loss,trainer_D_a2t,device)
            running_correct += torch.sum((torch.argmax(y_pred, dim=1).type(torch.FloatTensor) == labels.cpu()).detach()).float()
            pred_labels.append(list(y_pred.cpu().detach().numpy().reshape(-1)))
            
            labels = labels.cpu().numpy()
            total_samples += labels.shape[0]
            true_labels.append(list(labels))
            running_loss += loss.item()
          

        print(f"epoch {epoch+1}/{num_epochs};trainging loss: {running_loss/n_iterations:.4f}")
        print(f"epoch {epoch+1}/{num_epochs};training set acc: {running_correct/total_samples:.4f}")
       
        def merge(x,y):
            return x + y
        
        

    end = time.time()
    elapsed = end-start
    print(f"Training completed in {elapsed//60}m: {elapsed%60:.2f}s.")

    def roc_aupr_score(y_true, y_score, average="macro"):
        def _binary_roc_aupr_score(y_true, y_score):
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            
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
    fold_results = []

    # Testing phase
    with torch.no_grad():
        model.eval()
        for edges,labels in test_loader:

            edge_index = get_edge_index(edges_all).to(device)
            edges,labels = edges.to(device),labels.to(device)
            y_pred,_,_,_ = model(edge_index,train_feat_mat,edges,False)
            y_hat = F.softmax(y_pred,dim=1)
            total_pred.append(y_hat.cpu().numpy())
          
            total_labels.append(labels.cpu().numpy())
           

            n_test_samples += edges.shape[0]
           
            n_correct += torch.sum((torch.argmax(y_pred,dim=1).type(torch.FloatTensor) == labels.cpu()).detach()).float()
  
        acc = 100.0 * n_correct/n_test_samples
    
        total_pred = np.vstack(total_pred)
        total_labels = np.concatenate(total_labels)
        pred_type = np.argmax(total_pred,axis=1)
        y_one_hot = label_binarize(total_labels, classes=np.arange(event_num))
      
        aupr = roc_aupr_score(y_one_hot, total_pred, average='micro')
        auroc = roc_auc_score(y_one_hot, total_pred, average='micro')
        f1 = f1_score(total_labels,pred_type , average='macro')
        precision = precision_score(total_labels, pred_type, average='macro',zero_division=0)
        recall = recall_score(total_labels, pred_type, average='macro',zero_division=0)
        f1_mi = f1_score(total_labels,pred_type,average='micro')
        precision_mi = precision_score(total_labels, pred_type, average='micro',zero_division=0)
        recall_mi = recall_score(total_labels, pred_type, average='micro',zero_division=0)
        fold_results.append(k)
        fold_results.append(acc)
        fold_results.append(aupr)
        fold_results.append(auroc)
        fold_results.append(f1)
        fold_results.append(precision)
        fold_results.append(recall)
        fold_results.append(f1_mi)
        fold_results.append(precision_mi)
        fold_results.append(recall_mi)
        print(f"test set accuracy: {acc}")
        print(f"AUPR: {aupr}")
        print(f"AUROC: {auroc}")
        print(f"F1: {f1}")
        print(f"Precison: {precision}")
        print(f"Recall: {recall}")
        print(f"F1_micro: {f1_mi}")
        print(f"Precison_micro: {precision_mi}")
        print(f"Recall_micro: {recall_mi}")
     
    results.append(fold_results)

#Write cross-validation results to file
pd.DataFrame(results,columns=['fold','acc','aupr','auroc','f1','precision','recall','f1_micro','precision_micro','recall_micro']).to_csv('cross_validation.csv')