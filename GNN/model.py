import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GATConv,GraphConv
from transformers import AutoModel
from layers import BertRE
import pdb

class REGNN(nn.Module):
    def __init__(self,config):
        super(REGNN,self).__init__()
        self.config = config
        self.re_model = self.load_re_model(config.checkpoint_path)
        hidden_size = self.re_model.bert.config.hidden_size
        self.gnn_layers = nn.ModuleList()
        if config.gnn_type=='gat':
            for l in range(config.num_layers):
                self.gnn_layers.append(GATConv(hidden_size,hidden_size,num_heads=1,residual=config.residual ,allow_zero_in_degree=config.allow_zero_in_degree))
        elif config.gnn_type=='gcn':
            for l in range(config.num_layers):
                self.gnn_layers.append(GraphConv(hidden_size,hidden_size,norm=config.norm,allow_zero_in_degree=config.allow_zero_in_degree))
        else:
            pass
        self.dropout = nn.Dropout(config.dropout)
        self.loss_fuc = nn.CrossEntropyLoss()
        num_rels = 10      
        #这里去掉了之前那个feature_based的考虑，即，仅仅考虑joint loss的情况   
        if self.config.score_type in ['bilinear','both']:
            self.W1 = nn.Bilinear(hidden_size,hidden_size,num_rels)
        if self.config.score_type in ['concate','both']:
            self.W2 = nn.Linear(2*hidden_size,num_rels)     
    
    def load_re_model(self,ckpt_path):
        #这里导入之前训练的关系抽取的模型
        #NOTE: 需要import模型类
        checkpoint = torch.load(ckpt_path,map_location='cpu')
        config = checkpoint['args']
        model = BertRE(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def forward(self,input_ids,attention_mask,subjs,objs,g,feat,subj_idxs,obj_idxs,target)        :
        loss1 = self.re_model(input_ids,attention_mask,subjs,objs,target)
        subj_rep, obj_rep = self.re_model.subj_rep,self.re_model.obj_rep
        feat[subj_idxs]=subj_rep
        feat[obj_idxs]=obj_rep
        g_rep = feat
        #pdb.set_trace()
        for gl in self.gnn_layers:
            g_rep = gl(g,g_rep)
        if g_rep.shape[1]==1:
            g_rep = g_rep[:,0,:] #这里GAT存在mulit head的情况，中间多了一个head的维度
        subj_rep,obj_rep = g_rep[subj_idxs],g_rep[obj_idxs]
        if self.config.score_type=='bilinear':
            rel_rep = self.W1(subj_rep,obj_rep)
        elif self.config.score_type=='concate':
            #pdb.set_trace()
            rel_rep = self.W2(torch.cat([subj_rep,obj_rep],dim=1))
        elif self.config.score_type=='both':
            rel_rep = self.W1(subj_rep,obj_rep) + self.W2(torch.cat([subj_rep,obj_rep],dim=1))
        else:
            raise Exception("Invalid score type.")
        loss2 = self.loss_fuc(rel_rep,target)
        loss = self.config.alpha*loss1+(1-self.config.alpha)*loss2
        return loss

    def predict(self,input_ids,attention_mask,subjs,objs,g,feat,subj_idxs,obj_idxs):
        prob1 = self.re_model(input_ids,attention_mask,subjs,objs)
        subj_rep, obj_rep = self.re_model.subj_rep,self.re_model.obj_rep
        g_rep = feat
        pdb
        for gl in self.gnn_layers:
            g_rep = gl(g,g_rep)
        if g_rep.shape[1]==1:
            g_rep = g_rep[:,0,:]
        subj_rep,obj_rep = g_rep[subj_idxs],g_rep[obj_idxs]
        if self.config.score_type=='bilinear':
            rel_rep = self.W1(subj_rep,obj_rep)
        elif self.config.score_type=='concate':
            rel_rep = self.W2(torch.cat([subj_rep,obj_rep],dim=1))
        elif self.config.score_type=='both':
            rel_rep = self.W1(subj_rep,obj_rep) + self.W2(torch.cat([subj_rep,obj_rep],dim=1))
        else:
            raise Exception("Invalid score type.")       
        prob2 = rel_rep.softmax(dim=-1)
        prob = self.config.alpha*prob1+(1-self.config.alpha)*prob2
        return prob.argmax(dim=-1)