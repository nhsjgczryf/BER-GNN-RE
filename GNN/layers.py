'''
这里主要的目的是在之前的RE模型上面添加新的东西
'''


import torch
import torch.nn as nn
from transformers import  AutoModel

class BertRE(nn.Module):
    '''用Bert进行关系抽取的模型'''
    def __init__(self,config):
        super(BertRE,self).__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained(config.pretrained_model_name_or_path)
        self.bert.resize_token_embeddings(self.bert.config.vocab_size+4)
        hidden_size = self.bert.config.hidden_size
        #这个值和数据集有关
        num_rels = 10
        self.dropout = nn.Dropout(p=config.dropout)
        self.loss_fuc = nn.CrossEntropyLoss()
        if self.config.cls:
            #这里是用[CLS]的表示作为我们的relation的表示
            self.W0 = nn.Linear(hidden_size,num_rels)
        else:
            #这里是用head_rep+tail_rep的方式作为我们的relation的表示
            if self.config.score_type in ['bilinear','both']:
                self.W1 = nn.Bilinear(hidden_size,hidden_size,num_rels)
            elif self.config.score_type in ['concate','both']:
                self.W2 = nn.Linear(2*hidden_size,num_rels)

    def forward(self,input_ids,attention_mask,subjs,objs,target=None):
        '''
        #这里相比原来的模型，只有两个改动，第一个是保存了subj和obj的表示，第二个是target=None的时候，我们输出的是概率
        input_ids: (batch,seq_len)
        attention_mask: (batch,seq_len)
        subjs: (batch,)
        objs: (batch,)
        target: (batch,)
        '''
        rep, _ = self.bert(input_ids,attention_mask,return_dict=False)
        rep = self.dropout(rep) #(batch,seq_len,hidden_size)
        cls_rep = rep[:,0,:] #(batch,hidden_size)
        subj_rep = torch.stack([rep[i][subjs[i]] for i in range(len(rep))],dim=0) #(batch,hidden_size)
        obj_rep = torch.stack([rep[i][objs[i]] for i in range(len(rep))],dim=0) #(batch,hidden_size)
        self.subj_rep = subj_rep 
        self.obj_rep = obj_rep
        if self.config.cls:
            rel_rep = self.w0(cls_rep)
        else:
            if self.config.score_type=='bilinear':
                #第一种是x1Tx1的形式
                rel_rep = self.W1(subj_rep,obj_rep) #(batch,num_rel)
            elif self.config.score_type=='concate':
                #第二种是concate的形式
                rel_rep = self.W2(torch.cat([subj_rep,obj_rep],dim=1))
            elif self.config.score_type=='both':
                #第三种是前面两种的结合
                rel_rep = self.W1(subj_rep,obj_rep) + self.W2(torch.cat([subj_rep,obj_rep],dim=1))
            else:
                raise Exception()
        if target is not None:
            loss = self.loss_fuc(rel_rep,target)
            return loss
        else:
            return rel_rep.softmax(dim=-1)
    def get_ent_rep(self,input_ids,attention_mask,subjs,objs):
        #这个函数只有在update_graph的时候才会调用
        rep, _ = self.bert(input_ids,attention_mask,return_dict=False)
        rep1 = rep.cpu()
        del rep
        torch.cuda.empty_cache()
        subj_rep = torch.stack([rep1[i][subjs[i]] for i in range(len(rep1))],dim=0) #(batch,hidden_size)
        obj_rep = torch.stack([rep1[i][objs[i]] for i in range(len(rep1))],dim=0) #(batch,hidden_size)
        return subj_rep,obj_rep