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
            if self.config.score_type in ['concate','both']:
                self.W2 = nn.Linear(2*hidden_size,num_rels)

    def forward(self,input_ids,attention_mask,subjs,objs,target=None):
        '''
        我傻了啊，如果我们用CLS的表示，那么存在的一个问题是，我们没有entity的表示了啊。。。
        input_ids: (batch,seq_len)
        attention_mask: (batch,seq_len)
        subjs: (batch,)
        objs: (batch,)
        target: (batch,)
        '''
        rep, _ = self.bert(input_ids,attention_mask,return_dict=False)
        rep = self.dropout(rep) #(batch,seq_len,hidden_size)
        cls_rep = rep[:,0,:] #(batch,hidden_size)
        assert len(rep)==len(subjs) and len(subjs)==len(objs)
        subj_rep = torch.stack([rep[i][subjs[i]] for i in range(len(rep))],dim=0) #(batch,hidden_size)
        obj_rep = torch.stack([rep[i][objs[i]] for i in range(len(rep))],dim=0) #(batch,hidden_size)
        if self.config.cls:
            rel_rep = self.W0(cls_rep)
        else:
            if self.config.score_type=='bilinear':
                #第一种是x1Tx1的形式
                rel_rep = self.W1(subj_rep,obj_rep)
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
            return rel_rep.argmax(dim=-1).cpu()