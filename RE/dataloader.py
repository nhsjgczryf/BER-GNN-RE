import os
import math
import json
import shutil
import re

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler


relations = ['Cause-Effect',
            'Component-Whole',
            'Content-Container',
            'Entity-Destination',
            'Entity-Origin',
            'Instrument-Agency',
            'Member-Collection',
            'Message-Topic',
            'Other',
            'Product-Producer']

def collate_fn(batch):
    nbatch = {}
    for b in batch:
        for k, v in b.items():
            nbatch[k] = nbatch.get(k, []) + [v]
    input_ids = nbatch['input_ids']
    subjs = nbatch['subjs']
    objs = nbatch['objs']
    target = nbatch['target']
    input_ids = [torch.tensor(i) for i in input_ids]
    input_ids1 = pad_sequence(input_ids,batch_first=True,padding_value=0)
    target = torch.tensor(target)
    attention_mask = torch.zeros(input_ids1.shape)
    for i in range(len(input_ids)):
        attention_mask[i,:len(input_ids[i])]=1
    nbatch['input_ids'] = input_ids1
    nbatch['target'] = target
    nbatch['attention_mask'] = attention_mask
    return nbatch

def parse_file(path):
    """
    input: 文件路径
    output: 句子以及对应的target
    """
    f = open(path).readlines()
    f0 = [f[i] for i in range(0,len(f),4)] #sentence
    f1 = [f[i] for i in range(1,len(f),4)] #target
    f2 = [f[i] for i in range(2,len(f),4)] #comment
    f3 = [f[i] for i in range(3,len(f),4)] #'\n'
    sent_ids = []
    sentences = []
    targets = []
    for i in range(len(f0)):
        s = f0[i]
        s = s.strip()
        sent_id, sent =  s.split('\t')
        sent_id = int(sent_id)
        assert sent[0]==sent[-1] and sent[0]=='"'
        sent = sent[1:-1]
        target = f1[i]
        target_rel = target.split('(')[0] #这个是关系的类型
        #e1_span = re.search('<e1>.+</e1>').span()
        #e2_span = re.search('<e1>.+</e1>').span()
        #这里我们会把一个句子样本变为两个句子样本，但是呢，不会包含
        sent1 = sent.replace('<e1>','<subj> ')
        sent1 = sent1.replace('</e1>',' </subj>')
        sent1 = sent1.replace('<e2>','<obj> ')
        sent1 = sent1.replace('</e2>',' </obj>')
        target1 = target_rel if '(e1,e2)' in target else 'Other'
        sent_ids.append(sent_id) #注意我们这里sent_id加了两次，因为一个句子其实对应两个样本
        sentences.append(sent1)
        targets.append(target1)
        sent2 = sent.replace('<e1>','<obj> ')
        sent2 = sent2.replace('</e1>',' </obj>')
        sent2 = sent2.replace('<e2>','<subj> ')
        sent2 = sent2.replace('</e2>',' </subj>')
        target2 = target_rel if '(e2,e1)' in target else 'Other'
        sent_ids.append(sent_id)
        sentences.append(sent2)
        targets.append(target2)
    return sent_ids,sentences,targets


class MyDataset:
    '''
    其实这里有一个自然约束，就是，每一个输入中的(e1,e2)或者(e2,e1)之间的关系，只可能是10个关系中的一种，但是如果我们把一个句子变成两个句子，可能出现这种情况，即，(e1,e2)和(e2,e1)同时属于两个不同的关系，但是这两个关系并不相同。
    所以其实我们还可以这样考虑，即，考虑两个独立的分类任务，其中第一个分类任务判断关系的方向，第二个分类任务判断关系的类别，这样就有些类似multi task learning，。
    '''
    def __init__(self,path='',tokenizer=None,max_len=512):
        self.sent_ids,self.raw_sentences,self.raw_targets = parse_file(path)
        #这里主要考虑不同与训练模型的tokenizer的问题
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.input_ids = [] #句子经过tokenizer后的索引
        self.subjects = [] #subject实体在句子中的位置
        self.objects = [] #object实体在句子中的位置
        self.targets = [] #句子对应的target，这里是上面10个relation中的一个

        for sent,tgt in zip(self.raw_sentences,self.raw_targets):
            sent = tokenizer.tokenize(sent)
            assert set(['<subj>','</subj>','<obj>','</obj>']).issubset(set(sent))
            input_id = [tokenizer.convert_tokens_to_ids(w) for w in sent]
            sub_id = sent.index('<subj>')
            obj_id = sent.index('<obj>')
            assert max(sent.index('</subj>'),sent.index('</obj>'))+1<max_len
            if len(input_id)>max_len:
                input_id = input_id[:max_len-1]+input_id[-1]
            tgt_id = relations.index(tgt)
            self.input_ids.append(input_id)
            self.subjects.append(sub_id)
            self.objects.append(obj_id)
            self.targets.append(tgt_id)
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self,i):
        return {'input_ids':self.input_ids[i],'target':self.targets[i],
                'subjs':self.subjects[i],'objs':self.objects[i],'sent_id':self.sent_ids[i]}

def load_data(path,batch_size,pretrained_model_name_or_path,max_len=200,shuffle=False):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<subj>', '</subj>','<obj>','</obj>']})
    dataset = MyDataset(path,tokenizer,max_len)
    dataloader = DataLoader(dataset,batch_size,shuffle=shuffle,collate_fn=collate_fn)
    return dataloader

if __name__=="__main__":
    path='/home/wangnan/wangnan/SemEval/data/train3.txt'
    batch_size=2
    pretrained_model_name_or_path='/data/nfsdata/nlp/BERT_BASE_DIR/uncased_L-12_H-768_A-12'
    dl = load_data(path,batch_size,pretrained_model_name_or_path)
    None