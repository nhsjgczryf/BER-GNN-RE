"""
其实这个文件的代码设计很糟糕，主要是我们需要一些全局的东西，这里没怎么弄好
"""

import json
import copy
import random
import os
import math
import json
import re
import shutil

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import pdb

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

nodes = None
node2idx = None
edges = None
last_edges = None
last_drop_edges = None

def graph_sampling(adj,rel_edges,graph_drop_rate):
    '''
    这个函数对模型的输入做batch处理
    Args:
        g: 这是图的矩阵表示,就是一个Numpy二维数组
        rel_edges: 实体之间的边，注意这里不应该存在对称的元素，即,如果(a,b)在rel_edges中，那么(b,a)就不在rel_edges中
        graph_drop_rate: 对nodes对应的subgraph的所有边，采样的比例。（其实可以分析一下，一个subgraph一般有多少条边)。这个值为0代表不drop，为1代表全部drop
    '''    
    import random
    #这行代码非常非常费时间
    nadj = copy.deepcopy(adj) #没一个batch，对对应一个新的图
    drop_edges = random.sample(rel_edges,int(graph_drop_rate*len(rel_edges)))
    for e in drop_edges:
        x,y = e
        assert adj[y][x]==1 and adj[x][y]==1
        nadj[x][y]=0
        nadj[y][x]=0
    return nadj    

def graph_sampling1(rel_edges,graph_drop_rate):
  import random
  global edges,last_edges,last_drop_edges
  #nedges = copy.deepcopy(edges) #这里分配内存也要一定时间，但是比起上面，要好很多了，总之，这个函数是我们主要的的复杂度
  for e in last_drop_edges:
    last_edges.add(e)
  last_drop_edges=set()
  #这个时候应该有
  assert len(last_edges)==len(edges) and len(last_drop_edges)==0
  drop_edges = random.sample(rel_edges,int(graph_drop_rate*len(rel_edges)))
  #print(edges)
  #print(rel_edges)
  #print(drop_edges)  
  for e in drop_edges:
    x,y = e
    #assert ((x,y) in last_edges) and ((y,x) in last_edges)
    last_edges.remove((x,y))
    last_edges.remove((y,x))
    last_drop_edges.add((x,y))
    last_drop_edges.add((y,x))
  return last_edges

def collate_fn(batch):
    global edges
    nbatch = {}
    for b in batch:
        for k, v in b.items():
            nbatch[k] = nbatch.get(k, []) + [v]
    graph_drop_rate = nbatch['graph_drop_rate'][0]
    assert all([i==graph_drop_rate for i in nbatch['graph_drop_rate']])
    assert all([nbatch['adj'][0] is i for i in nbatch['adj']])
    adj = nbatch['adj'][0]
    input_ids = nbatch['input_ids']
    subj_idxs = nbatch['subj_idxs']
    obj_idxs = nbatch['obj_idxs']
    target = nbatch['target']
    rel_edges = set()
    for s,o,t in zip(subj_idxs,obj_idxs,target):
        if relations[t]!='Other':
            esi = node2idx[nodes[s][2]]
            eoi = node2idx[nodes[o][2]]
            if not(((esi,eoi) in edges) and ((eoi,esi) in edges)):
                #这里是考虑到dev和test上面，其实是不存边的
                continue
            if esi<eoi:
                rel_edges.add((esi,eoi))
            elif esi>eoi:
                rel_edges.add((eoi,esi))
            else:
                #这种情况是可能的,因为str相同，但是不同上下文中不同
                #raise Exception("Self Connection")
                pass
    #下面这段代码会导致我们的计算速度大大降低，为甚么？
    #nadj = graph_sampling(adj,rel_edges,graph_drop_rate)
    nadj = adj
    nedges = graph_sampling1(rel_edges,graph_drop_rate)
    input_ids = [torch.tensor(i) for i in input_ids]
    input_ids1 = pad_sequence(input_ids,batch_first=True,padding_value=0)
    target = torch.tensor(target)
    attention_mask = torch.zeros(input_ids1.shape)
    for i in range(len(input_ids)):
        attention_mask[i,:len(input_ids[i])]=1
    nbatch['input_ids'] = input_ids1
    nbatch['target'] = target
    nbatch['attention_mask'] = attention_mask
    nbatch['adj']=nadj
    nbatch['edges']=nedges
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
    def __init__(self,train_path,dev_path,test_path,tokenizer,max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.train_sent_ids,self.train_sentences,self.train_targets = parse_file(train_path)
        self.dev_sent_ids,self.dev_sentences,self.dev_targets = parse_file(dev_path)
        self.test_sent_ids,self.test_sentences,self.test_targets = parse_file(test_path)
        self.sentences = self.train_sentences+self.dev_sentences+self.test_sentences
        self.sent_ids = self.train_sent_ids+self.dev_sent_ids+self.test_sent_ids
        self.raw_targets = self.train_targets+self.dev_targets+self.test_targets
        self.train_size,self.dev_size,self.test_size = len(self.train_targets),len(self.dev_targets),len(self.test_targets)
        self.input_ids = []
        self.subjects = [] #subject实体在句子中的位置
        self.objects = [] #object实体在句子中的位置
        self.subj_idxs = []
        self.obj_idxs = []
        self.targets = []
        self.edges = set()

        #首先获取一些全局的信息，主要是mention的集合，以及entity的集合
        #一个mention代表输入样本中的一个具体的实体，用(sentece_id,mention_idx,subj or obj)，注意这里的
        #一个entity代表所有实体字符串的集合
        subjs = [] #subjs和objs是
        objs = []
        for s_id, raw_sent, tgt in zip(tqdm(self.sent_ids,desc='preprocess'), self.sentences,self.raw_targets):
            sub_str = re.search('<subj> (.*) </subj>',raw_sent).group(1)
            obj_str = re.search('<obj> (.*) </obj>',raw_sent).group(1)
            sent = tokenizer.tokenize(raw_sent)
            assert set(['<subj>','</subj>','<obj>','</obj>']).issubset(set(sent))
            input_id = [tokenizer.convert_tokens_to_ids(w) for w in sent]
            sub_id = sent.index('<subj>')
            obj_id = sent.index('<obj>')
            assert max(sent.index('</subj>'),sent.index('</obj>'))+1<max_len
            if len(input_id)>max_len:
                input_id = input_id[:max_len-1]+input_id[-1:]
            tgt_id = relations.index(tgt)
            self.input_ids.append(input_id)
            self.subjects.append(sub_id)
            self.objects.append(obj_id)
            self.targets.append(tgt_id)
            subjs.append((s_id,sub_id,sub_str,'subj'))
            objs.append((s_id,obj_id,obj_str,'obj'))
        assert len(set(subjs+objs))==len(subjs+objs)
        self.mentions = subjs+objs
        #TODO： 这里可以考虑更复杂的处理，比如lowercase，比如lemmalization等等，这里只是简单的字符串匹配而已
        self.entities = list(set([i[2] for i in self.mentions]))
        self.nodes = self.mentions+self.entities
        self.node2idx = dict(zip(self.nodes,range(len(self.nodes))))
        for s,o in zip(subjs,objs):
            self.subj_idxs.append(self.node2idx[s])
            self.obj_idxs.append(self.node2idx[o])
        self.adj = torch.zeros((len(self.nodes),len(self.nodes)),dtype=torch.uint8)
        i = 0
        for si,oi,t in zip(self.subj_idxs,self.obj_idxs,self.targets):
            if i==self.train_size:#这里不能把dev和test的边给加进去
                break
            i+=1
            if relations[t]!='Other':
                ns,no = self.nodes[si],self.nodes[oi]
                es,eo = ns[2],no[2]
                esi,eoi = self.node2idx[es],self.node2idx[eo]
                self.adj[si][esi]=1
                self.adj[esi][si]=1
                self.adj[oi][eoi]=1
                self.adj[eoi][oi]=1
                self.adj[esi][eoi]=1
                self.adj[eoi][esi]=1
                self.edges.add((si,esi))
                self.edges.add((esi,si))
                self.edges.add((oi,eoi))
                self.edges.add((eoi,oi))
                self.edges.add((esi,eoi))
                self.edges.add((eoi,esi))
        global adj,nodes,node2idx,edges,last_edges,last_drop_edges
        adj = self.adj           
        nodes = self.nodes
        node2idx = self.node2idx 
        edges = self.edges
        last_edges = copy.deepcopy(edges)
        last_drop_edges = set()
        #print(edges)
        #pdb.set_trace()

    def __len__(self):
        return len(self.sent_ids)


    def __getitem__(self,i):
        return {'adj':self.adj,'input_ids':self.input_ids[i],'target':self.targets[i],
                'subjs':self.subjects[i],'objs':self.objects[i],'sent_id':self.sent_ids[i],
                'subj_idxs':self.subj_idxs[i],'obj_idxs':self.obj_idxs[i]}

class MyDataset1:
    def __init__(self,dataset,idxs,graph_drop_rate):
        self.nodes = dataset.nodes
        self.node2idx = dataset.node2idx
        self.graph_drop_rate = graph_drop_rate
        self.data = [dataset[i] for i in idxs]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,i):
        t = self.data[i]
        t['graph_drop_rate'] = self.graph_drop_rate
        return t       

if __name__=="__main__":
    train_path = '/home/wangnan/wangnan/SemEval/data/train3.txt'
    dev_path = '/home/wangnan/wangnan/SemEval/data/train3 copy.txt'
    test_path = '/home/wangnan/wangnan/SemEval/data/train3 copy 2.txt'
    tokenizer = AutoTokenizer.from_pretrained('/data/nfsdata/nlp/BERT_BASE_DIR/uncased_L-12_H-768_A-12')
    tokenizer.add_special_tokens({'additional_special_tokens': ['<subj>', '</subj>','<obj>','</obj>']})
    max_len = 100
    ds = MyDataset(train_path,dev_path,test_path,tokenizer,max_len)
    idxs = list(range(ds.train_size))
    graph_drop_rate = 0.5
    ds1 = MyDataset1(ds,idxs,0.5)
    batch_size = 10
    dl = DataLoader(ds1,batch_size=10,collate_fn=collate_fn,shuffle=False)
    l = list(dl)
    None        