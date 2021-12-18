import time
import os
import random
import argparse
import pickle
import torch

#os.environ['CUDA_VISIBLE_DEVICES']='2'
#print(torch.cuda.device_count())
torch.autograd.set_detect_anomaly(True)


from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from transformers import AutoTokenizer
import dgl
from model import REGNN
from dataloader import MyDataset,MyDataset1,collate_fn
import pdb

root_dir = os.path.dirname(os.path.abspath(__file__))


from evaluate import evaluation

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def args_parser():
    """
    这里只考虑实体节点和mention节点的情况，然后由于考虑实体节点可能存在self feature的情形，所以只支持transductive的setup
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path")
    parser.add_argument('--checkpoint_path',help='预训练的关系抽取模型')
    parser.add_argument("--train_path")
    parser.add_argument("--dev_path")
    parser.add_argument("--test_path")
    parser.add_argument('--batch_size',type=int)
    parser.add_argument('--max_epochs',type=int)
    parser.add_argument('--max_len',type=int,help='输入的最大长度')
    parser.add_argument('--lr',type=float,help='新的参数的学习率')
    parser.add_argument('--pre_lr',type=float,help='预训练的关系抽取模型的学习率')
    parser.add_argument('--dropout',type=float)

    parser.add_argument('--gnn_type',type=str,choices=['gat','gcn','none'])
    parser.add_argument("--selfloop",action="store_true")
    parser.add_argument('--num_layers',type=int)
    parser.add_argument('--num_heads',type=int,default=1)#这个参数并不支持，因为会导致输出维度多一个head的维度
    parser.add_argument('--norm',type=str,choices=['right','both','none'])
    parser.add_argument('--residual',action='store_true')
    parser.add_argument('--allow_zero_in_degree',action='store_true')

    #其实还有很多想法没有实现，可以看最开始的GNN-RE1项目的代码，以及之前思考的文字记录
    parser.add_argument("--graph_drop_rate",type=float,default=0) #这里是batch里面图的drop的比例
    parser.add_argument("--eval_graph_drop_rate",type=float,default=0) #这个是评估的时候的drop比率
    parser.add_argument("--alpha",type=float,default=0.5) # 当feature_based为false的时候，即joint training的时候，权重的问题

    parser.add_argument("--freeze_pre",action="store_true") #这里冻结之前的关系抽取模型的参数。

    parser.add_argument('--score_type',type=str,choices=['concate','bilinear','both']) # 注意，这里我们是经过GCN layer的结果
    parser.add_argument('--warmup_rate',type=float,default=0.1)
    parser.add_argument("--weight_decay",type=float,default=0.01)
    parser.add_argument("--save_type",type=str,choices=['all','best','none'])
    parser.add_argument("--debug",action="store_true")
    parser.add_argument('--max_grad_norm',type=float,default=1)
    parser.add_argument('--amp',action='store_true')
    parser.add_argument("--seed",type=int,default=1)
    parser.add_argument("--tqdm_mininterval",type=int,default=1)
    parser.add_argument('--tensorboard',action='store_true')
    
    return parser.parse_args()

'''
#这里理一下整个模型的流程
1. 创建模型,optimizer
2. 在训练集上迭代，每个样本既是一篇文档，也是实体对组成的subgraph。
3. 更新所有mention节点的底层表示，然后顺便在train/dev/test三个集合上面进行评估
4. 保存模型
'''    

def subgraph(g,seed_nodes,num_layers,feat,subj_idxs,obj_idxs):
    '''
    输入：图，节点集合,GNN的层数
    输出：新的子图(保留节点索引)，
    '''
    #实现思路，我们首先调用(num_layers-1)次的in_graph，即，这里主要是获取我们需要的nodes
    #然后，我们根据最后一次获得的nodes，调用in_graph,这就是我们最后的一个graph了
    all_seed_nodes = []
    all_seed_nodes.append(seed_nodes)
    num_nodes = [len(seed_nodes)]
    for i in range(num_layers):
      sg = dgl.in_subgraph(g,seed_nodes)
      seed_nodes = sg.edges()[0].tolist()+sg.edges()[1].tolist()
      seed_nodes = list(set(seed_nodes))
      num_nodes.append(len(seed_nodes))
      all_seed_nodes.append(seed_nodes)
    ng = dgl.node_subgraph(g,seed_nodes)
    org_id = ng.ndata[dgl.NID]
    cur_id = ng.nodes()
    index = {o.item():c.item() for o,c in zip(org_id,cur_id)} #这里的key是原图的id,value是新图的id
    feat1 = torch.zeros((len(index),feat.shape[1])).type_as(feat)
    for k,v in index.items():
      feat1[v]=feat[k]
    #pdb.set_trace()
    subj_idxs1 = [index[i] for i in subj_idxs]
    obj_idxs1 = [index[i] for i in obj_idxs]
    #print(num_nodes)
    return ng,feat1,subj_idxs1,obj_idxs1

def update_graph(model,dataloader,device,amp=False,eval=False):
    if eval:
        model.eval()#这里主要是关闭模型的drop out
    all_subj_reps = []
    all_obj_reps = []
    all_subj_idxs = []
    all_obj_idxs = []
    #注意这里不能有梯度
    with torch.no_grad():
        for batch in tqdm(dataloader,desc='update graph'):
            torch.cuda.empty_cache()
            subj_idxs,obj_idxs = batch['subj_idxs'],batch['obj_idxs']
            input_ids,attention_mask,subjs,objs = batch['input_ids'],batch['attention_mask'],batch['subjs'],batch['objs']
            input_ids,attention_mask = input_ids.to(device),attention_mask.to(device)
            if amp:
                with autocast():
                    subj_rep,obj_rep = model.re_model.get_ent_rep(input_ids,attention_mask,subjs,objs)
            else:
                subj_rep,obj_rep = model.re_model.get_ent_rep(input_ids,attention_mask,subjs,objs)
            #很奇怪，下面两行代码大了，会导致GPU OOM，但是这只是在CPU上的操作啊，不管我怎么优化好像都没用
            all_subj_reps.append(subj_rep)
            all_obj_reps.append(obj_rep)
            all_subj_idxs.extend(subj_idxs)
            all_obj_idxs.extend(obj_idxs)
    #assert len(set(all_subj_idxs+all_obj_idxs))==len(all_subj_idxs+all_obj_idxs)
    feats = torch.zeros(len(dataloader.dataset.nodes),all_subj_reps[0].shape[1]).type_as(all_subj_reps[0]) 
    feats[all_subj_idxs]=torch.cat(all_subj_reps,dim=0)
    feats[all_obj_idxs]=torch.cat(all_obj_reps,dim=0)
    return feats    


def train(args,dataset):
    #准备数据
    train_size,dev_size,test_size = dataset.train_size,dataset.dev_size,dataset.test_size
    train_idxs = list(range(train_size))
    dev_idxs = [i+train_size for i in list(range(dev_size))]
    test_idxs = [i+train_size+dev_size for i in list(range(test_size))]
    #这个用于训练
    train_dataset = MyDataset1(dataset,train_idxs,graph_drop_rate=args.graph_drop_rate)
    train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=collate_fn)
    #这个用于更新节点feature
    all_dataset = MyDataset1(dataset,list(range(len(dataset))),graph_drop_rate=args.graph_drop_rate)
    all_dataloader = DataLoader(all_dataset,batch_size=args.batch_size,shuffle=False,collate_fn=collate_fn)
    #这两个用于评估
    dev_dataset = MyDataset1(dataset,dev_idxs,graph_drop_rate=args.eval_graph_drop_rate)
    test_dataset = MyDataset1(dataset,test_idxs,graph_drop_rate=args.eval_graph_drop_rate)
    dev_dataloader = DataLoader(dev_dataset,batch_size=args.batch_size,shuffle=False,collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False,collate_fn=collate_fn)    

    model = REGNN(args)
    if args.amp:
        scaler = GradScaler()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    if args.freeze_pre:
        model.re_model.require_grad=False
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {"params":[p for n,p in model.named_parameters() if  n.startswith('re_model') and (not any(nd in n for nd in no_decay)) and p.requires_grad],"weight_decay":args.weight_decay},
        {"params":[p for n,p in model.named_parameters() if (not n.startswith('re_model')) and (not any(nd in n for nd in no_decay)) and p.requires_grad],"weight_decay":args.weight_decay,'lr':args.pre_lr},
        {"params":[p for n,p in model.named_parameters() if n.startswith('re_model') and any(nd in n for nd in no_decay) and p.requires_grad],"weight_decay":0.0},
        {"params":[p for n,p in model.named_parameters() if (not n.startswith('re_model')) and any(nd in n for nd in no_decay) and p.requires_grad],"weight_decay":0.0,'lr':args.pre_lr},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    if args.warmup_rate > 0:
        num_training_steps = len(train_dataloader)*args.max_epochs
        warmup_steps = args.warmup_rate*num_training_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)
    mid = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
    if args.tensorboard:
        log_dir = "{}/logs/{}".format(root_dir,mid)
        writer = SummaryWriter(log_dir)    
    ltime = time.time()  
    best_score = -1      
    save_path = ''        
    #初始化feature
    feat = update_graph(model,all_dataloader,device,args.amp,eval=False)
    for epoch in range(args.max_epochs):
        feat = feat.to(device)
        model.train()
        tqdm_train_dataloader = tqdm(train_dataloader, desc="epoch:%d" % epoch, ncols=150)
        for i,batch in enumerate(tqdm_train_dataloader):
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            input_ids,attention_mask,subjs,objs,target = batch['input_ids'],batch['attention_mask'],batch['subjs'],batch['objs'],batch['target']
            input_ids,attention_mask,target = input_ids.to(device),attention_mask.to(device),target.to(device)
            #NOTE: 这里强调一下，我们这里的图g，其实是一个邻接矩阵，不是那个dgl里面的dgl.graph，所以也没有node feature这个概念。
            adj,subj_idxs,obj_idxs = batch['adj'],batch['subj_idxs'],batch['obj_idxs']
            edges = batch['edges']
            edges = list(zip(*edges))
            #g = dgl.graph(adj.nonzero(as_tuple=True)).to(device)
            #pdb.set_trace()
            g = dgl.graph((list(edges[0]),list(edges[1]))).to(device)
            if args.selfloop:
                g = dgl.add_self_loop(g)
            #assert len(set(subj_idxs+obj_idxs))==len(set(subj_idxs+obj_idxs))
            seed_nodes = list(set(subj_idxs+obj_idxs))
            g1,feat1,subj_idxs1,obj_idxs1 = subgraph(g,seed_nodes,args.num_layers,feat,subj_idxs,obj_idxs)
            if args.amp:
                with autocast():
                    #loss = model(input_ids,attention_mask,subjs,objs,g,feat,subj_idxs,obj_idxs,target)
                    loss = model(input_ids,attention_mask,subjs,objs,g1,feat1,subj_idxs1,obj_idxs1,target)
                scaler.scale(loss).backward()
                if args.max_grad_norm>0:
                    scaler.unscale_(optimizer)
                named_parameters = [(n, p) for n, p in model.named_parameters() if not p.grad is None]
                grad_norm = torch.norm(torch.stack([torch.norm(p.grad) for n, p in named_parameters])).item()
                clip_grad_norm_(model.parameters(),args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                #loss = model(input_ids,attention_mask,subjs,objs,g,feat,subj_idxs,obj_idxs,target)
                loss = model(input_ids,attention_mask,subjs,objs,g1,feat1,subj_idxs1,obj_idxs1,target)
                loss.backward()
                named_parameters = [(n, p) for n, p in model.named_parameters() if not p.grad is None]
                grad_norm = torch.norm(torch.stack([torch.norm(p.grad) for n, p in named_parameters])).item()
                if args.max_grad_norm>0:
                    clip_grad_norm_(model.parameters(),args.max_grad_norm)
                optimizer.step()
            #TODO 检查一下一致性的问题，我们这里相比之前有feat和feat1的区别
            feat[subj_idxs+obj_idxs]=feat1[subj_idxs1+obj_idxs1]
            feat.detach_() #这步很重要,我们之前因为这行代码，一直没有什么进展
            lr = optimizer.param_groups[0]['lr']
            if args.warmup_rate > 0:
                scheduler.step()
            if args.tensorboard:
                writer.add_scalar('loss', loss.item(), i + epoch*len(train_dataloader))
                writer.add_scalars("lr_grad", {"lr": lr, "grad_norm": grad_norm}, i+epoch*len(train_dataloader))
                writer.flush()
            if time.time()-ltime >= args.tqdm_mininterval:
                postfix_str = 'norm:{:.2f},lr:{:.1e},loss:{:.2e}'.format(grad_norm, lr, loss.item())
                tqdm_train_dataloader.set_postfix_str(postfix_str)
                ltime = time.time() 
        #epoch结束后，统一更新node feature
        feat = update_graph(model,all_dataloader,device,args.amp,eval=False)
        #下面是评估
        #NOTE：我们这里的评估，mention node的表示是考虑了drop edge的，主要是出于训练效率考虑,所以最好单独针对保存的checkpoint写一个eval函数
        #在dev上的评估结果
        print("dev:")
        score = evaluation(model,feat,dev_dataloader,args.selfloop,args.amp,device) #这里之前支持在eval的时候调整参数，现在不支持了
        #在test上的评估结果
        print("test")
        evaluation(model,feat,test_dataloader,args.selfloop,args.amp,device)
        #这里我们考虑save all
        if args.save_type=='none':
            #不保存就啥也不干
            pass
        elif args.save_type=='all':
            #这里保存所有的epoch
            save(model,args,mid,epoch,score)
        elif args.save_type=='best':
            if score>=best_score:
                save(model,args,mid,epoch,score)
        elif args.save_type=='latest':
            if save_path:
                os.remove(save_path)
            save_path = save(model,args,mid,epoch,score)
        else:
            raise Exception()
            
def save(model,args,mid,epoch=-1,score=0,save_dir=''):
    checkpoint = {}
    model_state_dict = model.state_dict()
    checkpoint['model_state_dict'] = model_state_dict
    checkpoint['args'] = args
    if not save_dir:
        save_dir = '%s/checkpoints'%root_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir,"%s_%d_%.4f.cpt"%(mid,epoch,score))
    torch.save(checkpoint, save_path)
    print("model saved at:", save_path)
    return save_path

if __name__=="__main__":
    args = args_parser()
    set_seed(args.seed)
    #args.debug=True
    if args.debug:
        args.seed = 1
        args.train_path = '/home/wangnan/wangnan/SemEval/data/train3.txt'
        args.dev_path = '/home/wangnan/wangnan/SemEval/data/train3 copy.txt'
        args.test_path = '/home/wangnan/wangnan/SemEval/data/train3 copy 2.txt'
        args.pretrained_model_name_or_path = '/data/nfsdata/nlp/BERT_BASE_DIR/uncased_L-12_H-768_A-12'
        args.checkpoint_path = '/home/wangnan/wangnan/SemEval/RE/checkpoints/BERT-RE/2021_04_08_13_00_42_30_1.0000.cpt'
        args.save_type = 'none'
        args.batch_size = 10
        args.max_epochs = 10
        args.max_len = 100
        args.lr = 4e-5
        args.pre_lr = 8e-6
        args.dropout = 0.1
        args.gnn_type = 'gcn'
        args.selfloop=True
        #这个是GCN的参数
        args.norm = 'both'
        #这个是GAT的参数
        args.num_heads = 1
        args.residual = False

        args.num_layers = 2
        args.allow_zero_in_degree = False
        args.graph_drop_rate = 0.5
        args.eval_graph_drop_rate = 0
        args.alpha = 0.5
        args.score_type = 'bilinear'
        args.warmup_rate = 0.01
        args.tensorboard = False
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)      
    tokenizer.add_special_tokens({'additional_special_tokens': ['<subj>', '</subj>','<obj>','</obj>']})
    dataset = MyDataset(args.train_path,args.dev_path,args.test_path,tokenizer,args.max_len)
    print(args)
    train(args,dataset)