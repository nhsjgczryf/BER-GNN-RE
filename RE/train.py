import time
import os
import random
import argparse
import pickle
import torch

#os.environ['CUDA_VISIBLE_DEVICES']='2'
#print(torch.cuda.device_count())

from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import trange, tqdm
import numpy as np

from torch.nn.utils import clip_grad_norm_

from dataloader import  *
from model import  BertRE
from evaluate import evaluation

root_dir = os.path.dirname(os.path.abspath(__file__))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path")
    parser.add_argument("--dev_path")
    parser.add_argument('--pretrained_model_name_or_path')
    parser.add_argument("--save_type",type=str,choices=['all','best','none'])
    parser.add_argument("--save_dir",type=str)
    parser.add_argument("--max_epochs",type=int)
    parser.add_argument("--batch_size",type=int)
    parser.add_argument("--max_len",type=int)
    parser.add_argument("--lr",type=float)
    parser.add_argument("--cls",action="store_true")
    parser.add_argument("--score_type",type=str,choices=['bilinear','concate','both'])
    parser.add_argument("--warmup_rate",type=float,default=-1)
    parser.add_argument("--weight_decay",default=0.1,type=float)
    parser.add_argument("--dropout",type=float,default=0.1)
    parser.add_argument("--max_grad_norm",type=float,default=1)
    parser.add_argument("--amp",action="store_true")
    parser.add_argument("--debug",action="store_true")
    parser.add_argument("--tensorboard",action="store_true")
    parser.add_argument("--tqdm_mininterval",type=int,default=1)
    parser.add_argument("--seed",type=int,default=0)
    return parser.parse_args()

def train(args,train_dataloader,dev_dataloader): 
    model = BertRE(args)
    if args.amp:
        scaler = GradScaler()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], "weight_decay":args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay":0.0}
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
    for epoch in range(args.max_epochs):
        model.train()
        tqdm_train_dataloader = tqdm(train_dataloader, desc="epoch:%d" % epoch, ncols=150)
        for i, batch in enumerate(tqdm_train_dataloader):
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            input_ids,attention_mask,subjs,objs,target = batch['input_ids'],batch['attention_mask'],batch['subjs'],batch['objs'],batch['target']
            input_ids,attention_mask,target = input_ids.to(device),attention_mask.to(device),target.to(device)
            if args.amp:
                with autocast():
                    loss = model(input_ids,attention_mask,subjs,objs,target)
                scaler.scale(loss).backward()
                if args.max_grad_norm>0:
                    scaler.unscale_(optimizer)
                named_parameters = [(n, p) for n, p in model.named_parameters() if not p.grad is None]
                grad_norm = torch.norm(torch.stack([torch.norm(p.grad) for n, p in named_parameters])).item()                    
                clip_grad_norm_(model.parameters(),args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model(input_ids,attention_mask,subjs,objs,target)
                loss.backward()
                named_parameters = [(n, p) for n, p in model.named_parameters() if not p.grad is None]
                grad_norm = torch.norm(torch.stack([torch.norm(p.grad) for n, p in named_parameters])).item()                
                if args.max_grad_norm>0:
                    clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
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
        score = evaluation(model,dev_dataloader,args.amp,device)                
        if args.save_type=='none':
            #不保存就啥也不干
            pass
        elif args.save_type=='all':
            #这里保存所有的epoch
            save(model,args,mid,epoch,score)
        elif args.save_type=='best':
            if score>=best_score:
                if best_score!=-1:
                    os.remove(save_path)
                best_score = score
                save_path = save(model,args,mid,epoch,score)
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

if __name__=='__main__':
    args = args_parser()
    set_seed(args.seed)
    #args.debug=True
    if args.debug:
        args.train_path = '/home/wangnan/wangnan/SemEval/data/train3.txt'
        args.dev_path = '/home/wangnan/wangnan/SemEval/data/train3.txt'
        #args.save_dir = '/data/nfs/wangnan/BERT-GBB-RE/BERT-RE/conll04'
        args.pretrained_model_name_or_path = '/data/nfsdata/nlp/BERT_BASE_DIR/uncased_L-12_H-768_A-12'
        args.save_type = 'best'
        args.max_epochs = 100
        args.batch_size = 2
        args.max_len = 200
        args.lr = 2e-5
        args.cls
        args.score_type = 'concate'
        args.warmup_rate = -1
        args.dropout = 0.1
        args.max_grad_norm = 1
        args.amp = False
        #args.tensorboard = True
    train_dataloader = load_data(args.train_path,args.batch_size,args.pretrained_model_name_or_path,args.max_len,True)
    dev_dataloader = load_data(args.dev_path,args.batch_size,args.pretrained_model_name_or_path,args.max_len,False)
    print(args)
    train(args,train_dataloader,dev_dataloader)    