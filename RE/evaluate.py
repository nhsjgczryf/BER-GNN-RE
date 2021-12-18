import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
from dataloader import relations
from sklearn.metrics import classification_report,f1_score

labels = ['Cause-Effect',
            'Component-Whole',
            'Content-Container',
            'Entity-Destination',
            'Entity-Origin',
            'Instrument-Agency',
            'Member-Collection',
            'Message-Topic',
            'Product-Producer']

def get_score(gold_set, predict_set):
    TP = len(set.intersection(gold_set, predict_set))
    # print("#TP",TP,"#gold",len(gold_set),"#predict",len(predict_set))
    precision = TP/(len(predict_set)+1e-6)
    recall = TP/(len(gold_set)+1e-6)
    f1 = 2*precision*recall/(precision+recall+1e-6)
    if len(predict_set)==0 and len(gold_set)==0:
        precision, recall, f1=1,1,1
    return precision, recall, f1

def evaluation(model,dataloader,amp=False,device=torch.device('cpu')):
    model.eval()
    model.to(device)
    tqdm_dataloader = tqdm(dataloader, desc='eval')
    golds = [] # (sentence_id,subject_id,object_id,relation_type)
    predicts = [] #
    with torch.no_grad():
        for batch in tqdm_dataloader:
            input_ids,attention_mask,subjs,objs,target,sent_id = batch['input_ids'],batch['attention_mask'],batch['subjs'],batch['objs'],batch['target'],batch['sent_id']
            input_ids,attention_mask = input_ids.to(device),attention_mask.to(device)
            if amp:
                with autocast():
                    predict = model(input_ids,attention_mask,subjs,objs)
            else:
                predict = model(input_ids,attention_mask,subjs,objs)
            golds.extend(decode(sent_id,subjs,objs,target))
            predicts.extend(decode(sent_id,subjs,objs,predict))
    predicts1 = {l:[] for l in labels}
    golds1 = {l:[] for l in labels}
    for p in predicts:
        predicts1[p[-1]].append(p)
    for g in golds:
        golds1[g[-1]].append(g)
    scores = {l:None for l in labels}
    for l in labels:
        assert len(golds1[l])==len(set(golds1[l])) and len(predicts1[l])==len(set(predicts1[l]))
        p,r,f = get_score(set(golds1[l]),set(predicts1[l]))
        scores[l]=(p,r,f)
    macro_p,macro_r,macro_f = 0,0,0
    output = ['P\tR\tF1\tSupport\tLabel']
    for l in labels:
        p,r,f = scores[l]
        macro_p+=p
        macro_r+=r
        macro_f+=f
        output.append('{:.4f}\t{:.4f}\t{:.4f}\t{}\t{}'.format(p,r,f,len(golds1[l]),l))
    macro_p,macro_r,macro_f = macro_p/len(labels),macro_r/len(labels),macro_f/len(labels)
    output.append('{:.4f}\t{:.4f}\t{:.4f}\t{}\tMacro Score'.format(macro_p,macro_r,macro_f,sum([len(golds1[l]) for l in labels])))
    print('\n'.join(output))
    return macro_f

def decode(sent_id,subjs,objs,predict):
    res = []
    for s_id,subj,obj,pre in zip(sent_id,subjs,objs,predict):
        pre = relations[pre]
        if pre!='Other':
            res.append((s_id,subj,obj,pre)) #由于在给定句子的饿时候，实体已经给定了，所以这里我们直接用实体前的特殊符号的token的索引标识句子，这是没有问题的。
    return res
                   