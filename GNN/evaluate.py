import torch
from tqdm import tqdm
import dgl
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

def subgraph(g,seed_nodes,num_layers,feat,subj_idxs,obj_idxs):
    '''
    输入：图，节点集合,GNN的层数
    输出：新的子图(保留节点索引)，
    '''
    #实现思路，我们首先调用(num_layers-1)次的in_graph，即，这里主要是获取我们需要的nodes
    #然后，我们根据最后一次获得的nodes，调用in_graph,这就是我们最后的一个graph了
    num_nodes = [len(seed_nodes)]
    for i in range(num_layers):
      sg = dgl.in_subgraph(g,seed_nodes)
      seed_nodes = sg.edges()[0].tolist()+sg.edges()[1].tolist()
      seed_nodes = list(set(seed_nodes))
      num_nodes.append(len(seed_nodes))
    ng = dgl.node_subgraph(g,seed_nodes)
    org_id = ng.ndata[dgl.NID]
    cur_id = ng.nodes()
    index = {o.item():c.item() for o,c in zip(org_id,cur_id)} #这里的key是原图的id,value是新图的id
    feat1 = torch.zeros((len(index),feat.shape[1])).type_as(feat)
    for k,v in index.items():
      feat1[v]=feat[k]
    subj_idxs1 = [index[i] for i in subj_idxs]
    obj_idxs1 = [index[i] for i in obj_idxs]
    #print(num_nodes)
    return ng,feat1,subj_idxs1,obj_idxs1

def evaluation(model,feat,dataloader,selfloop,amp=False,device=torch.device('cpu')):
    model.eval()        
    model.to(device)
    tqdm_dataloader = tqdm(dataloader, desc='eval')
    golds = [] # (sentence_id,subject_id,object_id,relation_type)
    predicts = [] #
    feat = feat.to(device)
    with torch.no_grad():
        for batch in tqdm_dataloader:
            torch.cuda.empty_cache()
            input_ids,attention_mask,subjs,objs,target = batch['input_ids'],batch['attention_mask'],batch['subjs'],batch['objs'],batch['target']
            input_ids,attention_mask = input_ids.to(device),attention_mask.to(device)
            sent_id = batch['sent_id']
            #NOTE: 这里强调一下，我们这里的图g，其实是一个邻接矩阵，不是那个dgl里面的dgl.graph，所以也没有node feature这个概念。
            adj,subj_idxs,obj_idxs = batch['adj'],batch['subj_idxs'],batch['obj_idxs']
            edges = batch['edges']
            edges = list(zip(*edges))  
            g = dgl.graph((list(edges[0]),list(edges[1]))).to(device)
            if selfloop:
                g = dgl.add_self_loop(g)            
            #g = dgl.graph(adj.nonzero(as_tuple=True)).to(device)
            assert len(set(subj_idxs+obj_idxs))==len(set(subj_idxs+obj_idxs))
            seed_nodes = list(set(subj_idxs+obj_idxs))
            g1,feat1,subj_idxs1,obj_idxs1 = subgraph(g,seed_nodes,model.config.num_layers,feat,subj_idxs,obj_idxs)
            if selfloop:
                g = dgl.add_self_loop(g)            
            if amp:
                with autocast():
                    predict = model.predict(input_ids,attention_mask,subjs,objs,g1,feat1,subj_idxs1,obj_idxs1)
            else:
                predict = model.predict(input_ids,attention_mask,subjs,objs,g1,feat1,subj_idxs1,obj_idxs1)
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