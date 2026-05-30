#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   WTM_run.py
@Time    :   2020/10/04 21:03:13
@Author  :   Leilan Zhang
@Version :   1.0
@Contact :   zhangleilan@gmail.com
@Desc    :   None
'''


import os
import re
import torch
import pickle
import argparse
import logging
import time
# from models import WTM
from WTM_model import WTM
from utils import *
from new_ds import mm_Dataset
from multiprocessing import cpu_count
import matplotlib.pyplot as plt

def list_of_strings(arg):
    return arg.split(',')

# use as python3 WTM/WTM_run.py -f "gmp_as_docs.txt" -l "covid-host.sample.meta.txt" --num_epochs 50 --learning_rate 0.001 --dir_alpha 0.1 --rebuild True
parser = argparse.ArgumentParser('WLDA topic model')
parser.add_argument('-n', '--task_name',type=str,required=True,help='name associated with model trained') # gmp_as_docs.txt
parser.add_argument('-l', '--labels', type=str, required=True,help='sample labels and metadata')
parser.add_argument('-f', '--data_folder', type=str,required=True,help='data folder')
parser.add_argument('-m', '--mode_names', type=list_of_strings,required=True,help='name of data modes (ie dna, rna, ...)')
parser.add_argument('-e', '--expmat_fnames',type=list_of_strings,required=True,help='expression matrix of docs')


#flag args
parser.add_argument('--log_transform', action='store_true', help='Add flag to log-transform data')
parser.add_argument('--rebuild',action='store_true',help='Whether to rebuild the corpus, such as tokenization, build dict etc.(default False)') # False
parser.add_argument('--auto_adj',action='store_true',help='To adjust the no_above ratio automatically (default:rm top 20)')
parser.add_argument('--norm', action='store_true', help='Norm or No')


parser.add_argument('--use_tfidf',type=bool,default=False,help='Whether to use the tfidf feature for the BOW input') # False
parser.add_argument('--no_below',type=int,default=0,help='The lower bound of count for words to keep, e.g 10') # 0
parser.add_argument('--no_above',type=float,default=1,help='The ratio of upper bound of count for words to keep, e.g 0.3') # 1
parser.add_argument('--num_epochs',type=int,default=10,help='Number of iterations (set to 100 as default, but 1000+ is recommended.)') # 5
parser.add_argument('--n_topic',type=int,default=10,help='Num of topics') # 10
parser.add_argument('--bkpt_continue',type=bool,default=False,help='Whether to load a trained model as initialization and continue training.') # False

# defaults
parser.add_argument('--dist',type=str,default='dirichlet',help='Prior distribution for latent vectors: (dirichlet,gmm_std,gmm_ctm,gaussian etc.)') # Dirichlet
parser.add_argument('--batch_size',type=int,default=32,help='Batch size (default=512)') #32
parser.add_argument('--criterion',type=str,default='cross_entropy',help='The criterion to calculate the loss, e.g cross_entropy, bce_softmax, bce_sigmoid')

parser.add_argument('--ckpt',type=str,default=None,help='Checkpoint path') # no checkpoint to start training from yet
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Training Learning Rate')
parser.add_argument('--dir_alpha', type=float, default=1e-4, help='If Dirichlet prior, alpha parameter')
parser.add_argument('--log_every', type=int, default=1, help='How often to save and output train details')

##PN 
parser.add_argument('--beta', type=float, default=1.0, help='parameter of NMM')


args = parser.parse_args()


def main():
    global args
    task_name = args.task_name
    labels_fname = args.labels
    data_dir = args.data_folder 
    mode_names = args.mode_names
    expmat_fnames = args.expmat_fnames
    no_below = args.no_below
    no_above = args.no_above
    num_epochs = args.num_epochs
    n_topic = args.n_topic
    n_cpu = cpu_count()-2 if cpu_count()>2 else 2
    bkpt_continue = args.bkpt_continue
    use_tfidf = args.use_tfidf
    rebuild = args.rebuild
    dist = args.dist
    batch_size = args.batch_size
    criterion = args.criterion
    ckpt = args.ckpt
    dir_alpha = args.dir_alpha
    learning_rate = args.learning_rate
    log_every = args.log_every
    log_transform = args.log_transform
    do_normalize = args.norm
    beta = args.beta

    

    print(f'dir_alpha: {dir_alpha}\tlearning_rate: {learning_rate}')
    
    if rebuild:
        print("rebuilding corp")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # docSet = new_DocDataset(taskname,no_below=no_below,no_above=no_above,rebuild=rebuild)
    docSet = mm_Dataset(rebuild=True, data_dir=data_dir,mode_names=mode_names,
                            expmat_fnames=expmat_fnames, metadata_fname=labels_fname,
                            out_fname=f'{task_name}_docDataset.pkl',
                           scale='gene', log=log_transform, do_normalize=do_normalize) 
    
    voc_size = docSet.vocabsize
    print('voc size:',voc_size)

    
    if ckpt:
        checkpoint=torch.load(ckpt)
        param=checkpoint["param"]
        param.update({"device": device})
        model = WTM(**param)
        trainloss_lst = model.train(train_data=docSet,batch_size=batch_size,dir_alpha=dir_alpha,
                    learning_rate=learning_rate,test_data=None,num_epochs=num_epochs,
                    log_every=log_every,beta=beta,ckpt=checkpoint)
    else:
        model = WTM(bow_dim=voc_size,n_topic=n_topic,device=device,dist=dist,taskname=task_name,dropout=0.4)
        trainloss_lst = model.train(train_data=docSet,batch_size=batch_size,dir_alpha=dir_alpha,
                    learning_rate=learning_rate,test_data=None,num_epochs=num_epochs,
                    log_every=log_every,beta=beta) # changed beta=1.0 to beta=beta PN

    # get coherence evaluation metrics
    # (c_v, c_w2v, c_uci, c_npmi, mimno_tc, td_score),\
    # (cv_per_topic, c_w2v_per_topic, c_uci_per_topic, c_npmi_per_topic) = model.evaluate(test_data=docSet, calc4each=True)

    # embeds = model.get_embed(train_data=docSet, num=1000) # used tobe txt_lst, embeds; embeds is np.array (105, n_topics)

    # with open(os.path.join(data_dir, labels_fname), 'r') as file:
    #     labels_list = [line.strip() for line in file.readlines()]
    #     pickle.dump({'txts':labels_list,'embeds':embeds},open('wtm_data/wtm_embeds.pkl','wb'))
    smth_pts = smooth_curve(trainloss_lst)

    plt.plot(np.array(range(len(smth_pts)))*log_every, smth_pts)
    #plt.plot(np.array(range(len(trainloss_lst))) * log_every, trainloss_lst) #PN
    plt.xlabel('epochs')
    plt.title('Train Loss')
    plt.savefig(f"{data_dir}/wlda_trainloss_{num_epochs}_beta_{beta}.png")

    
if __name__ == "__main__":
    main()