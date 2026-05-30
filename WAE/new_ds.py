#!/usr/bin/env python
# coding: utf-8

import os
import time
import numpy as np
import pandas as pd
import gensim
import pickle
import random
import torch
from tqdm import tqdm
# from tokenization import *
from torch.utils.data import Dataset,DataLoader
from collections import Counter
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from collections import Counter
from collections.abc import Mapping
import sys
import re
# sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)



class DDictionary(Dictionary):
    # inherits from gensim's Dictionary, simply will redefine how to initialize

        
    
    def __init__(self, expmats_dict):
        # expect expmats to be list of dfs
        
        self.token2id = {} # map from token to id
        self.id2token = {} # map from id to token
        self.cfs = {} # number of total instances of word in whole corpus
        self.dfs = {} # number of documents containing the word
        self.num_docs = 0 # number of docs processed
        self.num_pos = 0 # number of words processed
        self.num_nnz = 0 # number of unique words per document in entire corpus (or just number of nonzero in matrix)

        id = 0 # 
        for type, df in expmats_dict.items():

            self.num_docs = df.shape[0]
            self.num_pos += df.shape[1]
            self.num_nnz += np.count_nonzero(df)
            
            for word, col in df.items():
                if word in self.token2id:
                    word = word + "_" + type # because crc has MND1 gene and MND1 microbe
                
                self.token2id[word] = id
                self.id2token[id] = word
                self.cfs[id] = col.sum()
                self.dfs[id] = (col != 0).sum()
                
                id += 1

# used to default data_dir='/gpfs/gibbs/project/gerstein/rtl35/privacy_network/covid_data', outfile_name='docDataset_sn.pkl'
class mm_Dataset(Dataset):
    def __init__(self,rebuild,
                 data_dir,
                 mode_names,
                 expmat_fnames,
                 metadata_fname,
                 out_fname,
                 no_below=0,no_above=1,stopwords=[],use_tfidf=False,
                 scale=None, do_normalize=True,log=False
                ):
        
        if not os.path.exists(data_dir):
            print('provided datadir does not exist')
            return

        if not rebuild:
            rebuild_path = os.path.join(data_dir, 'model_data',out_fname)
            if os.path.exists(rebuild_path):
                with open(rebuild_path, 'rb') as handle:
                    save = pickle.load(handle)
                    self.__dict__.update(save.__dict__)
                print('docSet loading succeeded')
            else:
                print('docSet loading failed, could not locate file')

        else:
            print('Rebuilding')

            self.txtLines = None
            self.samp_metadata = pd.read_csv(os.path.join(data_dir, metadata_fname), sep=',', index_col=0)
            
            self.dictionary = None
            self.bows,self.docs = None,None
            self.use_tfidf,self.tfidf,self.tfidf_model = False,None,None

            
            # Load dfs
            self.expmats = {}
            self.expbool = {}
            for name, fname in zip(mode_names, expmat_fnames):
                self.expmats[name] = pd.read_csv(os.path.join(data_dir, fname), index_col=0)
                print(os.path.join(data_dir, fname))
                colnames = self.expmats[name].columns
                self.expmats[name] = self.expmats[name].loc[:, (~colnames.str.startswith('MT-')) & (~colnames.isin(stopwords)) & (self.expmats[name].sum(axis=0)!=0)] # remove MT, stopwords, and expressionless columns

                self.expbool[name] = self.expmats[name] > 0 # store for linkage purposes later
                
                if log:
                    self.expmats[name] = np.log(self.expmats[name] + 1e-5)
                    self.expmats[name] = self.expmats[name] + abs(min(self.expmats[name].min().min(), 0)) # ensure no negative values 
            #print(mode_names)
            #print(self.expmats)
            self.label_type = np.concatenate([np.repeat(mode_names,[self.expmats[name].shape[1] for name in mode_names])])
            
            # Build dictionary
            self.dictionary = DDictionary(self.expmats)

            for key in self.expmats.keys():
                self.expmats[key].columns = [self.dictionary.token2id[name] for name in self.expmats[key].columns]
            
            
            # scaling expression frequencies
            if scale is not None: # scale
                    
                if scale in self.expmats:
                    depths = self.expmats[scale].sum(axis=1)
                    # Avoid division by zero if a sample has 0 total counts
                    self.scales = depths.max() / depths.replace(0, 1) 

                    for m_name in self.expmats.keys():
                        self.expmats[m_name] = self.expmats[m_name].mul(self.scales, axis=0)
                else:
                    print(f"Warning: Scaling modality '{scale}' not found.")
                    self.scales = 1
            
            else:
                self.scales = 1
            
            # global min-max normalization for expression frequencies
            if do_normalize:
                for type, exp_df in self.expmats.items():
                    self.expmats[type] = (exp_df - exp_df.min().min())/(exp_df.max().max() - exp_df.min().min())
            

            # min-max normalization per sample (was wtih old code, currently broken!)
            # for doc, (_, genes), (_, microbes), (_, premirs) in zip(self.docs,
            #                                          self.gene_df.iterrows(),
            #                                          self.microbeR_df.iterrows(),
            #                                          self.premiR_df.iterrows()):

            #     min-max normalization, per samples
            #     genes = (genes - genes.min())/(genes.max() - genes.min())
            #     microbes = (microbes - microbes.min())/(microbes.max() - microbes.min())
            #     premirs = (premirs - premirs.min())/(premirs.max() - premirs.min())
            
            # make BoWs
            self.bows, _docs = [],[]
            all_exp_df = pd.concat([exp_df for type, exp_df in self.expmats.items()], axis=1)
            for i, vals in all_exp_df.iterrows():
                exp_bow = [(label, value) for label, value in vals.items() if value > 0]
                if exp_bow != 0:
                    self.bows.append(exp_bow)
            self.vocabsize = self.dictionary.num_pos # can't do len(dict) because some words are not expressed
            self.numDocs = self.dictionary.num_docs

            # save a copy for future loading (when rebuild=False)
            with open(os.path.join(data_dir,out_fname), 'wb') as handle:
                pickle.dump(self, handle)
                print('successfully saved after rebuilding')
        
    
    def __getitem__(self,idx):
        bow = torch.zeros(self.vocabsize)
        if self.use_tfidf:
            item = list(zip(*self.tfidf[idx]))
        else:
            item = list(zip(*self.bows[idx])) # bow = [[token_id1,token_id2,...],[freq1,freq2,...]]
        bow[list(item[0])] = torch.tensor(list(item[1])).float()
        # txt = self.docs[idx]
        return None,bow # remove docs, txt from pipeline; only use normalized in BoW
    
    def __len__(self):
        return self.numDocs
    
    def collate_fn(self,batch_data):
        _, bows = list(zip(*batch_data))
        return None, torch.stack(bows,dim=0)

    def __iter__(self):
        for bow in self.bows:
            yield bow

    def show_dfs_topk(self,topk=20):
        ndoc = self.numDocs
        dfs_topk = sorted([(self.dictionary.id2token[k],fq) for k,fq in self.dictionary.dfs.items()],key=lambda x: x[1],reverse=True)[:topk]
        for i,(word,freq) in enumerate(dfs_topk):
            print(f'{i+1}:{word} --> {freq}/{ndoc} = {(1.0*freq/ndoc):>.13f}')
        return dfs_topk

    def show_cfs_topk(self,topk=20):
        ntokens = sum([v for k,v in self.dictionary.cfs.items()])
        cfs_topk = sorted([(self.dictionary.id2token[k],fq) for k,fq in self.dictionary.cfs.items()],key=lambda x: x[1],reverse=True)[:topk]
        for i,(word,freq) in enumerate(cfs_topk):
            print(f'{i+1}:{word} --> {freq}/{ntokens} = {(1.0*freq/ntokens):>.13f}')
    
    def topk_dfs(self,topk=20):
        ndoc = self.numDocs
        dfs_topk = self.show_dfs_topk(topk=topk)
        return 1.0*dfs_topk[-1][-1]/ndoc