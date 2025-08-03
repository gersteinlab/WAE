#!/usr/bin/python
# -*- coding: utf-8 -*-
'''how to random shuffle?

use the same model, with the phi
different theta, same phi, means the document topic

repeat for inference means:
the model are different, the links are different.

'''



from __future__ import print_function  # (at top of module)

import numpy as np
from copy import deepcopy
import pandas as pd
import os
from os import listdir, makedirs
from os.path import isfile, join, exists
from numpy.random import random
from scipy.special import psi
from math import log,exp
from scipy.special import gammaln as lgamma
import pickle



def cumDicing(p0, n=1):
    '''cumulative dice rolling
        '''

    plen = p0.shape[0]
    p = p0.copy()
    p = np.cumsum(p)

    px = random(n) * p[-1]

    topic_idx = [np.min(np.where(p > i)) for i in px]
    return topic_idx


def main(
    k0,
    a0,
    b0,
    m0,
    d0,
    nt0,
    shuffle,
    out,
    ):

    # #parameters:
    import re
    pattern = re.compile('^(.+)_(m\d+|s).([^\.]+)$')
    
    K0 = k0
    alpha0 = a0
    beta0 = b0

    Nrand=1  ##here change 1000 to 1 because it not neccessary
    metafile = m0  # #file including corresponding docname, one column string

    mldadict={}
    # ##read folder
    #the folder should only contains data files
    mypath = d0

    outdir =os.path.dirname(out)
    docName = np.genfromtxt(metafile, dtype=str, delimiter='\n')
    niter0 = nt0
    ndoc0 = docName.shape[0]

    filelist = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    ntype=len(filelist)
    print(filelist)
    

    #mlda.set_alpha(a0)
    '''
    name of input is the type name with lda_type, for example: a1_m1.txt a2_m1.txt b1_s.txt b2_s.txt
    format of input matrix:
    no header, tab demilited
    row is document/sample
    colname is gene/microbes
    
    '''
    print("Start reading file")
    for file0 in filelist:
        ###tname is used to fix the order of 
        #type_name = os.path.splitext(os.path.basename(file0))[0]
        fn = os.path.basename(file0)
        m = pattern.match(fn)
        lda_type=""
        fext=""
        mlda =None ## mLDA(K0, ntype, ndoc0)
        print(fn+" in processing")
            
        if m:
            type_name = m.group(1)
            lda_group= m.group(2)
            fext = m.group(3)
            print("type_name:"+ type_name+", lda_group:"+lda_group+", fext:"+fext)
            
            tmpdata0 = np.genfromtxt(mypath + '/' + file0, dtype=int,
                                            delimiter='\t')  # #nparray
            tmpdata=tmpdata0[:, 1:]
            tmpskip= list(tmpdata0[:,0])
            
            if shuffle>0: ###save shuffle file
                [np.random.shuffle(x) for x in tmpdata]
                np.savetxt(outdir+'/shuffle.'+type_name+'.txt', tmpdata, "%d", delimiter='\t')
                
            if lda_group not in mldadict:
                mldadict[lda_group] = mLDA(K0, ntype, ndoc0, out,lda_group)  ##todo: K0 setting
                mldadict[lda_group].set_alpha(a0)

                
            tmplda= LDA(K0, type_name,a0,  beta0, tmpdata,skip=tmpskip)
            tmplda.init()
            mldadict[lda_group].ldaDict[type_name] = tmplda
            mldadict[lda_group].init_LDA(tmplda)
            mldadict[lda_group].loglikelihoods[type_name] = tmplda.l  ##get likelihood
            mldadict[lda_group].Vbsum.append(np.sum(tmplda.nV*tmplda.beta))
            mldadict[lda_group].tnames.append(type_name)     


            
        else:
            print("The format input filename is: assayType_(m1|s).txt, please change your filename and do it again")
            return 1
        


    ##skl todo: add some process to handle this parameter automatically

    for kk in mldadict:
        print(kk+" start to update:")
        for it in range(niter0):
            print(str(it)+" run start: ")
            mldadict[kk].collapse_sampling()
            #print('calculate phi')
            #mldadict[kk].calc_phi()
        print("done iteration")

    
    print("start to output data")
    for kk in mldadict:
        print("output:"+kk+"\n")
        mldadict[kk].calc_phi()
        
        ##close loglh and opt
        mldadict[kk].llhoutfh.close()
        mldadict[kk].optfh.close()
    
        print('calculate theta')
        mldadict[kk].calc_theta()
        print('Now output training matrix:')

        np.savetxt(out + '-' + kk + '-' + 'mlda.theta.txt', mldadict[kk].theta, delimiter=',')
        for tt0 in mldadict[kk].ldaDict:
            tt = mldadict[kk].ldaDict[tt0]
            # ##nw_t
            np.savetxt(out + '-' + kk + '-' + tt0 + '.nw_t.txt', tt.nw_t, delimiter=',')
            # nw_t
            np.savetxt(out + '-' + kk + '-' + tt0 + '.nd_t.txt', tt.nd_t, delimiter=',')
            np.savetxt(out + '-' + kk + '-' + tt0 + '.ndSum.txt', tt.ndSum,delimiter=',')
            np.savetxt(out + '-' + kk + '-' + tt0 + '.nwSum.txt', tt.nwSum,delimiter=',')
            np.savetxt(out + '-' + kk + '-' + tt0 + '.phi.txt', tt.phi, delimiter=',')


def optAlpha(old, d_t):
    '''
    old_p in pdim,1 or pdim, shape
    mat in d, k or v, k
    
    '''

    pdim = old.shape[0]
    ndim = d_t.shape[0]
    n2dim = d_t.shape[1]

    if pdim != n2dim:
        return None

    nupper = 0.0  # #numerator
    nlower = 0.0  # denominator
    for n0 in range(ndim):
        nupper = psi(d_t[n0, :] + old) - psi(old)
        nlower = psi(np.sum(d_t[n0, :] + old)) - psi(np.sum(old))

    new_p = old * (nupper / nlower)

    return new_p


def optBeta(old, w_t):
    '''
    old_p in pdim,1 or pdim, shape
    mat in d, k or v, k
    MINKA'S FIXED-POINT ITERATION
    
    '''

    pdim = old.shape[0]
    ndim = w_t.shape[0]
    n2dim = w_t.shape[1]

    if pdim != n2dim:
        return None

    nupper = 0.0  # #numerator
    nlower = 0.0  # denominator
    for n0 in range(ndim):
        nupper = psi(w_t[n0, :] + old) - psi(old)
        nlower = psi(np.sum(w_t[n0, :] + old)) - psi(np.sum(old))

    new_p = old * (nupper / nlower)

    return new_p


class LDA:

    def __init__(
        self,
        K,
        t_name,
        alpha0,
        beta0,
        data,
        skip=None
    ):

        self.K = K
        self.data = data

        self.t_name = t_name
        self.nV = data.shape[1]
        # self.alpha=[]  ###
        self.alpha = np.full((K),alpha0)
        self.beta = np.full((self.nV),beta0)  # #1:V
        self.ndoc = data.shape[0]
        self.skip=skip if skip is not None else [1]*self.ndoc

        self.l = 0.0 ##loglikelihood
        # self.ldaMat=[]

        self.doc_len = np.sum(data, axis=1)  # ## rowSums 0 is column, 1 is row
        self.p = None

        # ##matrix for lda
        # ##store the topic for each word

        self.z = np.empty((self.ndoc, self.nV), dtype='object')  # ## readable (w) topic assignment for words, size m  x doc.length;
        self.nw_t = np.zeros((self.nV, self.K))  # # writable w number of word(i) to topic(j)  size v x k for each type
        self.nd_t = np.zeros((self.ndoc, self.K))  # # readable r times of document i to j topic, Mx K
        self.ndSum = np.zeros(self.ndoc)  # #total words to topic size  M
        self.nwSum = np.zeros(self.K)  # ## total words in document  K,

        # self.train_theta  ##=nd_t doc->topic

        self.phi = np.zeros((self.nV, self.K))  # #=nw_t topic->word

        # #self.tA   ##topic assign
        # self.word

    def init(self):
        for m in range(self.ndoc):
            if self.skip[m]==0:
                continue
            # #doc_tmp = self.data[m,:]
            # #z_tmp = self.z[m,:]  ###need keep z_tmp
            for n in range(self.nV):
                
                if self.data[m, n] > 0:  # ##integer
                    rndxx = random(self.data[m, n])
                    tt = list(map(int, rndxx * self.K))  # ## topic index 0-based
                    tt_uniq = np.unique(tt, return_counts=True)  # ##0 is the topic, 1 is the topics count
                    self.z[m, n] = ','.join(map(str, tt))  # ##or None if no this word in the text

                    # #store the topic to z matrix
                    # #document to topic count

                    self.nd_t[m, tt_uniq[0].tolist()] = self.nd_t[m,tt_uniq[0].tolist()] + tt_uniq[1].tolist()

                    # # totl words in doc

                    self.ndSum[m] = self.ndSum[m] + self.data[m, n]

                    # # word to topic

                    self.nw_t[n, tt_uniq[0].tolist()] = self.nw_t[n,tt_uniq[0].tolist()] + tt_uniq[1].tolist()

                    # # ttotal word in topic k

                    self.nwSum[tt_uniq[0].tolist()] = \
                        self.nwSum[tt_uniq[0].tolist()] \
                        + tt_uniq[1].tolist()

        ###get loglikelihood
        self.loglikelihood()
        
        print('LDA initialization done!')

    def calc_phi(self):
        for n in range(self.K):
            self.phi[:,n] = (self.nw_t[:, n] + self.beta) / (self.nwSum[n]
                                                             + np.sum(self.beta))

    def loglikelihood(self):
        '''
        
        '''
        ##k topic


        xxdata = self.data
        Mx = xxdata.shape[0]
        lll = 0.0
        for m in range(Mx):
            lll += lgamma(np.sum(self.alpha)) - np.sum(lgamma(self.alpha))
            lll += np.sum(lgamma(self.nd_t[m,:]+self.alpha))
            lll -= lgamma(np.sum(self.nd_t[m,:]+self.alpha))
            
        for k in range(self.K):
            lll += lgamma(np.sum( self.beta)) - np.sum(lgamma(self.beta))
            lll += np.sum(lgamma(self.nw_t[:,k]+self.beta))
            lll -= lgamma(np.sum(self.nw_t[:,k]+self.beta))



        self.l=lll
        

    def gibbs(self):
        '''
        sampling for the lda
        '''

        #loglikelihood=0.0
        xxdata = self.data
        Mx = xxdata.shape[0]
        Vx = xxdata.shape[1]

        for m in range(Mx):
            tmpdata = xxdata[m, :]
            for v in range(Vx):
                v_num = tmpdata[v]
                if v_num >0:
                    tmp_z = list(map(int, self.z[m, v].split(',')))
                    tmp_z0 = tmp_z


                    for n in range(v_num):
                        self.nd_t[m, tmp_z[n]] = self.nd_t[m, tmp_z[n]]-1
                        self.nw_t[v, tmp_z[n]] = self.nw_t[v, tmp_z[n]]-1
                        self.nwSum[tmp_z[n]] = self.nwSum[tmp_z[n]] -1
                        #self.nd_t[m, tmp_z[n]] = self.nd_t[m, tmp_z[n]]-1
                        
                        px =  (self.nd_t[m, :] + self.alpha) * (self.nw_t[v, :] + self.beta[v]) / ((np.sum(self.nw_t)+np.sum( self.beta))) # * (np.sum(self.ndSum[m] + self.alpha)))
                        px= px/np.sum(px)
                        
                        tmp_z0[n] = cumDicing(px)[0]

                        self.nd_t[m, tmp_z0[n]] = self.nd_t[m, tmp_z0[n]] +1
                        self.nw_t[v, tmp_z0[n]] = self.nw_t[v, tmp_z0[n]] +1
                        self.nwSum[tmp_z0[n]] = self.nwSum[tmp_z0[n]] +1
                    self.z[m,v] = ''
                    self.z[m,v] = ','.join(list(map(str, tmp_z0)))
                    
        self.loglikelihood()

        print("LDA LogLikelihood: %8.3f" % self.l)



        


    def calc_theta(self):
        for m in range(self.ndoc):
            self.theta[m,:] = (self.nd_t[m,:] +self.alpha) / (np.sum(self.nwSum[m])+ np.sum(self.alpha))
        return 0




class Link2type:

    def __init__(
        self,
        x1,
        x2,
        n1,
        n2,
        n0,
        ):

        self.n1 = n1
        self.n2 = n2
        self.x1 = x1
        self.x2 = x2
        self.n0 = n0
        self.mat = np.zeros((n1, n2, n0))  ##n1type, n2type, n0-document)
        self.mat0 = np.zeros((n1,n2))  ##n1type, n2type
        self.nullmat = None

    def init_null(self, nn):
        self.nullmat = np.zeros((self.n1, self.n2, nn))

    def link_prob(self):

        if self.nullmat is not None:

                        # ##do something

            return None


class mLDA:

    def __init__(
        self,
        k0,
        nt0,
        ndoc0,
        out0,
        ldagroup='m1',
        interfile=True,
        opt_hyper=True ###optimize hyper parameter
        ):
        '''
        todo: alpha and beta use different level
        todo: self.intermediate
        '''
        self.group=ldagroup
        self.iteration=0
        self.intermediate=interfile
        self.K = k0  # num of topics
        self.nt = nt0  # #number of types for each document
        self.theta = np.zeros((ndoc0, k0))
        self.ndoc = ndoc0
        self.doc_sets = None  # #document set
        self.ldaDict = {}
        self.opthyper = opt_hyper
        self.out=out0
        # self.tA=None  ###topic assign
        # #self.nIter=n0iter

        self.a0d_t = np.zeros((self.ndoc, self.K))
        self.a0wSum = np.zeros(self.K)
        self.a0dSum = np.zeros(self.ndoc)
        self.alpha = None
        self.p1 = 0.0
        self.Vbsum = []
        self.loglikelihoods = {}
        self.linkObj = []
        self.linkObj4shuffle=[]
        self.tnames=[]

        self.llhoutfh=open(out0+'-'+self.group+'.logliklihood.txt', 'a+')
        self.optfh = open(out0+'-'+self.group+'.optAB.txt', 'a+')

        
    def getlinkComb(self):
        tname = self.tnames #list(self.ldaDict.keys())

        if self.nt != len(tname):
            print('error')
            return

        for x in range(self.nt-1):
            for y in range(x+1, self.nt):
                x1 = tname[x]
                x2 = tname[y]
                n1 = self.ldaDict[x1].nV
                n2 = self.ldaDict[x2].nV
                n0 = self.ndoc
                link2t = Link2type(x1, x2, n1, n2, n0)

                # ## n x V1 = nxk %x% kxV1 = nxV1 (n is the document count)
                nv1 = np.matmul(self.theta, np.transpose(self.ldaDict[x1].phi))

                # # n x V2
                nv2 = np.matmul(self.theta,np.transpose(self.ldaDict[x2].phi))
                for z in range(nv1.shape[0]): ##document-wise interaction
                    link2t.mat[:, :, z] = np.outer(nv1[z, :], np.transpose(nv2[z, :]))
                link2t.mat0 = np.matmul(self.ldaDict[x1].phi, np.transpose(self.ldaDict[x2].phi))
                
                self.linkObj.append(link2t)

    def set_alpha(self, alpha0): ###vector of alpha0
        self.alpha = np.full((self.K),alpha0)

    def init_LDA(self, lda):
        '''
        need this
        '''
        
        
        self.a0d_t = self.a0d_t + lda.nd_t
        self.a0wSum = self.a0wSum + lda.nwSum
        self.a0dSum = self.a0dSum + lda.ndSum

        print('mLDA initialization done!')

    def calc_phi(self):

        for tt in self.ldaDict:
            self.ldaDict[tt].calc_phi()

    def init():
        '''
        psudo: 
        for (m) in  doc_set :
            topic_idx = random int from [0~k-1]
            for (t, doc) in m:
                for word in doc:
                    word_idx
                    n   nth word in  doc
                   lda_idx = m*self.nt + t
                   self.ldaMat[lda_idx][2].z[m,n] = topic_idx
                   self.ldaMat[lda_idx][2].nw[word_idx, topic_idx]++
                   self.ldaMat[lda_idx][2].nwsum[topic_idx]++
                   self.ldaMat[lda_idx][2].nd[m][topic_idx]++
                   self.ldaMat[lda_idx][2].ndsum[m]++

        '''

        # ##init

    def perplexity(self):
        '''
        perplexity() = exp{-\fraction{\sum^M_{d=1} log p(w_d)}{\sum^M_{d=1} N_d}}
        log p(w_d) = log \sum_{w_i \in d} (p(z|d) p(w_i|z))

        
        '''
        ##k topic

        px=0
        totlen=0
        for tt0 in self.ldaDict:
            print(str(tt0))
            tt = self.ldaDict[tt0]
            xxdata = tt.data  # ## m x v matrix
            Mx = xxdata.shape[0]
            Vx = xxdata.shape[1]
            totlen+=tt.doc_len
            ###docs
            for m in range(Mx):
                if tt.skip[m]==0: ##if this sample has missing data, then skip
                    continue
                ##add document skip, if some document can be skip then go to next 
                tmpdata = xxdata[m, :]
                
                ###words
                for v in range(Vx):
                    v_num = tmpdata[v]
                    if v_num>0:
                        #### topic id z
                        #tmp_z = list(map(int, tt.z[m, v].split(',')))
                    
                        tp = 0;
                        for kk in range(tt.K):

                            tp+=   ((tt.nw_t[v, kk] + tt.beta[v])/(tt.nwSum[kk] + np.sum(tt.beta))) / (( self.a0d_t[m, kk] + self.alpha[kk]  )/ ( np.sum(self.a0dSum[m]+self.alpha) ) )
                            ###each word
                        ##
                        px += log(tp) * v_num
                
        px += exp(-(1.0/totlen)*px)
                            
        return px

    def optAlphaBeta2(self):
        '''
        optimize the alpha beta
        dt:
        '''
        tmp_alpha0=0.0
        tmp_alpha1=0.0
        print("before optimize alpha:", self.alpha)
        for d in range(self.ndoc):
            #tmp_alpha0 += (self.a0d_t[d,:]/(self.a0d_t[d,:] -1 + self.alpha )) 
            #tmp_alpha1 += np.sum(self.a0d_t[d,:])/( np.sum(self.a0d_t[d,:]) - 1 + np.sum(self.alpha) )
            tmp_alpha0 += (self.a0d_t[d,:]+1)/(self.a0d_t[d,:] + self.alpha )
            tmp_alpha1 += np.sum(self.a0d_t[d,:]+1)/( np.sum(self.a0d_t[d,:]) + np.sum(self.alpha) )
        self.alpha *= tmp_alpha0 / tmp_alpha1

        #print("after optimize alpha:", self.alpha)
        
        for tt0 in self.ldaDict:
            tt = self.ldaDict[tt0]
            tmp_beta0=0.0
            tmp_beta1=0.0
            #print("Before optimize beta for", tt0, ":",  tt.beta)
            for k in range(tt.K):
                tmp_beta0 += (tt.nw_t[:,k]+1) / (tt.nw_t[:,k]  +tt.beta)
                tmp_beta1 += np.sum(tt.nw_t[:,k]+1)/(np.sum(tt.nw_t[:,k]) + np.sum(tt.beta))
                #tmp_beta0 += (tt.nw_t[:,k] / (tt.nw_t[:,k] -1 +tt.beta))
                #tmp_beta1 += np.sum(tt.nw_t[:,k])/(np.sum(tt.nw_t[:,k]) - 1 + np.sum(tt.beta))
            tt.beta  = tt.beta* (tmp_beta0/tmp_beta1)
            #print("After optimize beta for", tt0, ":",  tt.beta)



    
    def optAlphaBeta(self):
        '''
        optimize the alpha beta
        dt:
        '''
        nzAdd=0.001
        tmp_alpha0=0.0
        tmp_alpha1=0.0
        #print("before optimize alpha:", self.alpha)
        for d in range(self.ndoc):
            tmp_alpha0 += psi(self.a0d_t[d,:] + self.alpha + nzAdd) -psi(self.alpha )
            tmp_alpha1 += (psi(np.sum(self.a0d_t[d,:] + nzAdd + self.alpha))- psi(np.sum(self.alpha)))
        self.alpha *= (tmp_alpha0 )  / (tmp_alpha1 )

        #print("after optimize alpha:", self.alpha)
        self.optfh.write("Alpha:\t" + "\t".join(str(e) for e in self.alpha) + "\n")

        for tt0 in self.ldaDict:
            tt = self.ldaDict[tt0]
            tmp_beta0=0.0
            tmp_beta1=0.0
            #print("Before optimize beta for", tt0, ":",  tt.beta)
            for k in range(tt.K):
                tmp_beta0 += psi(tt.nw_t[:, k]+tt.beta + nzAdd )- psi(tt.beta)
                tmp_beta1 += psi(np.sum(tt.nw_t[:,k] + nzAdd +tt.beta))- psi(np.sum(tt.beta))
            tt.beta  = tt.beta* (tmp_beta0) / (tmp_beta1 )
            #print("After optimize beta for", tt0, ":",  tt.beta)
            self.optfh.write("beta:\t" + "\t".join(str(e) for e in tt.beta) + "\n")

    def collapse_sampling(self):
        ###loglikelihood = 0.0
        totlll=0.0
        for tt0 in self.ldaDict:
            totlll+= self.ldaDict[tt0].l
        

        
        for tt0 in self.ldaDict:
            print(str(tt0))
            tt = self.ldaDict[tt0]
            xxdata = tt.data  # ## m x v matrix
            Mx = xxdata.shape[0]
            Vx = xxdata.shape[1]
            

            for m in range(Mx):
                if tt.skip[m]==0: ##if this sample has missing data, then skip
                    continue

                ##add optimization
                if self.opthyper:
                    #optAlphaBeta()
                    self.optAlphaBeta()
                    #self.optAlphaBeta2()

                
                tmpdata = xxdata[m, :]
                
                for v in range(Vx):
                    v_num = tmpdata[v]
                    if v_num > 0:  # ## has
                        # #
                        tmp_z = list(map(int, tt.z[m, v].split(',')))
                        tmp_z0 = tmp_z

                        # random(v_num)

                        for n in range(v_num):
                            # ##update here each time
                            tt.nd_t[m, tmp_z[n]] = tt.nd_t[m, tmp_z[n]] - 1
                            tt.nw_t[v, tmp_z[n]] = tt.nw_t[v, tmp_z[n]] - 1
                            tt.nwSum[tmp_z[n]] = tt.nwSum[tmp_z[n]] - 1
                            self.a0d_t[m, tmp_z[n]] = self.a0d_t[m,tmp_z[n]] - 1
                            self.a0wSum[tmp_z[n]] = self.a0wSum[tmp_z[n]] - 1

                            # # ()1:k + 1:k) *(1:k + 1:K)/()
                            ##
                            tmp_p0= (self.a0d_t[m, :] + self.alpha) * (tt.nw_t[v, :] + tt.beta[v]) / (np.sum(tt.nw_t) + np.sum(tt.beta))
                            self.p1 = tmp_p0/np.sum(tmp_p0)

                            # print self.p1

                            tmp_z0[n] = cumDicing(self.p1)[0]
                            tt.nd_t[m, tmp_z0[n]] = tt.nd_t[m,tmp_z0[n]] + 1
                            tt.nw_t[v, tmp_z0[n]] = tt.nw_t[v,tmp_z0[n]] + 1
                            tt.nwSum[tmp_z0[n]] = tt.nwSum[tmp_z0[n]] + 1
                            self.a0d_t[m, tmp_z0[n]] = self.a0d_t[m,tmp_z0[n]] + 1
                            self.a0wSum[tmp_z0[n]] = self.a0wSum[tmp_z0[n]] + 1

                            
                        tt.z[m, v] = ''  ###debug here
                        tt.z[m, v] = ','.join(list(map(str, tmp_z0)))

            totlll -= tt.l
            tt.loglikelihood()
            totlll += tt.l

            self.iteration +=1
            if self.intermediate:
                self.output()
                
        print("mLDA LogLikelihood: %8.3f" % totlll)
        self.llhoutfh.write("Loglikelihood for iter:%d\t%8.3f\n"% (self.iteration , totlll))
        self.llhoutfh.flush()
        
        
    def output(self):
        
        ###output the current information
        ###self.intermediate
        out = self.out
        self.calc_phi()
        self.calc_theta()
        np.savetxt(out + '-' + self.group + '-' +str(self.iteration)+ '.mlda.theta.txt', self.theta, delimiter=',')
        for tt0 in self.ldaDict:
            print(str(tt0))
            tt = self.ldaDict[tt0]
            np.savetxt(out + '-' + self.group + '-' + tt0 +'-'+str(self.iteration)+ '.nw_t.txt', tt.nw_t, delimiter=',')
            # nw_t
            np.savetxt(out + '-' + self.group + '-' + tt0+'-'+str(self.iteration)+ '.nd_t.txt', tt.nd_t, delimiter=',')
            np.savetxt(out + '-' + self.group + '-' + tt0 +'-'+str(self.iteration)+ '.ndSum.txt', tt.ndSum,delimiter=',')
            np.savetxt(out + '-' + self.group + '-' + tt0 +'-'+str(self.iteration)+ '.nwSum.txt', tt.nwSum,delimiter=',')
            np.savetxt(out + '-' + self.group + '-' + tt0 +'-'+str(self.iteration)+ '.phi.txt', tt.phi, delimiter=',')
                    
    def calc_theta(self):
        for m in range(self.ndoc):
            # #1:k
            self.theta[m,:] = (self.a0d_t[m, :] + self.alpha) \
                / (np.sum(self.a0dSum[m]) + np.sum(self.alpha))

    def linking():
        '''link each type to multiple doc
           theta(n x k ) * k by v1 = > n x v1, n x v2
            n x v2;
            v1 x v2 for each 
        '''

    def infer(self):
        '''infer doc topic 
        
        '''

    def sumUp():
        '''final sumup
        
        '''


if __name__ == '__main__':
    import sys
    shuffle=0
    if len(sys.argv) != 9:
        print('Usage: ' + sys.argv[0] + ' topic_number alpha beta metafile folder_ds(name is type) iter_for_learn shuffle-flag output-prefix \n')
        sys.exit()

    k0 = int(sys.argv[1])  # #k topic
    a0 = float(sys.argv[2])  # alpha
    b0 = float(sys.argv[3])  # beta
    m0 = sys.argv[4]  # meta info file
    d0 = sys.argv[5]  # folder for dataset
    nt0 = int(sys.argv[6])  # # n iteration
    shuffle=int(sys.argv[7])
    out = sys.argv[8]

    comstr=' '.join(sys.argv)
    ###generate shuffle folder
    outdir=os.getcwd() 
    if shuffle>0:
        outdir = outdir+'/shuffle/s'+str(shuffle)
    else:
        outdir = outdir+'/'+out

    if not exists(outdir):
        makedirs(outdir)
            
    runfn=outdir+"/000run."+out+".txt"
    out = outdir+"/"+out
    
    # ####doc name list with folder
    ff = open(runfn,'w')
    ff.write('Usage: ' + sys.argv[0] + ' topic_number alpha beta metafile folder_ds(name is type) iter_for_learn shuffle-flag output-prefix\n')
    ff.write(comstr)
    ff.close()
    
    main(
        k0,
        a0,
        b0,
        m0,
        d0,
        nt0,
        shuffle,
        out
        )


			