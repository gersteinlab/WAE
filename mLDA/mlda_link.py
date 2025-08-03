#!/usr/bin/python
'''
log:
todo Jan 18 2021:
add distance correlation
add logistic transformation then use pearson correlation
'''
import glob
#from dcor import distance_correlation
import scipy.stats
#from dcor import independence
import numpy as np
import copy
from scipy import stats
import pickle
from memory_profiler import profile
import os
class Link2type:
    '''
    n1>n2
    x2 as the ref
    '''
    def __init__(
        self,
        x1,
        x2,
        n1,
        n2,
        n0,
        st,
        ed
    ):

        self.n1 = n1
        self.n2 = n2
        self.x1 = x1
        self.x2 = x2
        self.n0 = n0
        self.st=st
        self.ed=ed
        assert ed > st and ed <=n1
        n10 = ed-st ###only subset
        #self.mat = np.zeros((n10, n2, n0),dtype=float)
        self.mat0 = np.zeros((n10,n2),dtype=float)
        self.mat1 = np.zeros((n10, n2, n0),dtype=float)


bglinkObj=[] ##background
def vecmode(x1mat,x2mat):
    xxmode=np.matmul(np.sqrt(np.sum(x1mat*x1mat, axis=1)).reshape(-1,1) , np.sqrt(np.sum(x2mat*x2mat, axis=1)).reshape(1,-1))
    return xxmode

###july 29, add indict for indicatorDict
def getlinkComb(indict, phiDict,theta,nt,foldid,nfold, exp1=0):
    """
    Args:
        phiDict (dictionary): Phi dictionary for all the types, vxk
        theta (str): theta matrix for the document, nxk
        foldid: 1:nfold
        nfold: split the big matrix to nfold to avoid the matrix is too big and eatup the memory

    Returns:
        linkobj
 

    """
    tname = list(phiDict)
    linkObj=[]  ###
                
    for x in range(nt-1):
        for y in range(x+1, nt):
            x1 = tname[x]
            x2 = tname[y]
            n1 = phiDict[x1].shape[0]
            n2 = phiDict[x2].shape[0]
            n0 = theta.shape[0]  #the number of document
                
            if n1 < n2:
                ##n1 should > n2, use n1 to split 
                n3=n1
                n1=n2
                n2=n3
                x1=tname[y]
                x2=tname[x]
                
            interval = int(n1/nfold)+1

            start = (foldid-1)*interval
            end = foldid*interval if  foldid*interval< n1 else n1

            
            link2t = Link2type(x1, x2, n1, n2, n0, start, end)
            #print("id:", str(foldid),  "tname1:",x1, "tname2:",x2, ' n1:',str(n1), ' n2:',str(n2), ' n0:', n0, ' start:', start, ' end:', end)
            ##july 29
            x1mat=phiDict[x1][start:end,:] 
            x2mat=phiDict[x2]
            # ## n x V1
            #nv1 = np.matmul(theta, np.transpose(phiDict[x1]))
            
            ##axis =0 column, 1 for row;  nx x v     nx x1           1x n  = nx x n 
            x1x2mode=np.matmul(np.sqrt(np.sum(x1mat*x1mat, axis=1)).reshape(-1,1) , np.sqrt(np.sum(x2mat*x2mat, axis=1)).reshape(1,-1))
            # ,....
            for z in range(n0): ###
                ##n1xn2
                #link2t.mat[:, :, z] =   np.matmul(theta[z,:]*x1mat, np.transpose(x2mat))
                ##mode n1 x k, n2x k = n1     1x k  * nx x k , k x v = (nx * n) / (nx x n)

                x1x2indict=np.matmul(indict[x1][z,start:end].reshape(-1,1),   indict[x2][z,:].reshape(1,-1))
                #x1x2indict2=np.where(x1x2indict==0, )

                with np.errstate(divide='ignore'):
                    #print("Version July 23,2021")
                    #print("Version July 25,2021")

                    if exp0==1: ##squre  run2
                        ##july 15
                        #link2t.mat1[:, :, z] =  np.matmul(theta[z,:]*x1mat, np.transpose(theta[z,:]*x2mat))/vecmode(theta[z,:]*x1mat,theta[z,:]*x2mat)      ##link2t.mat1 in Nx X N X Z , where N_x = end=start, N = N2 (smaller one)
                        ##july 23
                        link2t.mat1[:, :, z] =x1x2indict*(np.matmul(theta[z,:]*x1mat, np.transpose(theta[z,:]*x2mat)))      ##link2t.mat1 in Nx X N X Z , where N_x = end=start, N = N2 (smaller one)

                    elif exp0==2:   #run2 
                        #july 15
                        #link2t.mat1[:, :, z] =  np.matmul(theta[z,:]*x1mat, np.transpose(x2mat))/vecmode(theta[z,:]*x1mat, x2mat)      ##link2t.mat1 in Nx X N X Z , where N_x = end=start, N = N2 (smaller one)
                        #july 23
                        link2t.mat1[:, :, z] = x1x2indict*( np.matmul(theta[z,:]*x1mat, np.transpose(x2mat)) )     ##link2t.mat1 in Nx X N X Z , where N_x = end=start, N = N2 (smaller one)

                    elif exp0==3: ##cross entropy
                        link2t.mat1[:, :, z] = ( np.matmul(theta[z,:]*x1mat, np.log10(np.transpose(x2mat))) )
                        
                    elif exp0==4:##adjusted cosine similarity
                        link2t.mat1[:, :, z] = np.matmul(theta[z,:]*(x1mat-np.mean(x1mat, axis=0).reshape(1,-1) ) , np.transpose(x2mat - np.mean(x2mat,axis=0).reshape(1,-1)))/vecmode(x1mat-np.mean(x1mat, axis=0).reshape(1,-1) ,  (x2mat - np.mean(x2mat,axis=0).reshape(1,-1) )) 

                    elif exp0==5:
                        link2t.mat1[:, :, z] =   (np.matmul(theta[z,:]*(x1mat-np.mean(x1mat, axis=0).reshape(1,-1) ), np.transpose(theta[z,:]*(x2mat - np.mean(x2mat,axis=0).reshape(1,-1))))/ vecmode(theta[z,:]*(x1mat-np.mean(x1mat, axis=0).reshape(1,-1)) , theta[z,:]* (x2mat - np.mean(x2mat,axis=0).reshape(1,-1) ))   )  ##link2t.mat1 in Nx X N X Z , where N_x = end=start, N = N2 (smaller one)


                    elif exp0==6:##adjusted cosine similarity
                        link2t.mat1[:, :, z] = np.matmul(theta[z,:]*(x1mat ) , np.transpose(x2mat  ) )/vecmode(x1mat ,  x2mat) 

                    elif exp0==7:
                        link2t.mat1[:, :, z] =   (np.matmul(theta[z,:]*(x1mat), np.transpose(theta[z,:]*(x2mat)))/ vecmode(theta[z,:]*(x1mat  ) , theta[z,:]* (x2mat)) )  ##link2t.mat1 in Nx X N X Z , where N_x = end=start, N = N2 (smaller one)

                    elif exp0==8:## cosine similarity   run2
                        link2t.mat1[:, :, z] = x1x2indict* np.matmul(theta[z,:]*(x1mat ) , np.transpose(x2mat  ) )/vecmode(x1mat ,  x2mat) 

                    elif exp0==9: ### run2 
                        link2t.mat1[:, :, z] = x1x2indict*  (np.matmul(theta[z,:]*(x1mat), np.transpose(theta[z,:]*(x2mat)))/ vecmode(theta[z,:]*(x1mat  ) , theta[z,:]* (x2mat)) )  ##link2t.mat1 in Nx X N X Z , where N_x = end=start, N = N2 (smaller one)

                    elif exp0==10: ##greater, the better
                        link2t.mat1[:, :, z] = ( np.matmul(theta[z,:]*x1mat, np.log10(np.transpose(x2mat))) ) + np.matmul(theta[z,:]*(1.0-x1mat), np.log10(np.transpose(1.0-x2mat))) 

                   # elif exp0==11: #kl divergence:
                   #     link2t.mat1[:, :, z] = ( np.matmul(theta[z,:]*x1mat, np.log10(np.transpose(x2mat))) ) + np.matmul(theta[z,:]*(1-x1mat), np.log10(np.transpose(1.0-x2mat))) 

                    else:
                        ##july 15
                        #link2t.mat1[:, :, z] =  np.matmul(theta[z,:]*x1mat, np.transpose(x2mat))/x1x2mode                        
                        #july 23, adjust cosine
                        link2t.mat1[:, :, z] =  np.matmul(theta[z,:]*x1mat, np.transpose(x2mat))/x1x2mode
                        



                        
            with np.errstate(divide='ignore'):
                ##July 15
                #link2t.mat0 = np.matmul(x1mat, np.transpose(x2mat))/x1x2mode ### Nx X N
                ##July 23
                if exp0==1:##cross netropy
                    link2t.mat0 = np.matmul(x1mat, np.transpose(x2mat))
                elif exp0==2:
                    link2t.mat0 = np.matmul(x1mat, np.log10(np.transpose(x2mat)))
                elif exp0==4:
                    link2t.mat0 = np.matmul(x1mat - np.mean(x1mat, axis=0).reshape(1,-1), np.transpose( (x2mat - np.mean(x2mat,axis=0).reshape(1,-1))))/vecmode(x1mat-np.mean(x1mat, axis=0).reshape(1,-1) ,  (x2mat - np.mean(x2mat,axis=0).reshape(1,-1) ))
                else:
                    link2t.mat0 = np.matmul(x1mat, np.transpose(x2mat))
                    
            linkObj.append(link2t)

    return linkObj



def getlinkComb_dcor(phiDict,theta,nt, foldid, nfold, pval=True):
    """
    Args:
        phiDict (dictionary): Phi dictionary for all the types, vxk
        theta (str): theta matrix for the document, nxk
        foldid: 1:nfold
        nfold: split the big matrix to nfold to avoid the matrix is too big and eatup the memory
        pval: if True, will directly calculate pvalue, or will just calculate correlation and then test with random set.
    Returns:
        linkobj
 

    """
    tname = list(phiDict)
    linkObj=[]  ###
                
    for x in range(nt-1):
        for y in range(x+1, nt):
            x1 = tname[x]
            x2 = tname[y]
            n1 = phiDict[x1].shape[0]
            n2 = phiDict[x2].shape[0]
            n0 = theta.shape[0]  #the number of document
                
            if n1 < n2:
                ##n1 should > n2, use n1 to split 
                n3=n1
                n1=n2
                n2=n3
                x1=tname[y]
                x2=tname[x]
                
            interval = int(n1/nfold)+1

            start = (foldid-1)*interval
            end = foldid*interval if  foldid*interval< n1 else n1
            
            link2t = Link2type(x1, x2, n1, n2, n0, start, end)

            x1mat=phiDict[x1][start:end,:]
            x2mat=phiDict[x2]
            # ## n x V1
            #nv1 = np.matmul(theta, np.transpose(phiDict[x1]))

            ##axis =0 column, 1 for row;  nx x v     nx x1           1x n  = nx x n 
            #x1x2mode=np.matmul(np.sqrt(np.sum(x1mat*x1mat, axis=1)).reshape(-1,1) , np.sqrt(np.sum(x2mat*x2mat, axis=1)).reshape(1,-1))
            
            for z in range(n0): ###
                ##n1xn2
                #link2t.mat[:, :, z] =   np.matmul(theta[z,:]*x1mat, np.transpose(x2mat))
                ##mode n1 x k, n2x k = n1     1x k  * nx x k , k x v = (nx * n) / (nx x n) 
                #link2t.mat1[:, :, z] =  np.matmul(theta[z,:]*x1mat, np.transpose(x2mat))/x1x2mode      ##link2t.mat1 in Nx X N X Z , where N_x = end=start, N = N2 (smaller one)
                print("comment codes below because dcor cannot be imported correctly")
                #if pval:
                #    with np.errstate(divide='ignore'):
                #        link2t.mat1[:,:, z] =  np.reshape([ np.array(independence.distance_correlation_t_test(a1*theta[z,:],b1*theta[z,:]))[0] for a1 in x1mat for b1 in x2mat], ( (end-start), n2))
                #else:
                #    with np.errstate(divide='ignore'):
                #        link2t.mat1[:,:, z] =  np.reshape([ distance_correlation(a1*theta[z,:],b1*theta[z,:]) for a1 in x1mat for b1 in x2mat], ( (end-start), n2))
                
                #np.array(dcor.independence.distance_correlation_t_test(a0[1,:], b0[0,:]))[0])
            #link2t.mat0 = np.matmul(x1mat, np.transpose(x2mat))/x1x2mode ### Nx X N
            #if pval:
            #    with np.errstate(divide='ignore'):
            #        link2t.mat0 = np.reshape([ np.array(independence.distance_correlation_t_test(a1,b1))[0] for a1 in x1mat for b1 in x2mat], ( (end-start), n2))
            #else:
            #    with np.errstate(divide='ignore'):
            #        link2t.mat0 = np.reshape([ distance_correlation(a1,b1) for a1 in x1mat for b1 in x2mat], ( (end-start), n2))
                
            #linkObj.append(link2t)

    return linkObj

def getlinkComb_pcor(phiDict,theta,nt, foldid, nfold, pval=True):
    """
    Args:
        phiDict (dictionary): Phi dictionary for all the types, vxk
        theta (str): theta matrix for the document, nxk
        foldid: 1:nfold
        nfold: split the big matrix to nfold to avoid the matrix is too big and eatup the memory
        pval: if True, will directly calculate pvalue, or will just calculate correlation and then test with random set.
    Returns:
        linkobj
 

    """
    tname = list(phiDict)
    linkObj=[]  ###
                
    for x in range(nt-1):
        for y in range(x+1, nt):
            x1 = tname[x]
            x2 = tname[y]
            n1 = phiDict[x1].shape[0]
            n2 = phiDict[x2].shape[0]
            n0 = theta.shape[0]  #the number of document
                
            if n1 < n2:
                ##n1 should > n2, use n1 to split 
                n3=n1
                n1=n2
                n2=n3
                x1=tname[y]
                x2=tname[x]
                
            interval = int(n1/nfold)+1

            start = (foldid-1)*interval
            end = foldid*interval if  foldid*interval< n1 else n1
            
            link2t = Link2type(x1, x2, n1, n2, n0, start, end)

            x1mat=phiDict[x1][start:end,:]
            x2mat=phiDict[x2]
            # ## n x V1
            #nv1 = np.matmul(theta, np.transpose(phiDict[x1]))

            ##axis =0 column, 1 for row;  nx x v     nx x1           1x n  = nx x n 
            #x1x2mode=np.matmul(np.sqrt(np.sum(x1mat*x1mat, axis=1)).reshape(-1,1) , np.sqrt(np.sum(x2mat*x2mat, axis=1)).reshape(1,-1))
            
            for z in range(n0): ###
                ##n1xn2
                #link2t.mat[:, :, z] =   np.matmul(theta[z,:]*x1mat, np.transpose(x2mat))
                ##mode n1 x k, n2x k = n1     1x k  * nx x k , k x v = (nx * n) / (nx x n) 
                #link2t.mat1[:, :, z] =  np.matmul(theta[z,:]*x1mat, np.transpose(x2mat))/x1x2mode      ##link2t.mat1 in Nx X N X Z , where N_x = end=start, N = N2 (smaller one)
                if pval:
                    with np.errstate(divide='ignore'):
                        link2t.mat1[:,:, z] =  np.reshape([ scipy.stats.spearmanr(a1*theta[z,:],b1*theta[z,:])[1] for a1 in x1mat for b1 in x2mat], ( (end-start), n2))
                    #np.reshape([scipy.stats.spearmanr(a1,b1)[1] for a1 in a0 for b1 in b0],(4,3))
                else:
                    with np.errstate(divide='ignore'):
                        link2t.mat1[:,:, z] =  np.reshape([ scipy.stats.spearmanr(a1*theta[z,:],b1*theta[z,:])[0] for a1 in x1mat for b1 in x2mat], ( (end-start), n2))
                
                #np.array(dcor.independence.distance_correlation_t_test(a0[1,:], b0[0,:]))[0])
            #link2t.mat0 = np.matmul(x1mat, np.transpose(x2mat))/x1x2mode ### Nx X N
            if pval:
                with np.errstate(divide='ignore'):
                    link2t.mat0 = np.reshape([ scipy.stats.spearmanr(a1,b1)[1] for a1 in x1mat for b1 in x2mat], ( (end-start), n2))
            else:
                with np.errstate(divide='ignore'):
                    link2t.mat0 = np.reshape([ scipy.stats.spearmanr(a1,b1)[0] for a1 in x1mat for b1 in x2mat], ( (end-start), n2))
                
            linkObj.append(link2t)

    return linkObj



    
def getlinkbg(phiDict,theta,nt, foldid, nfold):
    """
    get the linkbg matrix for calculate runing mean and 
    Args:
        phiDict (dictionary): Phi dictionary for all the types, vxk
        theta (str): theta matrix for the document, nxk
    

    Returns:
        linkobj

    """
    tname = list(phiDict)
    linkObj=[]  ###
                
    for x in range(nt-1):
        for y in range(x+1, nt):
            x1 = tname[x]
            x2 = tname[y]
            n1 = phiDict[x1].shape[0]
            n2 = phiDict[x2].shape[0]
            n0 = theta.shape[0]  #the number of document
            if n1 < n2:
                ##n1 should > n2, use n1 to split 
                n3=n1
                n1=n2
                n2=n3
                x3=x1
                x1=x2
                x2=x3
                
            interval = int(n1/nfold)+1

            start = (foldid-1)*interval
            end = foldid*interval if  foldid*interval< n1 else n1
            
            
            link2t = Link2type(x1, x2, n1, n2, n0,start,end)
            
            linkObj.append(link2t)

    return linkObj

def runningStat2(l0, l00, l01, l02, l1,  idx1, tot0):
    """ calculate running mean and std for shuffle data
    Args:
        l0 linkobj for background with everything is zero
        l1 the shuffle data link
        l00 linkobj background for previous-old
        l01  linkobj var
        l02 linkobj varold
        idx0 the index for running, each time the count is different depend on the value of linkgb
    """
    if len(l0) != len(l1):
        return

    nn= len(l0)
    #l00 = copy.deepcopy(l0)
    #l01 = copy.deepcopy(l0)
    #l02 = copy.deepcopy(l0)

    for idx in range(nn):
        bgx = l0[idx]
        x = l1[idx]
        oldbgx=l00[idx]
        varx = l01[idx]
        varx2 = l02[idx]
        idx0 = idx1[idx]
        
        if tot0 ==0:
            bgx.mat0 = copy.deepcopy(x.mat0)
            oldbgx.mat0 = copy.deepcopy(x.mat0)
            #bgx.mat = copy.deepcopy(x.mat)
            #oldbgx.mat = copy.deepcopy(x.mat)
            bgx.mat1 = copy.deepcopy(x.mat1)
            oldbgx.mat1 = copy.deepcopy(x.mat1)
            idx0.mat0 = np.where(x.mat0 ==0, idx0.mat0, idx0.mat0+1)
            idx0.mat1 = np.where(x.mat1 ==0, idx0.mat1, idx0.mat1+1)
            
        else:
            #bgx.mat = oldbgx.mat + (x.mat-oldbgx.mat)/(idx0+1)
            #varx.mat = varx2.mat + (x.mat - oldbgx.mat)*(x.mat-bgx.mat)

            ###update old
            #oldbgx.mat = copy.deepcopy(bgx.mat)
            #varx2.mat = copy.deepcopy(varx.mat)

            #####mat0
            #bgx.mat0 = oldbgx.mat0 + (x.mat0-oldbgx.mat0)/(idx0+1)
            #varx.mat0 = varx2.mat0 + (x.mat0 - oldbgx.mat0)*(x.mat0-bgx.mat0)
            idx0.mat0 = np.where(x.mat0 ==0, idx0.mat0, idx0.mat0+1)
            with np.errstate(invalid='ignore', divide='ignore'):
                bgx.mat0 =np.where(x.mat0==0, oldbgx.mat0, oldbgx.mat0 + (x.mat0-oldbgx.mat0)/idx0.mat0 )
            
            varx.mat0 = np.where(x.mat0==0, varx2.mat0, varx2.mat0 + (x.mat0 - oldbgx.mat0)*(x.mat0-bgx.mat0) )



            
            ###update old, no need to change for zero
            oldbgx.mat0 = copy.deepcopy(bgx.mat0)
            varx2.mat0 = copy.deepcopy(varx.mat0)

            ###mat1
            idx0.mat1 = np.where(x.mat1 ==0, idx0.mat1, idx0.mat1+1)
            with np.errstate(invalid='ignore', divide='ignore'):
                bgx.mat1 = np.where(x.mat1==0, oldbgx.mat1,  oldbgx.mat1 + (x.mat1-oldbgx.mat1)/(idx0.mat1) )
            varx.mat1 = np.where(x.mat1==0, varx2.mat1, varx2.mat1 + (x.mat1 - oldbgx.mat1)*(x.mat1-bgx.mat1) )

            ###update old, no need to change
            oldbgx.mat1 = copy.deepcopy(bgx.mat1)
            varx2.mat1 = copy.deepcopy(varx.mat1)

        


def runningStat(l0, l00, l01, l02, l1, idx0):
    """ calculate running mean and std for shuffle data
    Args:
        l0 linkobj for background with everything is zero
        l1 the shuffle data link
        l00 linkobj background for previous-old
        l01  linkobj var
        l02 linkobj varold
        idx0 the index for running
    """
    if len(l0) != len(l1):
        return

    nn= len(l0)
    #l00 = copy.deepcopy(l0)
    #l01 = copy.deepcopy(l0)
    #l02 = copy.deepcopy(l0)

    for idx in range(nn):
        bgx = l0[idx]
        x = l1[idx]
        oldbgx=l00[idx]
        varx = l01[idx]
        varx2 = l02[idx]
        
        if idx0==0:
            bgx.mat0 = copy.deepcopy(x.mat0)
            oldbgx.mat0 = copy.deepcopy(x.mat0)
            #bgx.mat = copy.deepcopy(x.mat)
            #oldbgx.mat = copy.deepcopy(x.mat)
            bgx.mat1 = copy.deepcopy(x.mat1)
            oldbgx.mat1 = copy.deepcopy(x.mat1)
        else:
            #bgx.mat = oldbgx.mat + (x.mat-oldbgx.mat)/(idx0+1)
            #varx.mat = varx2.mat + (x.mat - oldbgx.mat)*(x.mat-bgx.mat)

            ###update old
            #oldbgx.mat = copy.deepcopy(bgx.mat)
            #varx2.mat = copy.deepcopy(varx.mat)

            #####mat0
            bgx.mat0 = oldbgx.mat0 + (x.mat0-oldbgx.mat0)/(idx0+1)
            varx.mat0 = varx2.mat0 + (x.mat0 - oldbgx.mat0)*(x.mat0-bgx.mat0)
            #bgx.mat0 =np.where(x.mat0==0, oldbgx.mat0, oldbgx.mat0 + (x.mat0-oldbgx.mat0)/(idx0+1))
            #varx.mat0 = varx2.mat0 + (x.mat0 - oldbgx.mat0)*(x.mat0-bgx.mat0)



            
            ###update old, no need to change for zero
            oldbgx.mat0 = copy.deepcopy(bgx.mat0)
            varx2.mat0 = copy.deepcopy(varx.mat0)

            ###mat1
            bgx.mat1 = oldbgx.mat1 + (x.mat1-oldbgx.mat1)/(idx0+1)
            varx.mat1 = varx2.mat1 + (x.mat1 - oldbgx.mat1)*(x.mat1-bgx.mat1)

            ###update old, no need to change
            oldbgx.mat1 = copy.deepcopy(bgx.mat1)
            varx2.mat1 = copy.deepcopy(varx.mat1)

        
def flogit(x):
    return np.exp(x)/sum(np.exp(x))

##@profile, no
def main(path, prefix1, prefix2, K, nfold, foldid, out,  end=2,cor="orig", logit=True, onlineShuffle=True, pvflag= False, exp=0):
    '''
    read the file list and data
    *.theta.*
    *.nV
    *ndoc
    *ibd.iter3.0701-(xxx).phi.txt
    #Jan19 2021
    add a logistic norm transformation

    new param:
    onlineShuff: True or False
    logit: True or False
    cor:["dcor", "pcor","orig"]
    pvflag: whether to calculate pvalue

    onlineShuffle True, pvflag=False, cor="orig"
    onlineShuffle False, pvflag = True, cor="dcor" or "pcor"

    default orig: cor="orig", onlineShuffle=True, pvflag=False, logit=True or False
    
    
    
    '''

    ###get the data
    import re

    ##online shuffle only works on the orignal mode, not pval, and dcor
    if onlineShuffle:
        pvflag=False
        cor="orig"

        
    #testPhi=glob.glob(path+'/*.phi.txt') #ibd.iter3.0701-ibd.phi.txt
    #shufflePhi=glob.glob(path+'/shuffle/*.phi.txt') #shuffle/ibd.shuffle.10-microbe.phi.txt 
    
    testTheta=glob.glob(path+'/*.theta.txt') #ibd.iter3.0701-mlda.theta.txt
    #shuffleTheta=glob.glob(path+'/shuffle/*.theta.txt') #shuffle/ibd.shuffle.10-mlda.theta.txt

    ###test names
    testphi=glob.glob(path+'/*.phi.txt') #ibd.iter3.0701-mlda.theta.txt
    ###here the format has changed again
    tnames=[re.findall('[^-]+-([^\.-]+)\.phi\.txt',x)[0] for x in  testphi]

    #testraw=glob.glob(path+'/*.raw.txt')  ##hostgene.raw.txt; microbeR.raw.txt; premiR.raw.txt

    
    phidict={}
    #thetadict={}
    
    #theta, row: sample, column, topic
    theta=np.genfromtxt(testTheta[0], dtype=float, delimiter=',')
    ##logistic convertion

    if logit:
        theta= np.apply_along_axis(flogit, 1, theta)
        
    meta={}
    phimean={}
    indicator={}
    link4test=None
    for type_name,fname in zip(tnames, testphi): ##type name, host-gene, microbe
        ###
        
        #type_name=re.findall('[^-]+-([^\.]+)\.phi\.txt', fname)[0]
        phidict[type_name]=np.genfromtxt(fname, dtype=float, delimiter=',')
        if logit:
            phidict[type_name] = np.apply_along_axis(flogit, 0, phidict[type_name])
        phimean[type_name] = np.mean(phidict[type_name], axis=0)  ##
        #phidict[type_name] = phidict[type_name] - phimean[type_name]
        print(phimean[type_name])
        meta[type_name] = np.genfromtxt(prefix1+"/"+type_name+".meta.txt", dtype=str, delimiter='\n')
        ##harcoding here July 29, need  *_m1.txt
        indicator[type_name]= np.genfromtxt(prefix1+"/data/"+type_name+"_m1.txt", dtype=int, delimiter='\t')
        indicator[type_name]=indicator[type_name][:,1:]
        oneindex=np.where(indicator[type_name]>0) #set all the non-negative to be 1
        indicator[type_name][oneindex] = 1 
        
        
        
    ###get the links for the testing dataset, here need to change
    if cor == "dcor":
        link4test =getlinkComb_dcor(phidict, theta, len(tnames),foldid, nfold, pval=pvflag)
    elif cor=="pcor":
        link4test =getlinkComb_pcor(phidict, theta, len(tnames),foldid, nfold, pval=pvflag)
    else:
        ##July29
        link4test =getlinkComb(indicator,phidict, theta, len(tnames),foldid, nfold, exp1=exp)


        ###output
    if not os.path.exists(out+'/allpv1'):
        os.makedirs(out+'/allpv1')
        
    if not os.path.exists(out+'/allpv0'):
        os.makedirs(out+'/allpv0')

    if not os.path.exists(out+'/signif'):
        os.makedirs(out+'/signif')

    if not os.path.exists(out+'/allraw'):
        os.makedirs(out+'/allraw')    


    ##pvflag is True, then no need to do further analysis
    if pvflag:
        
        if cor!="dcor" and cor!="pcor":
            raise ValueError("cor must be dcor for distance-correlation, pcor for pearson correlation when pvflag is True")
        
        saveobj=(link4test)
        with open(out+"/"+out+".dump.pikle", 'wb') as f:
            pickle.dump( saveobj, f)
        f.close()


            
        for idx in range(len(link4test)):
            x = link4test[idx]
            #mean= linkbg[idx]
            #var = linkbgvar[idx]
            name1=x.x1
            name2=x.x2
            x1meta=meta[name1][x.st:x.ed]
            x2meta=meta[name2] ###all the 
        
            #linkpv[idx].mat = 1-stats.t.cdf((x.mat - mean.mat)/np.sqrt(var.mat), df=K-1)
            #linkpv[idx].mat0 = 1-stats.t.cdf((x.mat0 - mean.mat0)/np.sqrt(var.mat0), df=K-1)
            #linkpv[idx].mat1 = 1-stats.t.cdf((x.mat1 - mean.mat1)/np.sqrt(var.mat1), df=K-1)
        
            ##output all the pv
            #with open(out+'.'+name1+"-"+name2+'.'+str(x.st)+'-'+str(x.ed)+'.allpv.txt', 'w') as f:
            with open(out+"/allpv0/"+out+'.'+name1+"-"+name2+'.allpv0.txt', 'a+') as f:
                #np.savetxt(out+"/allpv0/"+out+'.'+name1+"-"+name2+'.'+str(x.st)+'-'+str(x.ed)+'.allpv0.txt', link4test[idx].mat0, delimiter='\t')
                np.savetxt(f, np.c_[range(x.st,x.ed),link4test[idx].mat0 ],delimiter='\t')
            f.close()
            #np.savetxt(out+'.'+name1+"-"+name2+'.'+str(x.st)+'-'+str(x.ed)+'.allpv1.txt', linkpv[idx].mat1, delimiter='\t')
            #f.close()
     
            mat_pidx = np.where(link4test[idx].mat0 < 0.005)
            #with open(out+'/signif/'+out+'.'+name1+"-"+name2+'.'+str(x.st)+'-'+str(x.ed)+'.significance.txt', 'ab') as f:
            with open(out+'/signif/'+out+'.'+name1+"-"+name2+'.significance.txt', 'a+') as f:
                x1idx=mat_pidx[0]
                x2idx=mat_pidx[1]
            
                for ddd in list(range(x1idx.shape[0])):
                    idx1 = x1idx[ddd]
                    idx2 = x2idx[ddd]
                    f.write("%s\t%s\t%s\t%s\t%f\n"%(name1,name2, x1meta[idx1], x2meta[idx2], link4test[idx].mat0[idx1,idx2]))
            f.close()

            for idx30 in range(link4test[idx].mat1.shape[2]):
                with open(out+'/allpv1/'+out+'.'+name1+"-"+name2+'.'+ str(idx30)+'.allpv1.txt','a+') as f:
                #np.savetxt(out+'/allpv1/'+out+'.'+name1+"-"+name2+'.'+ str(idx30)+'.'+str(x.st)+'-'+str(x.ed)+'.allpv1.txt', link4test[idx].mat1[:,:,idx30], delimiter='\t')
                    np.savetxt(f, np.c_[range(x.st,x.ed), link4test[idx].mat1[:,:,idx30] ])
                f.close()

                
                                            
            mat_pidx = np.where(link4test[idx].mat1 < 0.005)
            #with open(out+'/signif/'+out+'.'+name1+"-"+name2+'.'+str(x.st)+'-'+str(x.ed)+'.significance_docNorm.txt', 'w') as f:
            with open(out+'/signif/'+out+'.'+name1+"-"+name2+'.significance_docNorm.txt', 'a+') as f:
                x1idx=mat_pidx[0]
                x2idx=mat_pidx[1]     
                x3idx=mat_pidx[2] ##ndoc, sample, patient
            
                for ddd in list(range(x1idx.shape[0])):
                    idx1 = x1idx[ddd]
                    idx2 = x2idx[ddd]
                    idx3 = x3idx[ddd]
                    f.write("%s\t%s\t%s\t%s\t%d\t%f\n"%(name1,name2, x1meta[idx1], x2meta[idx2], idx3, link4test[idx].mat1[idx1,idx2,idx3]))
            f.close()
        
        
                ###write the files

        
        return
    
    ###define aux obj for shuffling dataset to get the background value
    linkbg = getlinkbg(phidict, theta, len(tnames), foldid, nfold)
    linkbg0 = copy.deepcopy(linkbg)
    linkbgvar = copy.deepcopy(linkbg)
    linkbgvar0 = copy.deepcopy(linkbg)
    linkpv = copy.deepcopy(linkbg)
    linkidx0 = copy.deepcopy(linkbg)

    ###here may change, no need to do 1000 times shuffleing
    totn=0  ##
     ##todo shuffle times
    ###for x in `seq 1 20`;  do for y in `find s$x -name "*27*"`; do newy=`echo $y|perl -lane '$_=~s/-27//g;print $_;'` ; ydir=$(dirname $y); mkdir fs/$ydir; ln -s `pwd`/$y `pwd`/fs/$newy; done; done
    ##cd non-shuffle
    ### mkdir orig; mv *.txt orig
    ###for y in `find orig -name "*27*"`; do  yn=$(basename $y); newy=`echo $yn|perl -lane '$_=~s/-27//g; print $_'`; ln -s `pwd`/$y `pwd`/$newy  ; done


    '''
    update Jan 20, 2021: shuffling from the original dataset, no need to shuffling data to get redo mLDA
    onlineshuff means 

    '''
    #onlineShuff=True
    #end= 1000 ##by default, can change

    if onlineShuffle:
        #theta, phi, and indictor all shuffled
        for idx in list(range(1, end)):
            ##shuffle, out + '-' + kk(group) + '-' + tt0(type) + '.nw_t.txt'
            ##todo: handle this when has the 's' group
            shufdata={}
            shufmean={}
            #if not os.path.exists(prefix2+'.mlda.theta.txt'):
            ##prefix2 + '/s'+str(idx)+'/'+path+'-m1.mlda.theta.txt'
            thetaTmp=copy.deepcopy(theta)
            
            #if logit:
            #    thetaTmp = np.apply_along_axis(flogit, 1, thetaTmp)

            #thetaTmp = np.transpose(thetaTmp)
            [np.random.shuffle(x) for x in thetaTmp]
            #thetaTmp = np.transpose(thetaTmp)
            
            #thetaTmp=np.transpose(np.random.shuffle(np.transpose(thetaTmp)))

            shufdata = copy.deepcopy(phidict)
            indict0= copy.deepcopy(indicator)
            for type_name in tnames:
                #shufdata[type_name] = np.genfromtxt(prefix2+'-'+type_name+".phi.txt", dtype=float, delimiter=',')
                #-microbeR.phi.txt 
                
                #np.genfromtxt(prefix2 + '/s'+str(idx)+'/'+path+'-m1-'+ type_name+'.phi.txt', dtype=float, delimiter=',')
                #if logit: (no need to redo-this for shuffling
                #    shufdata[type_name] = np.apply_along_axis(logit, 0, shufdata[type_name])
                shufdata[type_name] = np.transpose(shufdata[type_name]) ##ntopic x nword
                [np.random.shuffle(x) for x in shufdata[type_name]] ## shuffle nword for each topic
                shufdata[type_name] = np.transpose(shufdata[type_name]) ###nword x ntopic
            
                shufmean[type_name]=np.mean(shufdata[type_name], axis=0)
                #shufdata[type_name] = shufdata[type_name] - shufmean[type_name]
                ##shuffle
                [np.random.shuffle(x) for x in indict0[type_name] ]     #nsample x nword
        
            ###
            link4shuff=getlinkComb(indict0, shufdata, thetaTmp,  len(tnames), foldid, nfold, exp1=exp)
            ###if 
            if end - 1 > 1:
                ##because set 0 for no-expression and abundance data, mean is relative lower, more significance pairs will find
                #runningStat(linkbg, linkbg0, linkbgvar, linkbgvar0, link4shuff, totn)
                if exp >=3 and exp <= 7 :
                    runningStat(linkbg, linkbg0, linkbgvar, linkbgvar0, link4shuff, totn)

                else:
   
                    runningStat2(linkbg, linkbg0, linkbgvar, linkbgvar0, link4shuff,linkidx0, totn)

                ###calculate the mean and sd
            else: ###calculate the mean and sd for one shuffling data, 
            
                nn= len(linkbg)
                for idx in range(nn):
                    x = link4shuff[idx]
                    #linkbg[idx].mat0 = np.mean(x.mat0)
                    #linkbg[idx].mat1 = np.mean(x.mat1)
                    #linkbgvar[idx].mat0 = np.var(x.mat0)
                    #linkbgvar[idx].mat1 = np.var(x.mat1)
                    linkbg[idx].mat0 =np.true_divide( x.mat0.sum(), (x.mat0!=0).sum())
                    linkbg[idx].mat1 = np.true_divide(x.mat1.sum(), (x.mat1!=0).sum())
                    linkbgvar[idx].mat0 = np.var(x.mat0, where=np.where(x.mat0!=0,True, False))
                    linkbgvar[idx].mat1 = np.var(x.mat1, where=np.where(x.mat1!=0, True, False))                    
            
            totn+=1

         ##will no need to do next

    else:#if onlineshuffle
        
        for idx in list(range(1, end)):
            ##shuffle, out + '-' + kk(group) + '-' + tt0(type) + '.nw_t.txt'
            ##todo: handle this when has the 's' group
            shufdata={}
            shufmean={}
            #if not os.path.exists(prefix2+'.mlda.theta.txt'):
            ##prefix2 + '/s'+str(idx)+'/'+path+'-m1.mlda.theta.txt'
            if not os.path.exists(prefix2 + '/s'+str(idx)+'/'+path+'-m1.mlda.theta.txt'):
                continue  ###covid-host-m-1013-iter10(name)-m1(group)

            thetaTmp=np.genfromtxt(prefix2 + '/s'+str(idx)+'/'+path+'-m1.mlda.theta.txt',
                                        dtype=float, delimiter=',')
            if logit:
                thetaTmp = np.apply_along_axis(flogit, 1, thetaTmp)
            for type_name in tnames:
                #shufdata[type_name] = np.genfromtxt(prefix2+'-'+type_name+".phi.txt", dtype=float, delimiter=',')
                #-microbeR.phi.txt 
                shufdata[type_name] = np.genfromtxt(prefix2 + '/s'+str(idx)+'/'+path+'-m1-'+ type_name+'.phi.txt', dtype=float, delimiter=',')
                if logit:
                    shufdata[type_name] = np.apply_along_axis(flogit, 0, shufdata[type_name])
            
                shufmean[type_name]=np.mean(shufdata[type_name], axis=0)
                #shufdata[type_name] = shufdata[type_name] - shufmean[type_name]

        
            
            link4shuff=getlinkComb(indicator, shufdata, thetaTmp,  len(tnames), foldid, nfold, exp1=exp)

            if end - 1 > 1:

                if exp >=3 and exp <= 7 :
                    runningStat(linkbg, linkbg0, linkbgvar, linkbgvar0, link4shuff, totn)

                else:
   
                    runningStat2(linkbg, linkbg0, linkbgvar, linkbgvar0, link4shuff,linkidx0, totn)

                ###calculate the mean and sd
            else: ###calculate the mean and sd for one shuffling data
            
                nn= len(linkbg)
                for idx in range(nn):
                    x = link4shuff[idx]
                    linkbg[idx].mat0 = np.mean(x.mat0)
                    linkbg[idx].mat1 = np.mean(x.mat1)
                    linkbgvar[idx].mat0 = np.var(x.mat0)
                    linkbgvar[idx].mat1 = np.var(x.mat1)
            
            totn+=1


        
            ###
            #####generate variance for shuffling data
    ##todo here for 
    for idx in range(len(linkbgvar)):
        #linkbgvar[idx].mat = linkbgvar[idx].mat/totn
        #linkbgvar[idx].mat0 = linkbgvar[idx].mat0/totn
        #linkbgvar[idx].mat1 = linkbgvar[idx].mat1/totn
        if exp >=3 and exp <= 7:
            linkbgvar[idx].mat0 = linkbgvar[idx].mat0/totn
            linkbgvar[idx].mat1 = linkbgvar[idx].mat1/totn

        else:
            with np.errstate(invalid='ignore', divide='ignore'):
                linkbgvar[idx].mat0 = np.where(linkidx0[idx].mat0==0, 0, linkbgvar[idx].mat0/linkidx0[idx].mat0)
                linkbgvar[idx].mat1 = np.where(linkidx0[idx].mat1==0, 0,linkbgvar[idx].mat1/linkidx0[idx].mat1 )

            
            ##make sure all the data structure are the same
    assert len(link4test) == len(linkbg) and len(link4test) ==len(linkbgvar)
    
    ###get pvalue for the links
    for idx in range(len(link4test)):
        x = link4test[idx]
        mean= linkbg[idx]
        var = linkbgvar[idx]
        name1=x.x1
        name2=x.x2
        x1meta=meta[name1][x.st:x.ed]
        x2meta=meta[name2] ###all the 

        ##zero dividing issue: np.mean(var.mat1[np.where(var.mat1>0)])
        #linkpv[idx].mat = 1-stats.t.cdf((x.mat - mean.mat)/np.sqrt(var.mat), df=K-1)
        ##df = totn -1
        #linkpv[idx].mat0 = 1-stats.t.cdf((x.mat0 - mean.mat0)/np.sqrt(var.mat0), df=K-1)
        #linkpv[idx].mat1 = 1-stats.t.cdf((x.mat1 - mean.mat1)/np.sqrt(var.mat1), df=K-1)
        ##0804 change totn to linkidx0[idx].mat0 linkidx0[idx].mat1
        with np.errstate(invalid='ignore', divide='ignore'):  ##this is only work for x1x2indict exp=1,2,8,9
            linkpv[idx].mat0 =np.where(var.mat0==0, 1, 1-stats.t.cdf((x.mat0 - mean.mat0)/np.sqrt(var.mat0), df=totn))  #not totn
            
            if exp >=3 and exp <=7:
                linkpv[idx].mat1 =np.where(var.mat1==0, 1, 1-stats.t.cdf((x.mat1 - mean.mat1)/np.sqrt(var.mat1), df=totn)) #not totn
                n0 = linkpv[idx].mat1.shape[2]

                for z in range(n0):
                    x1x2mat1=  np.matmul(indicator[name1][z, x.st:x.ed].reshape(-1,1),  indicator[name2][z, :].reshape(1,-1))
                    ##if indicator 0, then p-value = 1, else keep the same
                    linkpv[idx].mat1[:,:,z] = np.where(x1x2mat1==0, 1, linkpv[idx].mat1[:,:,z])

            else: #exp=1,2,8,9
            
                linkpv[idx].mat1 =np.where(var.mat1==0, 1.0, 1-stats.t.cdf((x.mat1 - mean.mat1)/np.sqrt(var.mat1), df=linkidx0[idx].mat1)) #not totn

      
        ###for indicator, exp= 3,4,5, 6,7
        #x1x2indict  = npmatmul(indicator[x1][z, x.st:x.ed].reshape(-1,1),  indicator[x2][z, :].reshape(1,-1))
       
        
               
        ##output all the pv
        with open(out+'/allpv0/'+out+'.'+name1+"-"+name2+'.allpv0.txt', 'a+') as f:
            #np.savetxt(out+"/allpv0/"+out+'.'+name1+"-"+name2+'.'+str(x.st)+'-'+str(x.ed)+'.allpv0.txt', linkpv[idx].mat0, delimiter='\t')
            np.savetxt(f, np.c_[range(x.st, x.ed),linkpv[idx].mat0], delimiter='\t')
        f.close()
        #np.savetxt(out+'.'+name1+"-"+name2+'.'+str(x.st)+'-'+str(x.ed)+'.allpv1.txt', linkpv[idx].mat1, delimiter='\t')
        #f.close()
        
        ###output: significant links; p < 0.005
        mat_pidx = np.where(linkpv[idx].mat0 < 0.005)
        #with open(out+'/signif/'+out+'.'+name1+"-"+name2+'.'+str(x.st)+'-'+str(x.ed)+'.significance.txt', 'w') as f:
        with open(out+'/signif/'+out+'.'+name1+"-"+name2+'.significance.txt', 'a+') as f:
            x1idx=mat_pidx[0]
            x2idx=mat_pidx[1]
            
            for ddd in list(range(x1idx.shape[0])):
                idx1 = x1idx[ddd]
                idx2 = x2idx[ddd]
                f.write("%s\t%s\t%s\t%s\t%f\n"%(name1,name2, x1meta[idx1], x2meta[idx2], linkpv[idx].mat0[idx1,idx2]))
        f.close()


        for idx30 in range(linkpv[idx].mat1.shape[2]):
            with open(out+'/allpv1/'+out+'.'+name1+"-"+name2+'.'+ str(idx30)+'.allpv1.txt','a+') as f:
            #np.savetxt(out+'/allpv1/'+out+'.'+name1+"-"+name2+'.'+ str(idx30)+'.'+str(x.st)+'-'+str(x.ed)+'.allpv1.txt', linkpv[idx].mat1[:,:,idx30], delimiter='\t')
                np.savetxt(f, np.c_[range(x.st,x.ed), linkpv[idx].mat1[:,:,idx30]], delimiter='\t')
                
            f.close()

            with open(out+'/allraw/'+out+'.'+name1+"-"+name2+'.'+ str(idx30)+'.allraw.txt','a+') as f:
            #np.savetxt(out+'/allpv1/'+out+'.'+name1+"-"+name2+'.'+ str(idx30)+'.'+str(x.st)+'-'+str(x.ed)+'.allpv1.txt', linkpv[idx].mat1[:,:,idx30], delimiter='\t')
                np.savetxt(f, np.c_[range(x.st,x.ed), link4test[idx].mat1[:,:,idx30]], delimiter='\t')
                
            f.close()
            
            
        mat_pidx = np.where(linkpv[idx].mat1 < 0.005)
        with open(out+'/signif/'+out+'.'+name1+"-"+name2+'.significance_docNorm.txt', 'a+') as f:
        #with open(out+'/signif/'+out+'.'+name1+"-"+name2+'.'+str(x.st)+'-'+str(x.ed)+'.significance_docNorm.txt', 'w') as f:
            x1idx=mat_pidx[0]
            x2idx=mat_pidx[1]
            x3idx=mat_pidx[2] ##ndoc, sample, patient
            
            for ddd in list(range(x1idx.shape[0])):
                idx1 = x1idx[ddd]
                idx2 = x2idx[ddd]
                idx3 = x3idx[ddd]
                f.write("%s\t%s\t%s\t%s\t%d\t%f\n"%(name1,name2, x1meta[idx1], x2meta[idx2], idx3, linkpv[idx].mat1[idx1,idx2,idx3]))
        f.close()
        
        
    ###write the files
    saveobj=(link4test, linkbg, linkbgvar, linkpv)
    with open(out+"/" + out+".dump.pikle", 'wb') as f:
        pickle.dump( saveobj, f)
    f.close()

if __name__=='__main__':

    '''
    need to compare: orig vs logit; logit vs dcor; logit_pcor,
    -c orig -g 0 -l 1 -p 0
    -c orig -g 1 -l 1 -p 0
    -c dcor -g 0 -l 1 -p 0
    -c dcor -g 0 -l 1 -p 1
    -c pcor -g 1 -l 1 -p 0
    -c pcor -g 1 -l 1 -p 1

    
    '''
    import sys
    from optparse import OptionParser

    usage = "usage: %prog [options] arg1 arg2"
    parser = OptionParser(usage=usage)
    parser.add_option("-i", "--input", dest="input", help="input file, from the output folder of mlda [default]")
    parser.add_option("-1", "--meta", dest="prefix1",help="Prefix for meta file")
    parser.add_option("-2", "--shuffledir",dest="shuffledir", help="dir for the shuffle data")
    parser.add_option("-K", "--Ktopics",dest="k",default=10, help="the number of topics")

    parser.add_option("-n", "--folds",dest="nfold", default=50, help="the number of fold to reduce memory usage")
    parser.add_option("-d", "--foldid", dest="foldid", help="the fold idx")
    parser.add_option("-o", "--out", dest="out", help="output file")   
    parser.add_option("-s", "--shufflecnt", dest="shufflecnt", default="2", help="the number of shuffle")
    parser.add_option("-c", "--correlation", dest="correlation", default="orig", help="correlation/similarity calculation")
    parser.add_option("-l", "--onlineshuffle", dest="onlineshuffle", default=1, help="online shuffling")
    parser.add_option("-p", "--pval", dest="pvalflag", default=0, help="calculate pvalue directly, no need shuffling results")
    parser.add_option("-g", "--logit", dest="logit", default=1, help="convert topic distribution using logistic norm transformation")
    parser.add_option("-e", "--exp", dest="exp", default=1, help="topic distribution is considered by both the doc2topic and word2topic distribution in getlinkcomb function")

    (options,args)=parser.parse_args()
    #parser.add_option("-K", "--Ktopics",dest="k", default="10", help="the number of topics")
    
    #if len(sys.argv) < 8:
    #    print('Usage: '+ sys.argv[0]+' path(xxx/xxx-iter10) prefix_meta(xxx) prefix_shuffle(xxx/shuffle/s1) Ktopics nfold_split fold_id outputfile\n The meta file is the same as the prefix_testing: asthma.iter3.0701-gene.meta.txt\n fold id is useless here ')
    #    sys.exit()
    '''
    path=sys.argv[1] ###output of mlda
    prefix1=sys.argv[2] ###  meta prefix
    prefix2=sys.argv[3]  ###prefix for shuffle
    K = int(sys.argv[4])   ###the number of topics
    nfold=int(sys.argv[5])
    foldid=int(sys.argv[6])
    out= sys.argv[7]
    tot_shuffle = 2;
    '''
    print(' '.join([str(x) for x in sys.argv]))
    
    if options.input is None or options.prefix1 is None or options.k is None or options.nfold is None or options.foldid is None or options.out is None:
        parser.print_help()
        sys.exit(1)

    exp0 = 0 if options.exp is None else int(options.exp)
    path = options.input
    prefix1 = options.prefix1
    prefix2 = options.shuffledir
    K = int(options.k)
    nfold = int(options.nfold)
    foldid = int(options.foldid)
    out = options.out
    tot_shuffle = int(options.shufflecnt)

    cor =options.correlation
    onshuffle= bool(int(options.onlineshuffle))
    pflag= bool(int(options.pvalflag))
    logitflag=  bool(int(options.logit))
    
    
    if not os.path.exists(out):
        os.makedirs(out)
    #if len(sys.argv) > 8:

    #tot_shuffle = int(sys.argv[8])
        
    #for foldid in range(nfold):
    print("%d calculation start:\n"% foldid)
    if foldid <= 0:
        foldid = 1
        
    main(path, prefix1, prefix2, K, nfold, foldid, out, tot_shuffle, cor=cor, logit=logitflag, onlineShuffle=onshuffle, pvflag=pflag,exp=exp0)