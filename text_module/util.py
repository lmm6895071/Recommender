'''
Created on Sep 21, 2017

@author: ming
'''
import numpy as np
import math
import heapq # for retrieval topK
import multiprocessing
from time import time

def eval_RMSE_bais(R,U,V,TS,k=50,heldout=4):
    num_user = U.shape[0]
    sub_rmse = np.zeros(num_user)
    sub_recall=np.zeros(num_user)
    TS_count = 0
    for i in xrange(num_user):
        idx_item = TS[i]
        if len(idx_item) == 0:
            continue
        TS_count = TS_count + len(idx_item)
        approx_R_i = U[i].dot(V[idx_item].T)  # approx_R[i, idx_item]
        result=approx_R_i.tolist()
        R_i = R[i]
        sub_rmse[i] = np.square(result - R_i).sum()
        sub_recall[i]=recall_top_k(num_user,R_i,result,k,heldout)
    rmse = np.sqrt(sub_rmse.sum() / TS_count)
    m_recall=sub_recall.sum()/num_user
    return rmse,m_recall


def eval_RMSE_bais_list(R,U,V,TS,k=[50],user_bias=[]):
    num_user = U.shape[0]
    sub_rmse = np.zeros(num_user)
    sub_recall=np.zeros(num_user)
    TS_count = 0
    sub_mae = np.zeros(num_user)

    recall_result=np.zeros([len(k),num_user])

    for i in xrange(num_user):
        idx_item = TS[i]
        if len(idx_item) == 0:
            continue
        TS_count = TS_count + len(idx_item)
        approx_R_i = U[i].dot(V[idx_item].T)  # approx_R[i, idx_item]
        result=approx_R_i.tolist()
        R_i = R[i]
        sub_rmse[i] = np.square(result - R_i).sum()
        sub_mae[i] = np.abs(result- R_i).sum()
        for it in range(len(k)):
            # recall_result[it][i]=sub_recall[i]
            # sub_recall[i]
            if len(user_bias)==num_user:
                heldout=user_bias[i]
            recall_result[it][i]=sub_recall[i]=recall_top_k(num_user,R_i,result,k[it],heldout)

    rmse = np.sqrt(sub_rmse.sum() / TS_count)
    mae  = sub_mae.sum()/TS_count
    m_recall=[]#sub_recall.sum()/num_user
    for item in range(len(k)):
        m_recall.append(recall_result[item].sum()/num_user)
    return rmse,m_recall,mae

def eval_RMSE(R, U, V, TS):
    num_user = U.shape[0]
    sub_rmse = np.zeros(num_user)
    TS_count = 0
    for i in xrange(num_user):
        idx_item = TS[i]
        if len(idx_item) == 0:
            continue
        TS_count = TS_count + len(idx_item)
        approx_R_i = U[i].dot(V[idx_item].T)  # approx_R[i, idx_item]
        R_i = R[i]

        sub_rmse[i] = np.square(approx_R_i - R_i).sum()

    rmse = np.sqrt(sub_rmse.sum() / TS_count)

    return rmse


def make_CDL_format(X_base, path):
    max_X = X_base.max(1).toarray()
    for i in xrange(max_X.shape[0]):
        if max_X[i, 0] == 0:
            max_X[i, 0] = 1
    max_X_rep = np.tile(max_X, (1, X_base.shape[1]))
    X_nor = X_base / max_X_rep
    np.savetxt(path + '/mult_nor.dat', X_nor, fmt='%.5f')


'''
eval hit,NDCG
'''
# Global variables that are shared across processes
def recall_top(num_user,u,R_tag,R_pre,k=[],user_bias=[]):
    # print k
    sub_recall=np.zeros(num_user)
    sub_rmse = np.zeros(num_user)
    sub_mae = np.zeros(num_user)

    TS_count = len(u)
    recall_result=np.zeros([len(k),num_user])

    R_P={}
    R_T={}#{1:[],2:[],3:[],4:[]}

    for i in range(len(u)):
        if u[i] not in R_P:
            temp_p=[]
            temp_T=[]
            temp_p.append(R_pre[i])
            temp_T.append(R_tag[i])
            R_P[u[i]]=temp_p
            R_T[u[i]]=temp_T
        else:
            R_P[u[i]].append(R_pre[i])
            R_T[u[i]].append(R_tag[i])

    numbers= len(R_P.keys())
    print "####numbers=",num_user,numbers,len(R_T.keys()),"####"

    for i in range(len(R_T.keys())):
         
        sub_rmse[i] = np.square(np.array(R_T[u[i]]) - np.array(R_P[u[i]])).sum()
        sub_mae[i]  = np.abs(np.array(R_T[u[i]])-np.array(R_P[u[i]])).sum()
        for it in range(len(k)):
            # recall_result[it][i]=sub_recall[i]
            # sub_recall[i]
            recall_result[it][i]=sub_recall[i]=recall_top_k(num_user,np.array(R_T[u[i]]),np.array(R_P[u[i]]),k[it],user_bias[i])

    rmse = np.sqrt(sub_rmse.sum() / TS_count)
    mae = sub_mae.sum()/TS_count

    m_recall=[]#sub_recall.sum()/num_user
    for item in range(len(k)):
        m_recall.append(recall_result[item].sum()/num_user)
    return rmse,m_recall,mae

def recall_top_k(num_user,R_tag,R_pre,k=50,heldout=4):

    sub_recall = np.zeros(1)
    for i in range(0,1):
        pre={}
        tag={}
        t=R_tag

        p=R_pre
        # print t
        # print p

        for  j in range(len(t)):
            if t[j]>=heldout:
                tag[j]=t[j]
            if p[j]>=heldout:
                pre[j]=p[j]
        s_pre=sorted(pre.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
        s_tag=sorted(tag.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)

        s_pre=s_pre[:k]#top_k

        ss_pre_key=[x[0] for x in s_pre]
        ss_tag_key=[x[0] for x in s_tag]
        ret_list = list((set(ss_pre_key).union(set(ss_tag_key)))^(set(ss_tag_key)^set(ss_pre_key)))
        # print ret_list
        try:
            # sub_recall[i]=(len(ret_list)+0.0)/min(k,len(ss_tag_key))
            sub_recall[i]=(len(ret_list)+0.0)/len(ss_tag_key)

        except  Exception as err:
            sub_recall[i]=0
            # print "error {}".format(err)

    result= sub_recall.sum()
    return result

# Adam
def adam(g, v, sqr, lr, t):
    beta1 = 0.9
    beta2 = 0.999
    eps_stable = 1e-8

    v[:] = beta1 * v + (1. - beta1) * g  #v=beta1*V+(1-beta1)*g

    sqr[:] = beta2 * sqr + (1. - beta2) * np.square(g)

    v_bias_corr = v / (1. - beta1 ** (t+1))
    sqr_bias_corr = sqr / (1. - beta2 ** (t+1))
    div = lr * v_bias_corr / (np.sqrt(sqr_bias_corr) + eps_stable)
    return (v,sqr,div)

if __name__ == '__main__':
    r=[3,2,4,5,3]
    p=[3,3,3,4,2]
    r=np.array(r)
    p=np.array(p)
    # print recall_top_k(1,r,p,10

    u=[1,2,2,3,2]

    a,b= recall_top(3,u,r,p,[1,2,3],[3,3,3])
    print a,b
