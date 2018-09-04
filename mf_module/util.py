# -*- coding: utf-8 -*-
from __future__ import unicode_literals
'''
Created on Sep 21, 2017

@author: ming
'''
import numpy as np
import math
import heapq # for retrieval topK
import multiprocessing
from time import time



def eval_RMSE_bias(R,U,V,TS,k=50,heldout=4):
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
def eval_RMSE_bias_list(R,U,V,TS,k=[50],user_bias=[]):
    num_user = U.shape[0]
    sub_rmse = np.zeros(num_user)
    sub_recall=np.zeros(num_user)
    sub_mae = np.zeros(num_user)
    TS_count = 0

    ndcg=np.zeros([3,num_user])

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
        sub_mae[i]  = np.abs(result-R_i).sum()
        
        ndcg[0][i]= ndcg_score(R_i,result,5)
        ndcg[1][i]=ndcg_score(R_i,result,10)
        ndcg[2][i]=ndcg_score(R_i,result,20)

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
    return rmse,m_recall,mae,ndcg.mean(1)


'''
considering the bias of user,item; U_bias and Item_bias is contant
'''
def eval_RMSE_bias_list_alter(R,U,V,TS,k=[50],user_bias=[],item_bias=[],average=0):
    num_user = U.shape[0]
    sub_rmse = np.zeros(num_user)
    sub_recall=np.zeros(num_user)
    sub_mae = np.zeros(num_user)

    TS_count = 0

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
        sub_mae[i]  = np.abs(result-R_i).sum()

        '''
        recover the origin value 
        r_i,j =R_i + average+ u_bias+v_bias
        ''' 

        for t in range(len(idx_item)):
            temp_bias=TS[i][t]
            R_i[t]= R_i[t]+user_bias[i] -2* average+item_bias[temp_bias]
            result[t]=result[t]+user_bias[i]+item_bias[temp_bias]-2*average

        for it in range(len(k)):
            # recall_result[it][i]=sub_recall[i]
            # sub_recall[i]
            if len(user_bias)==num_user:
                heldout=user_bias[i]
            recall_result[it][i]=sub_recall[i]=recall_top_k(num_user,R_i,result,k[it],heldout)


    rmse = np.sqrt(sub_rmse.sum() / TS_count)
    m_recall=[]#sub_recall.sum()/num_user
    mae  = sub_mae.sum()/TS_count

    for item in range(len(k)):
        m_recall.append(recall_result[item].sum()/num_user)
    return rmse,m_recall


'''
considering the bias of user,item; 
U_bias is alptha 1*N
V_bias is beta  1*M

'''
def eval_RMSE_bias_alpha_beta(R,U,V,TS,k=[50],alptha=[],beta=[],user_bias=[]):
    num_user = U.shape[0]
    sub_rmse = np.zeros(num_user)
    sub_recall=np.zeros(num_user)
    TS_count = 0
    sub_mae = np.zeros(num_user)
    ndcg =np.zeros([3,num_user])

    recall_result=np.zeros([len(k),num_user])

    for i in xrange(num_user):
        idx_item = TS[i]
        if len(idx_item) == 0:
            continue
        TS_count = TS_count + len(idx_item)
        approx_R_i = U[i].dot(V[idx_item].T)  # approx_R[i, idx_item]
        result=approx_R_i.tolist()
        R_i = R[i]
        '''
        recover the origin value 
        r_i,j =R_i + alpha_i + beta_j
        ''' 
        for t in range(len(idx_item)):
            temp_bias=TS[i][t]
            result[t]=result[t]+alptha[i]+beta[temp_bias]


        sub_rmse[i] = np.square(result - R_i).sum()
        sub_mae[i]  = np.abs(result-R_i).sum()
        ndcg[0][i]= ndcg_score(R_i,result,5)
        ndcg[1][i]=ndcg_score(R_i,result,10)
        ndcg[2][i]=ndcg_score(R_i,result,20)
        for it in range(len(k)):
            # recall_result[it][i]=sub_recall[i]
            # sub_recall[i]
            if len(user_bias)==num_user:
                heldout=user_bias[i]
            recall_result[it][i]=sub_recall[i]=recall_top_k(num_user,R_i,result,k[it],heldout)



    rmse = np.sqrt(sub_rmse.sum() / TS_count)
    mae = sub_mae.sum()/TS_count
    m_recall=[]#sub_recall.sum()/num_user
    for item in range(len(k)):
        m_recall.append(recall_result[item].sum()/num_user)

    return rmse,m_recall,mae,ndcg.mean(1)




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

'''
eval hit,NDCG
'''
# Global variables that are shared across processes
def recall_top(num_user,u,R_tag,R_pre,k=[],heldout=4):
    # print k
    sub_recall=np.zeros(num_user)
    sub_rmse = np.zeros(num_user)

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
         
        sub_rmse[i] = np.square(np.array(R_T[u[i]]) - np.array( R_P[u[i]])).sum()
        for it in range(len(k)):
            # recall_result[it][i]=sub_recall[i]
            # sub_recall[i]
            recall_result[it][i]=sub_recall[i]=recall_top_k(num_user,np.array(R_T[u[i]]),np.array(R_P[u[i]]),k[it],heldout)


    rmse = np.sqrt(sub_rmse.sum() / TS_count)
    m_recall=[]#sub_recall.sum()/num_user
    for item in range(len(k)):
        m_recall.append(recall_result[item].sum()/num_user)

    return rmse,m_recall

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
def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k

    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).

    y_score : array-like, shape = [n_samples]
        Predicted scores.

    k : int
        Rank.

    gains : str
        Whether gains should be "exponential" (default) or "linear".

    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k

    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).

    y_score : array-like, shape = [n_samples]
        Predicted scores.

    k : int
        Rank.

    gains : str
        Whether gains should be "exponential" (default) or "linear".

    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best



if __name__ == '__main__':
    r=[3,2,4,5,3]
    p=[3,3,3,4,2]
    r=np.array(r)
    p=np.array(p)
    # print recall_top_k(1,r,p,10

    u=[1,2,2,3,2]

    a,b= recall_top(3,u,r,p,[1,2,3],3)
    print a,b


    ndcg_score()
