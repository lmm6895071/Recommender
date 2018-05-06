'''
Created on April 8, 2018

@author: ming
'''

import os,sys
import time
import logging
from util import eval_RMSE_bais_list
import math
import numpy as np

def JMF_Double(train_user, train_item, valid_user, test_user,
           R,max_iter=500, lambda_u=1, lambda_v=100,lambda_q=10,lambda_p=10):
    # explicit setting
    a = 1
    b = 0
    eta=-0.0001
    alpha=0.8
    lamda_m=1000
    lamda_n= 1000

    num_user = R.shape[0]
    num_item = R.shape[1]
    print "===================================ConvMF Models==================================="
    print "\tnum_user is:{}".format(num_user)
    print "\tnum_item is:{}".format(num_item)
    print "==================================================================================="
    PREV_LOSS = 1e-50

    Train_R_I = train_user[1] #this is rating; train_user_[0] is the item_index
    Train_R_J = train_item[1]
    Test_R = test_user[1]
    Valid_R = valid_user[1]

    # print train_user[1][0:5]

    '''
    compute the average of R
    '''

    train_sum=0
    test_sum=0
    valid_sum=0
    train_size=0
    test_size=0
    valid_size=0
    total_sum=0

    user_bais_sum=[]
    item_bais_sum=[]
    user_bais_size=[]
    item_bais_size=[]

    for item in train_user[1]:
        train_sum=train_sum+ np.sum(item)
        train_size=train_size+np.size(item)

        user_bais_sum.append(np.sum(item))
        user_bais_size.append(len(item))
    for item in test_user[1]:
        test_sum=test_sum+ np.sum(item)
        test_size=test_size+np.size(item)
    for item in valid_user[1]:
        valid_sum=valid_sum+ np.sum(item)
        valid_size=valid_size+np.size(item)
    for item in train_item[1]:
        item_bais_sum.append(np.sum(item))
        item_bais_size.append(len(item))
    total_size=train_size+test_size+valid_size
    total_sum=train_sum+test_sum+valid_sum
    global_average=total_sum*1.0/total_size
    user_bais=[user_bais_sum[i]/user_bais_size[i] for i in range(len(user_bais_sum))]
    item_bais=[item_bais_sum[i]/item_bais_size[i] for i in range(len(item_bais_sum))]
    print "######################################"
    print "sum: ",train_sum,test_sum,valid_sum
    print "size: ",train_size,test_size,valid_size
    print "average: ",train_sum*1.0/train_size, test_sum*1.0/test_size, valid_sum*1.0/valid_size
    print "global average: ",global_average
    print "user_bais:",  user_bais[0:50]
    print "item_bais:",   item_bais[0:50]
    print "######################################"
    '''
    preference matrix
    '''
    S_Train_R_I =[]# train_user[1]
    S_Train_R_J = []#train_item[1]
    S_Test_R = []#test_user[1]
    S_Valid_R = []#valid_user[1]

    iidex=0
    for item in train_user[1]:
        new_item=item.copy()
        for i in range(len(item)):
            if item[i] >= user_bais[iidex]: #global_average:
                new_item[i]=1#1.0/(1.0+math.exp(-item[i]+user_bais[iidex]))
            else:
                new_item[i]=0
        S_Train_R_I.append(new_item)
        iidex=iidex+1
    S_Train_R_I=np.array(S_Train_R_I)

    iidex=0
    for item in train_item[1]:
        new_item=item.copy()
        for i in range(len(item)):
            if item[i] >= item_bais[iidex]: #global_average:
                new_item[i]=1#1.0/(1.0+math.exp(-item[i]+item_bais[iidex]))
            else:
                new_item[i]=0
        S_Train_R_J.append(new_item)
        iidex=iidex+1
    S_Train_R_J=np.array(S_Train_R_J)

    pre_val_eval = 1e10
    V = np.random.uniform(0,0.5,size=(num_item,dimension))
    U = np.random.uniform(0,0.5,size=(num_user, dimension))
    Q = np.random.uniform(0,0.5,size=(num_item,dimension))
    P = np.random.uniform(0,0.5,size=(num_user,dimension))
    M = np.random.uniform(0,0.5,size=(dimension,dimension))
    N = np.random.uniform(0,0.5,size=(dimension,dimension))
    endure_count = 100
    count = 0
    XX=U+P.dot(M)
    YY=V+Q.dot(N)

    better_rmse=100
    for iteration in xrange(max_iter):
        loss = 0
        tic = time.time()
        print "%d iteration\t(patience: %d)" % (iteration, count)
        # VV = b * (V.T.dot(V)) + lambda_u * np.eye(dimension)#diagonal matrix
        sub_loss = np.zeros(num_user)
        print "=================================================================="
        print "the shape of U, U[i] {} {}".format(U.shape,U[0].shape)
        print "=================================================================="

        tm1=np.zeros([dimension,dimension])
        tn1=np.zeros([dimension,dimension])
        for i in xrange(num_user):
            idx_item = train_user[0][i]
            #train_user[0]=[[item1,item2,item3...],[item1,itme3],[item3,item2]...]
            #train_user[1]=[[rating1,rating2,rating3...],[rating1,rating3],[rating2,rating5]...]
            # V_i = V[idx_item]
            YY_i=YY[idx_item]
            R_i = Train_R_I[i]#[rating1,rating2,rating3...]

            '''
            perference matrix
            '''
            Q_i = Q[idx_item]
            S_R_i = S_Train_R_I[i]
            S_approx_R_i=P[i].dot(Q_i.T)

            approx_R_i =(XX[i]).dot(YY_i.T)
            t1=(YY_i * (np.tile(-R_i+approx_R_i, (dimension, 1)).T)).sum(0)

            U[i]=U[i]+eta*(t1+lambda_u*U[i])
            t2=((Q_i * (np.tile(-S_R_i+S_approx_R_i, (dimension, 1)).T)).sum(0) )+lambda_p*P[i]
            P[i]=P[i]+eta*(t1.dot(M)+t2)

            sub_loss[i] =sub_loss[i] -0.5 * lambda_u * np.dot(U[i], U[i])
            sub_loss[i] =sub_loss[i]-0.5 * lambda_p * np.dot(P[i], P[i])
            tm1=t1.T.dot(P[i])+tm1

        M=M+eta*(tm1+lamda_m * M)
        XX=U+P.dot(M)
        loss = loss + np.sum(sub_loss)
        print "=================================================================="
        print "the shape of V, V[i] {} {}".format(V.shape,V[0].shape)
        print "=================================================================="
        sub_loss = np.zeros(num_item)
        # UU = b * (U.T.dot(U))
        # SUU = b *(P.T.dot(P))
        for j in xrange(num_item):
            idx_user = train_item[0][j]
            XX_j = XX[idx_user]
            R_j = Train_R_J[j]

            A =  (a - b) * (XX_j.T.dot(XX_j))
            # B = (a * XX_j * (np.tile(R_j, (dimension, 1)).T)).sum(0)
            approx_R_j = XX_j.dot(YY[j].T)
            t1=(XX_j * (np.tile(-R_j+approx_R_j, (dimension, 1)).T)).sum(0)
            V[j]=V[j]+eta*(t1+lambda_v*V[j])


            sub_loss[j] = -0.5 * lambda_v * np.dot(V[j], V[j])

            temp_loss=-0.5 * np.square(R_j * a).sum()
            temp_loss=temp_loss+a * np.sum((XX_j.dot(YY[j])) * R_j)
            temp_loss=temp_loss - 0.5 * np.dot(YY[j].dot(A), YY[j])

            sub_loss[j]=sub_loss[j]+alpha*temp_loss

            '''
            Sentiment Matrix
            '''
            S_R_j = S_Train_R_J[j]
            P_j = P[idx_user]

            SA =   (a - b) * (P_j.T.dot(P_j))
            # SB = (a * P_j * (np.tile(S_R_j, (dimension, 1)).T)).sum(0)

            S_approx_R_j = P_j.dot(Q[j].T)
            t2=(P_j * (np.tile(-S_R_j+S_approx_R_j, (dimension, 1)).T)).sum(0)+lambda_q*Q[j]
            qqj=Q[j].copy()
            Q[j]=Q[j]+eta*(t1.dot(N)+t2)

            tn1=tn1+t1.T.dot(Q[j])
            sub_loss[j] =sub_loss[j] -0.5 * lambda_q * np.dot(Q[j], Q[j])

            temp_loss=-0.5 * np.square(S_R_j * a).sum()
            temp_loss=temp_loss+a * np.sum((P_j.dot(qqj)) * S_R_j)
            temp_loss=temp_loss - 0.5 * np.dot(qqj.dot(SA), qqj)
            sub_loss[j] = sub_loss[j] + (1-alpha)*temp_loss

        N=N+eta*(tn1+lamda_n*N)
        YY=V+Q.dot(N)
        loss = loss + np.sum(sub_loss)+lamda_m*np.sum(np.square(M))+lamda_n *np.sum(np.square(N))
        seed = np.random.randint(100000)

        topk=[3,5,10,15,20,25,30,40,50,100]
        tr_eval,tr_recall,tr_mae=eval_RMSE_bais_list(train_user[1], U, V, train_user[0],topk,user_bais)
        val_eval,va_recall,va_mae = eval_RMSE_bais_list(valid_user[1], U, V, valid_user[0],topk,user_bais)
        te_eval,te_recall,te_mae = eval_RMSE_bais_list(test_user[1], U, V, test_user[0],topk,user_bais)

        for i in range(len(topk)):
            print "recall top-{}: Train:{} Validation:{}  Test:{}".format(topk[i],tr_recall[i],va_recall[i],te_recall[i])

        toc = time.time()
        elapsed = toc - tic
        converge = abs((loss - PREV_LOSS) / PREV_LOSS)

        if (val_eval < pre_val_eval):
            print "Best Test result!!!!!"
        else:
            count = count + 1
        pre_val_eval = val_eval
        print "=====================RMSE============================="
        print "Loss: %.5f Elpased: %.4fs Converge: %.6f Train: %.5f Validation: %.5f Test: %.5f" % (
            loss, elapsed, converge, tr_eval, val_eval, te_eval)
        print "=====================MAE============================="
        print " Train: %.5f Validation: %.5f Test: %.5f" % ( tr_mae, va_mae, te_mae)
        if te_eval <better_rmse:
            better_rmse=te_eval
        print "==============better_rmse:{}===================".format(better_rmse)
        if (count == endure_count):
            break
        PREV_LOSS = loss
