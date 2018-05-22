'''
Created on Sep 8, 2017

@author: ming
'''

import os,sys
import time
import logging
from util import eval_RMSE_bais_list,adam
import math
import numpy as np

def WNMF(train_user, train_item, valid_user, test_user,R,max_iter=50, dimension=50):
    # explicit setting
    a = 1
    b = 0

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


    '''
    compute the average of R
    '''

    train_sum=0
    test_sum=0#= np.sum(test_user[1])
    valid_sum=0#np.sum(valid_user[1])
    train_size=0#np.size(train_user[1])
    test_size=0# np.size(test_user[1])
    valid_size=0#np.size(valid_user[1])
    total_sum=0#train_sum+test_sum+valid_sum
    user_bias_sum=[]
    item_bias_sum=[]
    user_bias_size=[]
    item_bias_size=[]
    '''
    obtain the average rating of each user
    '''
    for item in train_user[1]:
        train_sum=train_sum+ np.sum(item)
        train_size=train_size+np.size(item)

        user_bias_sum.append(np.sum(item))
        user_bias_size.append(len(item))

    for item in test_user[1]:
        test_sum=test_sum+ np.sum(item)
        test_size=test_size+np.size(item)
    for item in valid_user[1]:
        valid_sum=valid_sum+ np.sum(item)
        valid_size=valid_size+np.size(item)

    total_size=train_size+test_size+valid_size
    total_sum=train_sum+test_sum+valid_sum
    global_average=total_sum*1.0/total_size
    user_bias=[user_bias_sum[i]/user_bias_size[i] for i in range(len(user_bias_sum))]

    '''
    obtain the average rating of each item
    '''
    item_sum=0
    item_size=0
    for user in train_item[1]:
        item_sum=item_sum+ np.sum(user)
        item_size=item_size+np.size(user)

        item_bias_sum.append(np.sum(user))
        item_bias_size.append(len(user))

    item_bias=[item_bias_sum[i]/item_bias_size[i] for i in range(len(item_bias_sum))]



    print "######################################"
    print "sum: ",train_sum,test_sum,valid_sum
    print "size: ",train_size,test_size,valid_size
    print "average: ",train_sum*1.0/train_size, test_sum*1.0/test_size, valid_sum*1.0/valid_size
    print "global average: ",total_sum*1.0/total_size
    print "user_bias:",user_bias[0:10]
    print 'item_bias:',item_bias[0:10]
    print "######################################"



    pre_val_eval = 1e10
    V = np.random.uniform(0.001,0.5,size=(num_item,dimension))
    U = np.random.uniform(0.001,0.5,size=(num_user, dimension))

    print V[0,1:3]
    print U[0,1:3]
    endure_count = 100
    count = 0
    

    '''
    momentum method
    '''
    # sqrs_User=np.zeros([num_user,dimension])
    # momentum_V_User=np.zeros([num_user,dimension])
    # sqrs_Item=np.zeros([num_item,dimension])
    # momentum_V_Item=np.zeros([num_item,dimension])

    better_rmse = 100
    better_mae = 100
    for iteration in xrange(max_iter):
        loss = 0
        tic = time.time()
        print "%d iteration\t(patience: %d)" % (iteration, count)

        sub_loss = np.zeros(num_user)
        print "=================================================================="
        print "the shape of U, U[i] {} {}".format(U.shape,U[0].shape)
        print "=================================================================="
        for i in xrange(num_user):
            idx_item = train_user[0][i]
            #train_user[0]=[[item1,item2,item3...],[item1,itme3],[item3,item2]...]
            #train_user[1]=[[rating1,rating2,rating3...],[rating1,rating3],[rating2,rating5]...]

            V_i = V[idx_item]
            R_i = Train_R_I[i]#[rating1,rating2,rating3...]

            A = (a - b) * (V_i.T.dot(V_i))
            B = (a * V_i * (np.tile(R_i, (dimension, 1)).T)).sum(0) #np.tile() array copy; sum(0) is the sum of each column,sum(1) is the sum of each row;
            

            '''
            Multiply SGD method or momentum method
            '''
            approx_R_i = U[i].dot(V_i.T)
            g=((V_i * (np.tile(approx_R_i, (dimension, 1)).T)).sum(0) )            
            U[i]=U[i]*B/g


        loss = loss + np.sum(sub_loss)
        print "=================================================================="
        print "the shape of V, V[i] {} {}".format(V.shape,V[0].shape)
        print "=================================================================="
        sub_loss = np.zeros(num_item)
        for j in xrange(num_item):
            idx_user = train_item[0][j]
            U_j = U[idx_user]
            R_j = Train_R_J[j]

            A = (a - b) * (U_j.T.dot(U_j))
            B = (a * U_j * (np.tile(R_j, (dimension, 1)).T)).sum(0)
            
            '''
            multiply  SGD / momentum
            '''
            approx_R_j = U_j.dot(V[j].T)
            g=(U_j * (np.tile(approx_R_j, (dimension, 1)).T)).sum(0) 
            V[j]=V[j]*B/g

            sub_loss[j] = sub_loss[j]-0.5 * np.square(R_j * a).sum()
            sub_loss[j] = sub_loss[j] + a * np.sum((U_j.dot(V[j])) * R_j)
            sub_loss[j] = sub_loss[j] - 0.5 * np.dot(V[j].dot(A), V[j])

        loss =(loss + np.sum(sub_loss))
        seed = np.random.randint(100000)


        topk=[3,5,10,15,20,25,30,40,50,100]
        # holdout=4

        tr_eval,tr_recall,tr_mae=eval_RMSE_bais_list(Train_R_I, U, V, train_user[0],topk,user_bias)
        val_eval,va_recall,va_mae = eval_RMSE_bais_list(Valid_R, U, V, valid_user[0],topk,user_bias)
        te_eval,te_recall,te_mae = eval_RMSE_bais_list(Test_R, U, V, test_user[0],topk,user_bias)

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


        print "WNNMF =====================RMSE============================="
        print "Loss: %.5f Elpased: %.4fs Converge: %.6f Train: %.5f Validation: %.5f Test: %.5f" % (
            loss, elapsed, converge, tr_eval, val_eval, te_eval)
        print "WNMF=====================MAE============================="
        print " Train: %.5f Validation: %.5f Test: %.5f" % ( tr_mae, va_mae, te_mae)


        if te_eval <better_rmse:
            better_rmse=te_eval
        if te_mae < better_mae:
            better_mae = te_mae
        print "\n WNMF========better_rmse:{}   better_mae:{}==========\n".format(better_rmse,better_mae)

        if (count == endure_count):
            break
        

        PREV_LOSS = loss

