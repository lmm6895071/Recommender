'''
Created on Nay 29, 2018

@author: ming
'''

import os
import time
import logging
from util import eval_RMSE_bais_list,adam
import math
import numpy as np

def JMF_SU(train_user, train_item, valid_user, test_user,
           R, max_iter=50, lambda_u=1, lambda_v=100, dimension=50, lambda_p=0, lambda_q=50,momentum_flag=1):
    # explicit setting
    a = 1
    b = 0
    eta=-0.0005
    beta=0.02
    epsilon = 1e-20
    num_user = R.shape[0]
    num_item = R.shape[1]
    print "=================================== Models==================================="
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
    print "user_bais:",user_bais[0:10]
    print "######################################"
    
    '''
    user's prefrence matrix
    '''
    S_Train_R_I =[]# train_user[1]
    S_Train_R_J = []#train_item[1]
    S_Test_R = []#test_user[1]
    S_Valid_R = []#valid_user[1]

    uindx=0
    for item in train_user[1]:
        new_item=item.copy()
        for i in range(len(item)):
            if item[i] >user_bais[uindx]:
                new_item[i]=1.0/(1.0+(item[i]-user_bais[uindx])**(-2))#1.0/(1.0+math.exp(-item[i]+user_bais[iidex]))
            else:
                new_item[i]=0
        S_Train_R_I.append(new_item)
        uindx=uindx+1
    S_Train_R_I=np.array(S_Train_R_I)

    uindx=0
    for item in train_item[1]:
        new_item=item.copy()
        for i in range(len(item)):
            temp_bais=train_item[0][uindx][i]
            if item[i] >user_bais[temp_bais]:
                new_item[i]= 1.0/(1.0+(item[i]-user_bais[temp_bais])**(-2))#1.0/(1.0+math.exp(-item[i]+user_bais[temp_bais]))
            else:
                new_item[i]=0
        S_Train_R_J.append(new_item)
        uindx=uindx+1
    S_Train_R_J=np.array(S_Train_R_J)

    pre_val_eval = 1e10
    V = np.random.uniform(0.1,1,size=(num_item,dimension))
    U = np.random.uniform(0.1,1,size=(num_user, dimension))
    Q = np.random.uniform(0.1,1,size=(num_item,dimension))
    P = U#np.random.uniform(0,0.5,size=(num_user,dimension))

    '''
    sqrs_User=np.zeros([num_user,dimension])
    momentum_V_User=np.zeros([num_user,dimension])

    sqrs_Item=np.zeros([num_item,dimension])
    momentum_V_Item=np.zeros([num_item,dimension])

    sqrs_Q = np.zeros([num_item,dimension])
    momentum_V_Q=np.zeros([num_item,dimension])

    momentum_eta=-0.005
    iteration_flag=lambda_p
    '''
    sgd_eta=-0.0005

    endure_count = 100
    count = 0
    better_rmse = 100
    better_mae = 100.0
    for iteration in xrange(max_iter):
        loss = 0
        tic = time.time()
        print "%d iteration\t(patience: %d)" % (iteration, count)
        # VV = b * (V.T.dot(V)) + lambda_u * np.eye(dimension)#diagonal matrix
        sub_loss = np.zeros(num_user)
        print "=================================================================="
        print "the shape of U, U[i] {} {}".format(U.shape,U[0].shape)
        print "the shape of V, V[i] {} {}".format(V.shape,V[0].shape)
        print "=================================================================="
        # if momentum_flag==1 and  iteration <iteration_flag:
        #     print "momentum update",momentum_eta
        #     if iteration >=9:
        #         momentum_eta=-0.0001
        # elif momentum_flag !=0:
        #     print "sgd update", sgd_eta
        # else:
        #     print "grant=0"
        for i in xrange(num_user):
            
            idx_item = train_user[0][i]
            #train_user[0]=[[item1,item2,item3...],[item1,itme3],[item3,item2]...]
            #train_user[1]=[[rating1,rating2,rating3...],[rating1,rating3],[rating2,rating5]...]
            V_i = V[idx_item]
            R_i = Train_R_I[i]#[rating1,rating2,rating3...]
            
            '''
            preference matrix
            '''
            Q_i = Q[idx_item]
            S_R_i = S_Train_R_I[i]
            S_approx_R_i=P[i].dot(Q_i.T)


            approx_R_i = U[i].dot(V_i.T)
            upp=(Q_i * (np.tile(-S_R_i+S_approx_R_i, (dimension, 1)).T)).sum(0) 
            g=((V_i * (np.tile(-R_i+approx_R_i, (dimension, 1)).T)).sum(0) )+lambda_u*U[i]+ upp
            A= (a - b) * (V_i.T.dot(V_i))+lambda_u * np.eye(dimension)+Q_i.T.dot(Q_i)
            B = (a * V_i * (np.tile(R_i, (dimension, 1)).T)).sum(0) 
            B =B+ (Q_i*(np.tile(S_R_i,(dimension,1)).T)).sum(0)


            '''
            # sgd method
            # P[i]=P[i]+eta*(((Q_i * (np.tile(-S_R_i+S_approx_R_i, (dimension, 1)).T)).sum(0) )+lambda_p*P[i])
            if momentum_flag ==1:
                if iteration<iteration_flag:
                    try:
                        momentum_V_User[i],sqrs_User[i],div=adam(g,momentum_V_User[i],sqrs_User[i],-momentum_eta,iteration)
                        U[i]=U[i]-div
                    except Exception as err:
                        print err
                        return 
                else:
                    U[i]=U[i]+sgd_eta*g
            else:
            '''
            # U[i]=U[i]+eta*g

            U[i] =(np.linalg.solve(A.T, B.T)).T      #AX=B,X=A^(-1)B
            sub_loss[i] =sub_loss[i] -0.5 * lambda_u * np.dot(U[i], U[i])




            '''update S
            '''

            Xg = (S_R_i - S_approx_R_i) 
            S_R_i   =  S_R_i + eta*Xg

            X = (1.0/S_R_i -1)**2
            for i in range(len(X)):
                if X[i]<=0:
                    S_R_i[i] = 0




        P=U

        loss = loss + np.sum(sub_loss)
        print "=================================================================="
        print "the shape of V, V[i] {} {}".format(V.shape,V[0].shape)
        print "=================================================================="
        sub_loss = np.zeros(num_item)
        # UU = b * (U.T.dot(U))
        # SUU = b *(P.T.dot(P))
       
        for j in xrange(num_item):
            idx_user = train_item[0][j]
            U_j = U[idx_user]
            R_j = Train_R_J[j]

            A = (a - b) * (U_j.T.dot(U_j))
            B = (a * U_j * (np.tile(R_j, (dimension, 1)).T)).sum(0)
            approx_R_j = U_j.dot(V[j].T)
            g=(U_j * (np.tile(-R_j+approx_R_j, (dimension, 1)).T)).sum(0)+lambda_v*V[j]

            '''
            # V[j]=V[j]-div
            if momentum_flag==1:
                if iteration <iteration_flag:
                    try:
                        momentum_V_Item[j],sqrs_Item[j],div=adam(g,momentum_V_Item[j],sqrs_Item[j],-momentum_eta,iteration)
                        V[j]=V[j]-div
                    except Exception as err:
                        print err
                        return
                else:
                    V[j]=V[j]+ sgd_eta*g
            else:
            '''
            # V[j]=V[j]+eta*g

            V[j] = (np.linalg.solve((A+lambda_v * np.eye(dimension)).T, B.T)).T #A*X=B  X =A^-1*B

            sub_loss[j] = -0.5 * lambda_v * np.dot(V[j], V[j])
            sub_loss[j] = sub_loss[j]-0.5 * np.square(R_j * a).sum()
            sub_loss[j] = sub_loss[j] + a * np.sum((U_j.dot(V[j])) * R_j)
            sub_loss[j] = sub_loss[j] - 0.5 * np.dot(V[j].dot(A), V[j])

            '''
            preference Matrix
            '''
            S_R_j = S_Train_R_J[j]
            P_j = P[idx_user]

            SA = (a - b) * (P_j.T.dot(P_j))
            SB = (a * P_j * (np.tile(S_R_j, (dimension, 1)).T)).sum(0)
            S_approx_R_j = P_j.dot(Q[j].T)
            gq=(P_j * (np.tile(-S_R_j+S_approx_R_j, (dimension, 1)).T)).sum(0)+lambda_q*Q[j]

            '''
            if momentum_flag==1:
                if iteration <=iteration_flag:
                    try:
                        momentum_V_Q[j],sqrs_Q[j],div=adam(gq,momentum_V_Q[j],sqrs_Q[j],-momentum_eta,iteration)
                        Q[j]=Q[j]-div
                    except Exception as err:
                        print err
                        return 
                else:
                    Q[j]=Q[j]+sgd_eta*(gq)
            else:
            '''
            # Q[j]=Q[j]+eta*(gq)
            Q[j] = (np.linalg.solve((SA+lambda_q * np.eye(dimension)).T, SB.T)).T #A*X=B  X =A^-1*B

            sub_loss[j] =sub_loss[j] -0.5 * lambda_q * np.dot(Q[j], Q[j])
            sub_loss[j] = sub_loss[j]-0.5 * np.square(S_R_j * a).sum()
            sub_loss[j] = sub_loss[j] + a * np.sum((P_j.dot(Q[j])) * S_R_j)
            sub_loss[j] = sub_loss[j] - 0.5 * np.dot(Q[j].dot(SA), Q[j])


        loss = (loss + np.sum(sub_loss))
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
        if te_mae < better_mae:
            better_mae = te_mae
        print "\n JMF_S========better_rmse:{}   better_mae:{}==========\n".format(better_rmse,better_mae)

        if (count == endure_count):
            break
        PREV_LOSS = loss
