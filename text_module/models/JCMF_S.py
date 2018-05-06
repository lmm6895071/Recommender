'''
Created on April 9, 2018
@author: ming
'''

import os
import time
from util import eval_RMSE_bais_list
import math
import numpy as np
from text_analysis.keras_cnn import CNN_module


def JCMF_S(res_dir, train_user, train_item, valid_user, test_user,
           R, CNN_X, vocab_size, init_W=None, give_item_weight=True,
           max_iter=50, lambda_u=1, lambda_v=100, dimension=50,
           dropout_rate=0.2, emb_dim=200, max_len=300, num_kernel_per_ws=100,lambda_p=0,lambda_q=1):
    # explicit setting
    a = 1
    b = 0
    eta=-0.0005
    alpha=0.8

    num_user = R.shape[0]
    num_item = R.shape[1]
    # CNN_X=np.zeros([num_item,max_len])

    PREV_LOSS = 1e-50
    '''
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    f1 = open(res_dir + '/state.log', 'w')
    '''
    Train_R_I = train_user[1]
    Train_R_J = train_item[1]
    Test_R = test_user[1]
    Valid_R = valid_user[1]

    if give_item_weight is True:
        item_weight = np.array([math.sqrt(len(i))
                                for i in Train_R_J], dtype=float)
        item_weight = (float(num_item) / item_weight.sum()) * item_weight
    else:
        item_weight = np.ones(num_item, dtype=float)

    pre_val_eval = 1e12



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
    print "global average: ",global_average#3.58254603506
    print "user_bais:",  user_bais[0:50]
    print "item_bais:",   item_bais[0:50]
    print "######################################"
    # return
    '''
    preference matrix
    '''
    S_Train_R_I =[]# train_user[1]
    S_Train_R_J = []#train_item[1]
    S_Test_R = []#test_user[1]
    S_Valid_R = []#valid_user[1]

     
    '''
    user preference matrix
    '''
    iidex=0
    for item in train_user[1]:
        new_item=item.copy()
        for i in range(len(item)):
            if item[i] >user_bais[iidex]:
                new_item[i]=1.0/(1.0+(item[i]-user_bais[iidex])**(-2))   #1.0/(1.0+math.exp(-item[i]+user_bais[iidex]))
            else:
                new_item[i]=0
        S_Train_R_I.append(new_item)
        iidex=iidex+1
    S_Train_R_I=np.array(S_Train_R_I)

    iidex=0
    for item in train_item[1]:
        new_item=item.copy()
        for i in range(len(item)):
            temp_bais=train_item[0][iidex][i]
            if item[i] > user_bais[temp_bais]:
                new_item[i]= 1.0/(1.0+(item[i]-user_bais[temp_bais])**(-2))#1.0/(1.0+math.exp(-item[i]+user_bais[temp_bais]))
            else:
                new_item[i]=0
        S_Train_R_J.append(new_item)
        iidex=iidex+1
    S_Train_R_J=np.array(S_Train_R_J)

    Q = np.random.uniform(0,0.5,size=(num_item,dimension))
    P = np.random.uniform(0,0.5,size=(num_user,dimension))

    cnn_module = CNN_module(dimension, vocab_size, dropout_rate,emb_dim, max_len, num_kernel_per_ws, init_W)
    theta = cnn_module.get_projection_layer(CNN_X)
    np.random.seed(133)
    U = np.random.uniform(size=(num_user, dimension))
    V = theta

    endure_count = 5
    count = 0
    better_rmse=100.0
    better_mae =100.0
    for iteration in xrange(max_iter):
        loss = 0
        tic = time.time()
        print "%d iteration\t(patience: %d)" % (iteration, count)

        VV = b * (V.T.dot(V)) + lambda_u * np.eye(dimension)
        SVV = b * (Q.T.dot(Q)) + lambda_q * np.eye(dimension)

        sub_loss = np.zeros(num_user)

        for i in xrange(num_user):
            idx_item = train_user[0][i]
            V_i = V[idx_item]
            R_i = Train_R_I[i]
            A = VV + (a - b) * (V_i.T.dot(V_i))
            B = (a * V_i * (np.tile(R_i, (dimension, 1)).T)).sum(0)

            U[i] = np.linalg.solve(A, B)

            sub_loss[i] = -0.5 * lambda_u * np.dot(U[i], U[i])

            '''
            Q_i = Q[idx_item]
            S_R_i = S_Train_R_I[i]
            S_approx_R_i=P[i].dot(Q_i.T)
            approx_R_i = U[i].dot(V_i.T)
            # U[i]=U[i]+eta*(((V_i * (np.tile(-R_i+approx_R_i, (dimension, 1)).T)).sum(0) )+lambda_u*U[i])
            SA = SVV + (a - b) * (Q_i.T.dot(Q_i))
            SB = (a * Q_i * (np.tile(S_R_i, (dimension, 1)).T)).sum(0)
            P[i] = np.linalg.solve(SA, SB)
            # P[i]=P[i]+eta*(((Q_i * (np.tile(-S_R_i+S_approx_R_i, (dimension, 1)).T)).sum(0) )+lambda_p*P[i])      
            # sub_loss[i] =sub_loss[i] -0.5 * lambda_u * np.dot(U[i], U[i])
            sub_loss[i] =sub_loss[i]-0.5 * lambda_p * np.dot(P[i], P[i])
            '''
        P=U

        loss = loss + np.sum(sub_loss)
        sub_loss = np.zeros(num_item)
        UU = b * (U.T.dot(U))
        SUU = b *(P.T.dot(P))

        for j in xrange(num_item):
            idx_user = train_item[0][j]
            U_j = U[idx_user]
            R_j = Train_R_J[j]

            tmp_A = UU + (a - b) * (U_j.T.dot(U_j))
            A = tmp_A + lambda_v * item_weight[j] * np.eye(dimension)
            B = (a * U_j * (np.tile(R_j, (dimension, 1)).T)
                 ).sum(0) + lambda_v * item_weight[j] * theta[j]
            V[j] = np.linalg.solve(A, B)


            sub_loss[j] = -0.5 * lambda_v * np.dot(V[j], V[j])
            temp_loss=-0.5 * np.square(R_j * a).sum()
            temp_loss=temp_loss+a * np.sum((U_j.dot(V[j])) * R_j)
            temp_loss=temp_loss - 0.5 * np.dot(V[j].dot(A), V[j])

            sub_loss[j]=sub_loss[j]+alpha*temp_loss

            '''
            perference Matrix
            '''
            S_R_j = S_Train_R_J[j]
            P_j = P[idx_user]

            SA = SUU + (a - b) * (P_j.T.dot(P_j))
            SB = (a * P_j * (np.tile(S_R_j, (dimension, 1)).T)).sum(0)

            S_approx_R_j = P_j.dot(Q[j].T)


            tmp_A = SUU + (a - b) * (P_j.T.dot(P_j))
            # A = tmp_A + lambda_q * item_weight[j] * np.eye(dimension)
            # B = (a * P_j * (np.tile(S_R_j, (dimension, 1)).T)).sum(0)  
            Q[j] = np.linalg.solve(SA, SB)

            #SGD 
            # Q[j]=Q[j]+eta*((P_j * (np.tile(-S_R_j+S_approx_R_j, (dimension, 1)).T)).sum(0)+lambda_q*Q[j])
            
            sub_loss[j] =sub_loss[j] -0.5 * lambda_q * np.dot(Q[j], Q[j])
            temp_loss=-0.5 * np.square(S_R_j * a).sum()
            temp_loss=temp_loss+a * np.sum((P_j.dot(Q[j])) * S_R_j)
            temp_loss=temp_loss - 0.5 * np.dot(Q[j].dot(SA), Q[j])
            sub_loss[j] = sub_loss[j] + temp_loss#(1-alpha)*

        loss = loss + np.sum(sub_loss)
        seed = np.random.randint(100000)
        history = cnn_module.train(CNN_X, V, item_weight, seed)
        theta = cnn_module.get_projection_layer(CNN_X)
        cnn_loss = history.history['loss'][-1]

        loss = loss - 0.5 * lambda_v * cnn_loss * num_item

        topk=[3,5,10,15,20,25,30,40,50,100]
        tr_eval,tr_recall,tr_mae=eval_RMSE_bais_list(Train_R_I, U, V, train_user[0],topk,user_bais)
        val_eval,va_recall,va_mae = eval_RMSE_bais_list(Valid_R, U, V, valid_user[0],topk,user_bais)
        te_eval,te_recall,te_mae = eval_RMSE_bais_list(Test_R, U, V, test_user[0],topk,user_bais)

        for i in range(len(topk)):
            print "recall top-{}: Train:{} Validation:{}  Test:{}".format(topk[i],tr_recall[i],va_recall[i],te_recall[i])

        toc = time.time()
        elapsed = toc - tic

        converge = abs((loss - PREV_LOSS) / PREV_LOSS)

        if (val_eval < pre_val_eval):
            # cnn_module.save_model(res_dir + '/CNN_weights.hdf5')
            # np.savetxt(res_dir + '/U.dat', U)
            # np.savetxt(res_dir + '/V.dat', V)
            # np.savetxt(res_dir + '/theta.dat', theta)
            pass
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
        print "\ JCMF_S=============better_rmse:{}=====better_mae:{}==============\n".format(better_rmse,better_mae)
        if (count == endure_count):
            break
        PREV_LOSS = loss

        # f1.write("Loss: %.5f Elpased: %.4fs Converge: %.6f Tr: %.5f Val: %.5f Te: %.5f\n" % (
        #     loss, elapsed, converge, tr_eval, val_eval, te_eval))
    # f1.close()

