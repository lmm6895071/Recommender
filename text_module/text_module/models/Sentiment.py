'''
Created on Sep 8, 2017

@author: ming
'''

import os
import time
import logging
from util import  eval_RMSE_bais_list
import math
import numpy as np
from text_analysis.cnn_model import CNN_module

def Sentiment(res_dir, train_user, train_item, valid_user, test_user,
           R, CNN_X, vocab_size, init_W=None, give_item_weight=True,
           max_iter=50, lambda_u=1, lambda_v=100, dimension=50,
           dropout_rate=0.2, emb_dim=200, max_len=300, num_kernel_per_ws=100,lambda_p=1,lambda_q=1):
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
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    f1 = open(res_dir + '/state_CNN_bais.log', 'w')

    Train_R_I = train_user[1] #this is rating; train_user_[0] is the item_index
    Train_R_J = train_item[1]
    Test_R = test_user[1]
    Valid_R = valid_user[1]

    if give_item_weight is True:
        item_weight = np.array([math.sqrt(len(i))
                                for i in Train_R_J], dtype=float)
        item_weight = (float(num_item) / item_weight.sum()) * item_weight
    else:
        item_weight = np.ones(num_item, dtype=float)

    pre_val_eval = 1e10


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




    
    # cnn_module = CNN_module(dimension, vocab_size, dropout_rate,
    #                         emb_dim, max_len, num_kernel_per_ws, init_W )
    cnn_module = CNN_module(dimension, vocab_size, dropout_rate,
                            emb_dim, max_len, num_kernel_per_ws, init_W,num_item,lambda_v,lambda_p,lambda_q)

    '''
    add index of items
    '''
    theta = np.random.uniform(size=(num_item,dimension))

    item_index=np.arange(num_item).reshape(-1,1)
    m_VV=np.concatenate((theta,item_index),axis=1)
    theta=cnn_module.get_projection_layer(CNN_X,m_VV)
    np.random.seed(133)
    U = np.random.uniform(size=(num_user, dimension))
    V = theta


    endure_count = 5
    count = 0

    better_mae=100
    better_rmse=100
    max_iter=100
    for iteration in xrange(max_iter):
        loss = 0
        tic = time.time()
        print "%d iteration\t(patience: %d)" % (iteration, count)

        VV = b * (V.T.dot(V)) + lambda_u * np.eye(dimension)#diagonal matrix
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
            A = VV + (a - b) * (V_i.T.dot(V_i))
            B = (a * V_i * (np.tile(R_i, (dimension, 1)).T)).sum(0) #np.tile() array copy; sum(0) is the sum of each column,sum(1) is the sum of each row;

            U[i] = np.linalg.solve(A, B)      #AX=B,X=A^(-1)B

            sub_loss[i] = -0.5 * lambda_u * np.dot(U[i], U[i])

        loss = loss + np.sum(sub_loss)
        print "=================================================================="
        print "the shape of V, V[i] {} {}".format(V.shape,V[0].shape)
        print "=================================================================="
        sub_loss = np.zeros(num_item)
        UU = b * (U.T.dot(U))

        old_V=V
        for j in xrange(num_item):
            idx_user = train_item[0][j]
            U_j = U[idx_user]
            R_j = Train_R_J[j]

            tmp_A = UU + (a - b) * (U_j.T.dot(U_j))
            A = tmp_A + lambda_v * item_weight[j] * np.eye(dimension)
            B = (a * U_j * (np.tile(R_j, (dimension, 1)).T)
                 ).sum(0) + lambda_v * item_weight[j] * theta[j]

            V[j] = np.linalg.solve(A, B) #A*X=B  X =A^-1*B

            sub_loss[j] = -0.5 * np.square(R_j * a).sum()
            sub_loss[j] = sub_loss[j] + a * np.sum((U_j.dot(V[j])) * R_j)
            sub_loss[j] = sub_loss[j] - 0.5 * np.dot(V[j].dot(tmp_A), V[j])

        loss = loss + np.sum(sub_loss)
        seed = np.random.randint(100000)
        # history = cnn_module.train(CNN_X, V, item_weight, seed)
        # theta = cnn_module.get_projection_layer(CNN_X)
        # print history,history.history.keys(),
        # print history.history['loss']
        # cnn_loss = history.history['loss'][-1]

        m_VV=np.concatenate((V,item_index),axis=1)

        cnn_module.train(CNN_X,m_VV,item_weight,seed)
        cnn_loss=cnn_module.train_loss
        l1_p_loss=cnn_module.l1_p_loss
        l2_p_loss=cnn_module.l2_p_loss



        logging.info("----------------------------------")
        logging.info("CNN loss {},l2_P_loss {},l1_p_loss {}".format(cnn_loss,l2_p_loss,l1_p_loss))
        logging.info("----------------------------------")
        theta=cnn_module.get_projection_layer(CNN_X,m_VV)


        # loss = loss - 0.5 * lambda_v * cnn_loss * num_item
        # loss = loss - 0.5 * lambda_v * cnn_loss * num_item-l2_p_loss*lambda_p -l1_p_loss*lambda_q
        loss=loss-0.5* cnn_loss

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
        print "\Sentiment=============better_rmse:{}=====better_mae:{}==============\n".format(better_rmse,better_mae)
        if (count == endure_count):
            break
        PREV_LOSS = loss


        # f1.write("Loss: %.5f Elpased: %.4fs Converge: %.6f Tr: %.5f Val: %.5f Te: %.5f\n" % (
        #     loss, elapsed, converge, tr_eval, val_eval, te_eval))
    # f1.close()