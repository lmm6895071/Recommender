'''
Created on Sep 8, 2017

@author: ming
'''

import os
import time
import logging
from util import eval_RMSE
import util
import math
import numpy as np
from mlp_module.m_keras_mlp import Keras_MLP_Moudle

def ConvMF_keras_NMF(res_dir, train_user, train_item, valid_user, test_user,
           R, CNN_X, vocab_size, init_W=None, give_item_weight=True,
           max_iter=50, lambda_u=1, lambda_v=100, dimension=50,
           dropout_rate=0.2, emb_dim=200, max_len=300, num_kernel_per_ws=100):
    # explicit setting
    a = 1
    b = 0
    dimension=50
    num_class=5

    num_user = R.shape[0]
    num_item = R.shape[1]
    print "===================================ConvMF Models==================================="
    print "\tnum_user is:{}".format(num_user)
    print "\tnum_item is:{}".format(num_item)
    print "==================================================================================="
    PREV_LOSS = 1e-50
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    # f1 = open(res_dir + '/state_MF.log', 'w')

    Train_R_I = train_user[1] #this is rating; train_user_[0] is the item_index
    Train_R_J = train_item[1]
    Test_R = test_user[1]
    Valid_R = valid_user[1]
 
    pre_val_eval = 1e10


    endure_count = 100
    count = 0

    print "===================================numpy dot==================================="
    print np.dot.__module__
    print "==============================================================================="
    '''
    train_user[0]=[[item1,item2,item3...],[item1,itme3],[item3,item2]...]
    train_user[1]=[[rating1,rating2,rating3...],[rating1,rating3],[rating2,rating5]...]
    R_i = Train_R_I[i]#[rating1,rating2,rating3...]
    '''
    
    def get_instance(data):
        input_u=[]
        input_v=[]
        input_r=[]
        for i in xrange(num_user):
            R_i = data[1][i]
            idx_item = data[0][i]
            # print len(R_i),len(idx_item)
            # print idx_item
            for j in range(len(idx_item)):
                input_u.append(i)
                input_v.append(idx_item[j])
                input_r.append(R_i[j])
        input_r=np.array(input_r)
        input_v=np.array(input_v)
        input_u=np.array(input_u)
        print "=================================="
        print  "the shape of input_u,input_v,input_r ",input_u.shape,input_v.shape,input_r.shape
        print "=================================="
        return input_u,input_v,input_r

    
    input_u,input_v,input_r=get_instance(train_user)

    v_input_u,v_input_v,v_input_r=get_instance(valid_user)
    t_input_u,t_input_v,t_input_r=get_instance(test_user)

    model=Keras_MLP_Moudle(num_user,num_item)

    max_iter=3
    k=50
    for iteration in xrange(max_iter):
        loss = 0
        tic = time.time()
        print "%d iteration\t(patience: %d)" % (iteration, count)
        seed = np.random.randint(100000)
        loss=model.train(input_u,input_v,input_r)

        print "loss of the train process for mlp is {}".format(loss)

        loss=model.predict(input_u,input_v,input_r)
        print "loss of the predict process for train-mlp is {}".format(loss)


        tr_eval = loss
        val_eval = model.predict(v_input_u,v_input_v,v_input_r)
        v_out=model.out_put_score
        v_recall=util.recall_top_k(num_user,v_input_r,v_out,200)

        te_eval = model.predict(t_input_u,t_input_v,t_input_r)
        t_out=model.out_put_score
        t_recall=util.recall_top_k(num_user,t_input_r,t_out,200)

        toc = time.time()
        elapsed = toc - tic

        converge = abs((loss - PREV_LOSS) / PREV_LOSS)

        if (val_eval < pre_val_eval):
            pass
        else:
            count = count + 1

        pre_val_eval = val_eval

        print "Loss: %.5f Elpased: %.4fs Converge: %.6f Train: %.5f Validation: %.5f Test: %.5f" % (
            loss, elapsed, converge, tr_eval, val_eval, te_eval)
        print "recall@{} validation:{}  Test:{}".format(k,v_recall,t_recall)
        # f1.write("Loss: %.5f Elpased: %.4fs Converge: %.6f Train: %.5f Validation: %.5f Test: %.5f\n" % (
        #     loss, elapsed, converge, tr_eval, val_eval, te_eval))

        if (count == endure_count):
            break

        PREV_LOSS = loss

    # f1.close()
