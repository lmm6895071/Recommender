'''
Created on Sep 14, 2017

@author: ming
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import logging
class MLP_module():

    def one_hot(self,R,num_classes):
        result=[]
        for item in R:
            temp=np.zeros([num_classes])
            temp[int(item)-1]=1
            result.append(temp)
        return np.array(result).reshape(R.shape[0],num_classes)

    def __init__(self, num_classes, embedding_size,num_user,num_item,lambda_u,lambda_v):
        logging.info("this is MLP model")
        self.num_classes=1
        num_classes=1
        l2_reg_lambda=10
        '''
        classdocs
        '''
        self.batch_size =512
        self.epochs =10
        # More than this epoch cause easily over-fitting on our data sets
        self.display_step=1
        self.keep_prob = 0.7

        # Keeping track of l2 regularization loss (optional)

        n_hidden_1=256
        n_hidden_2=128
        n_hidden_3=64
        n_hidden_4=8

        self.model=tf.Graph()
        with self.model.as_default():
            '''
            W = tf.get_variable(
                    'W',
                    shape=[num_filters_total, num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            '''
            l2_loss =tf.to_float(tf.constant(0.0))
            self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
            self.input_u=tf.placeholder(tf.int32,[None],name="input_u")
            self.input_v=tf.placeholder(tf.int32,[None],name="input_v")
            self.input_r=tf.placeholder(tf.float32,[None],name="input_r")

            '''
            He approach:(Relu function)
            random_uniform: scale =  np.sqrt(6.0/embedding_size)
            normal initial: stddev = np.sqrt(2.0/embedding_size)
            
            Xavier approach:(softmax,tanh, sidmoid function)
            random_uniform: scale =  np.sqrt(3.0/embedding_size)
            normal initial: stddev = np.sqrt(embedding_size)
            
            '''
            # self.U=tf.Variable(tf.random_uniform([num_user, embedding_size],0,np.sqrt(6.0/embedding_size)), name='U')#
            # self.V=tf.Variable(tf.random_uniform([num_item,embedding_size],0,np.sqrt(6.0/embedding_size)),name='V')
            
            self.U=tf.Variable(tf.truncated_normal([num_user, embedding_size],stddev=np.sqrt(embedding_size)), name='U')#
            self.V=tf.Variable(tf.truncated_normal([num_item,embedding_size],stddev=np.sqrt(embedding_size)),name='V')
            '''
            self.U=tf.get_variable(
                'U',
                shape=[num_user, embedding_size],
                initializer=tf.contrib.layers.xavier_initializer())
            self.V = tf.get_variable(
                'V',
                shape=[num_item, embedding_size],
                initializer=tf.contrib.layers.xavier_initializer())
            '''

            # input_uu=tf.nn.l2_normalize(tf.nn.embedding_lookup(self.U, self.input_u),1)
            # input_vv=tf.nn.l2_normalize(tf.nn.embedding_lookup(self.V, self.input_v),1) #find vertor for word


            input_uu=tf.nn.embedding_lookup(self.U, self.input_u)
            input_vv=tf.nn.embedding_lookup(self.V, self.input_v)

            print "the shape of uu, vv",input_uu,input_vv

            input_x= tf.concat([input_uu,input_vv],1)

            print "the shape of input_x is {}".format(input_x)
            # '''
            self.weight = {
                'h1': tf.nn.l2_normalize(tf.Variable(tf.truncated_normal([2*embedding_size, n_hidden_1],stddev=np.sqrt(2.0/(2*embedding_size+ n_hidden_1)))),1),
                'h2': tf.nn.l2_normalize(tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],stddev=np.sqrt(2.0/(n_hidden_1+n_hidden_2)))),1),
                'h3': tf.nn.l2_normalize(tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],stddev=np.sqrt(2.0/(n_hidden_3+n_hidden_2)))),1),
                'h4': tf.nn.l2_normalize(tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4],stddev=np.sqrt(2.0/(n_hidden_3+n_hidden_4)))),1),
                # 'out':tf.Variable(tf.truncated_normal([n_hidden_3,num_classes],stddev=np.sqrt(2.0/(n_hidden_3))))
            }
            self.bias = {
                'h1': tf.Variable(tf.random_normal([n_hidden_1])),
                'h2': tf.Variable(tf.random_normal([n_hidden_2])),
                'h3': tf.Variable(tf.random_normal([n_hidden_3])),
                'h4': tf.Variable(tf.random_normal([n_hidden_4])),
                'out': tf.Variable(tf.random_normal([num_classes]))
            }
            # '''
            # self.weight = {
            #     'h1':  tf.Variable(tf.truncated_normal([2*embedding_size, n_hidden_1],stddev=np.sqrt(2*embedding_size+ n_hidden_1))) ,
            #     'h2':  tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],stddev=np.sqrt(n_hidden_1+n_hidden_2))), 
            #     'h3':  tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],stddev=np.sqrt(n_hidden_3+n_hidden_2))), 
            #     'h4':  tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4],stddev=np.sqrt(n_hidden_3+n_hidden_4))), 
            #     # 'out':tf.Variable(tf.truncated_normal([n_hidden_3,num_classes],stddev=2.0/(n_hidden_3)))
            # }
            # self.bias = {
            #     'h1': tf.Variable(tf.random_normal([n_hidden_1])),
            #     'h2': tf.Variable(tf.random_normal([n_hidden_2])),
            #     'h3': tf.Variable(tf.random_normal([n_hidden_3])),
            #     'h4': tf.Variable(tf.random_normal([n_hidden_4])),
            #     'out': tf.Variable(tf.random_normal([num_classes]))
            # }



            layer1 = tf.add(tf.matmul(input_x, self.weight['h1']), self.bias['h1'])
            layer1 = tf.nn.dropout(layer1,self.dropout_keep_prob)
            layer1 = tf.nn.relu(layer1)
            layer2 = tf.add(tf.matmul(layer1, self.weight['h2']), self.bias['h2'])
            layer2 = tf.nn.dropout(layer2,self.dropout_keep_prob)
            layer2 = tf.nn.relu(layer2)
            layer3 =tf.add(tf.matmul(layer2, self.weight['h3']), self.bias['h3'])
            layer3 = tf.nn.dropout(layer3,self.dropout_keep_prob)
            layer3 = tf.nn.relu(layer3)

            # layer4 = tf.add(tf.matmul(layer3, weight['h4']), bias['h4'])
            # layer4 = tf.nn.relu(layer4)
            # out_layer = tf.add(tf.matmul(layer3, weight['out']), bias['out'])

            alpha = 0.5
            gmf_out = alpha * tf.multiply(input_uu,input_vv)
            combine_layer = tf.concat((gmf_out,(1-alpha)*layer3),1)
            self.weight['out'] = tf.Variable(tf.truncated_normal([n_hidden_3+embedding_size,num_classes],stddev=np.sqrt(n_hidden_3)))

            self.scores=(tf.add(tf.matmul(combine_layer,self.weight['out']),self.bias['out']))#out_layer

            # self.final_score=tf.add(tf.cast(tf.argmax(self.scores,1),dtype=tf.int64),tf.constant(1,dtype=tf.int64))
            # self.real_score=tf.add(tf.cast(tf.argmax(self.input_r,1),dtype=tf.int64),tf.constant(1,dtype=tf.int64))

            # Calculate mean loss
            with tf.name_scope('loss'):
                # losses=tf.reduce_mean(tf.square(tf.to_float(self.real_score)-tf.to_float(self.final_score)))
                losses=tf.reduce_mean(tf.square(self.scores-self.input_r))
                # losses =-1*(tf.reduce_sum(tf.log(self.scores*self.input_r)))

                ws = ['h1','h2','h3','out']
                for item in ws:
                    l2_loss += tf.nn.l2_loss(self.weight[item])
                    l2_loss+= tf.nn.l2_loss(self.bias[item])

                self.loss = losses + l2_reg_lambda * (l2_loss + lambda_u * tf.reduce_sum(tf.square(self.U)) + lambda_v * tf.reduce_sum(tf.square(self.V)))
                self.out_loss=tf.sqrt(losses)


            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.train_op= tf.train.RMSPropOptimizer(0.001, 0.99).minimize(self.loss, global_step=self.global_step)
            # self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.loss,global_step=self.global_step)
            self.init=tf.global_variables_initializer()

        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        self.sess=tf.Session(graph=self.model,config=session_conf)
        self.sess.run(self.init)
        # print self.sess.run(self.weight)
        # tts  =  ['out','h3','h2','h1']

        # for item in tts:
        #     print "weight['{}']".format(item),
        #     print w[item]
        #     print "bias:['{}']".format(item),
        #     print b[item]



    def train(self, U, V, RR, seed):
        np.random.seed(seed)
        U = np.random.permutation(U)
        np.random.seed(seed)
        V = np.random.permutation(V)
        np.random.seed(seed)
        # Rs = np.random.permutation(RR)
        # R=self.one_hot(Rs,self.num_classes)
        R=np.random.permutation(RR)

        print("###############Train...MLP module################")
        # print("the shape of U is {},V is {},R is {}".format(U.shape,V.shape,R.shape))

        data_test_size=500

        data_size=len(U)
        best_loss=1000000;

        batch_size=self.batch_size
        total_batch = int(R.shape[0] / batch_size)
        num_batches_per_epoch = total_batch + 1
        shuffle=True

        for epoch in range(self.epochs):
            total_cost=0.0

            U_train=[]
            V_train=[]
            R_train=[]
            
            shuffle_indices = np.random.permutation(np.arange(data_size))
            U_train = U[shuffle_indices]
            V_train = V[shuffle_indices]
            R_train = R[shuffle_indices]

            for i in range(total_batch):
                if i <total_batch:
                    feed_dict={
                        self.input_u:U_train[i*batch_size:(i+1)*batch_size],
                        self.input_v:V_train[i*batch_size:(i+1)*batch_size],
                        self.input_r:R_train[i*batch_size:(i+1)*batch_size],
                        self.dropout_keep_prob: self.keep_prob
                    }
                elif (i+1)*batch_size<R.shape[0]:
                    feed_dict={
                        self.input_u:U[(i+1)*batch_size:R.shape[0]],
                        self.input_v:V[(i+1)*batch_size:R.shape[0] ],
                        self.input_r:R[(i+1)*batch_size:R.shape[0] ],
                        self.dropout_keep_prob:self.keep_prob
                    }
                _, c,s,w,b= self.sess.run([self.train_op,self.loss,self.global_step,self.weight,self.bias], feed_dict=feed_dict)
                total_cost += c



            if epoch% self.display_step ==0:
                # print '-------------------start epochs:',epoch+1,"-------------------------"
                print('epochs:', '%04d' % (epoch+1), 'cost=', '{:.9f}'.format(total_cost ))
                # print "weight:"
                # tts  =  ['out','h3','h2','h1']
                # for item in tts:
                #     print "weight['{}']".format(item),
                #     print w[item]
                #     print "bias:['{}']".format(item),
                #     print b[item].reshape(1,b[item].shape[0])
                # print "----------------end epochs:",epoch+1,"----------------------------"



                shuffle_indices = np.random.permutation(np.arange(data_size))
                U_test = U[shuffle_indices[0:data_test_size]]
                V_test=V[shuffle_indices[0:data_test_size]]
                R_test=R[shuffle_indices[0:data_test_size]]

                result=self.predict(U_test,V_test,R_test)
                if result<=best_loss:
                    best_loss=result

                # print "best loss is Test:",result
        print "weight:"
        tts  =  ['out','h1']
        for item in tts:
            print "weight['{}']".format(item),
            # print w[item].reshape(1,w[item].shape[0])
            print w[item]
            print "bias:['{}']".format(item),
            # print b[item].reshape(1,b[item].shape[0])
            print b[item]
        print "###############End...MLP module###############"

        return total_cost

    def predict(self,U,V,R):
        batch_size=self.batch_size
        total_batch = int(R.shape[0] / batch_size)

        total_loss=0.0
        self.out_put_score=[]
        for tb in range(total_batch+1):
            if tb <= total_batch:
                feed_dict={
                    self.input_u:U[tb*batch_size:(tb+1)*batch_size],
                    self.input_v:V[tb*batch_size:(tb+1)*batch_size],
                    self.input_r:R[tb*batch_size:(tb+1)*batch_size],
                    self.dropout_keep_prob:1.0
                }
            elif (tb+1)*batch_size<R.shape[0]:
                feed_dict={
                    self.input_u:U[(tb+1)*batch_size:R.shape[0]],
                    self.input_v:V[(tb+1)*batch_size:R.shape[0] ],
                    self.input_r:R[(tb+1)*batch_size:R.shape[0] ],
                    self.dropout_keep_prob:1.0
                }
            o,s= self.sess.run([self.out_loss,self.scores], feed_dict=feed_dict)
            total_loss = total_loss+o
            self.out_put_score = np.concatenate((np.array(self.out_put_score),s.reshape(s.shape[0])),axis=0)
        
        # print (np.array(self.out_put_score)).shape,len(self.out_put_score)
        # print "predict loss is {}".format(c)
        # print "the shape of out_put_score is :", np.array(self.out_put_score).shape
        return total_loss

    def model_close(self):

        self.model.close()

