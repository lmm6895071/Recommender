'''
Created on  march, 2018

@author: ming
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import logging
class Double_MLP_module():


    def __init__(self, num_classes, embedding_size,num_user,num_item,lambda_u,lambda_v):
        logging.info("this is MLP model")
        self.num_classes=1
        num_classes=1
        l2_reg_lambda=1
        '''
        classdocs
        '''
        self.batch_size =512
        self.epochs =5
        # More than this epoch cause easily over-fitting on our data sets
        self.display_step=1



        # self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # Keeping track of l2 regularization loss (optional)

        n_hidden_1=64
        n_hidden_2=32
        n_hidden_3=16
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

            self.input_u=tf.placeholder(tf.int32,[None],name="input_u")
            self.input_v=tf.placeholder(tf.int32,[None],name="input_v")
            self.input_r=tf.placeholder(tf.float32,[None],name="input_r")

            # self.U=tf.Variable(tf.random_uniform([num_user, embedding_size],-np.sqrt(6.0/embedding_size),np.sqrt(6.0/embedding_size)), name='U')#
            # self.V=tf.Variable(tf.random_uniform([num_item,embedding_size],-np.sqrt(6.0/embedding_size),np.sqrt(6.0/embedding_size)),name='V')
            '''
            input: 
            '''

            self.U=tf.Variable(tf.random_uniform([num_user, embedding_size],0,1), name='U')#
            self.V=tf.Variable(tf.random_uniform([num_item,embedding_size],1,1),name='V')
            

            # self.U=tf.get_variable(
            #     'U',
            #     shape=[num_user, embedding_size],
            #     initializer=tf.contrib.layers.xavier_initializer())
            # self.V = tf.get_variable(
            #     'V',
            #     shape=[num_item, embedding_size],
            #     initializer=tf.contrib.layers.xavier_initializer())


            # input_uu=tf.nn.l2_normalize(tf.nn.embedding_lookup(self.U, self.input_u),1)
            # input_vv=tf.nn.l2_normalize(tf.nn.embedding_lookup(self.V, self.input_v),1) #find vertor for word


            input_uu=tf.nn.embedding_lookup(self.U, self.input_u)
            input_vv=tf.nn.embedding_lookup(self.V, self.input_v)

            print "the shape of uu, vv",input_uu,input_vv
            # input_x= tf.concat([input_uu,input_vv],1)
            # print "the shape of input_x is {} ".format(input_x)


            weight_u = {
                'h1':  tf.Variable(tf.truncated_normal([embedding_size, n_hidden_1],stddev=2.0/(2*embedding_size+ n_hidden_1))), 
                'h2':  tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],stddev=2.0/(n_hidden_1+n_hidden_2))), 
                'h3':  tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],stddev=2.0/(n_hidden_3+n_hidden_2))), 
                'h4':  tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4],stddev=2.0/(n_hidden_3+n_hidden_4))), 
                'out': tf.Variable(tf.truncated_normal([n_hidden_3,num_classes],stddev=2.0/(n_hidden_3)))
            }
            bias_u = {
                'h1': tf.Variable(tf.random_normal([n_hidden_1])),
                'h2': tf.Variable(tf.random_normal([n_hidden_2])),
                'h3': tf.Variable(tf.random_normal([n_hidden_3])),
                'h4': tf.Variable(tf.random_normal([n_hidden_4])),
                'out': tf.Variable(tf.random_normal([num_classes]))
            }

            u_layer1 = tf.add(tf.matmul(input_uu, weight_u['h1']), bias_u['h1'])
            u_layer1 = tf.nn.relu(u_layer1)
            u_layer2 = tf.add(tf.matmul(u_layer1, weight_u['h2']), bias_u['h2'])
            u_layer2 = tf.nn.relu(u_layer2)
            u_layer3 = tf.add(tf.matmul(u_layer2, weight_u['h3']), bias_u['h3'])
            u_layer3 = tf.nn.relu(u_layer3)
            # u_layer4 = tf.add(tf.matmul(u_layer3, weight_u['h4']), bias_u['h4'])
            # u_layer4 = tf.nn.relu(u_layer4)
            u_out_layer = u_layer3# tf.add(tf.matmul(u_layer3, weight_u['out']), bias_u['out'])

            weight_v = {
                'h1':  tf.Variable(tf.truncated_normal([embedding_size, n_hidden_1],stddev=2.0/(2*embedding_size+ n_hidden_1))), 
                'h2':  tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],stddev=2.0/(n_hidden_1+n_hidden_2))), 
                'h3':  tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],stddev=2.0/(n_hidden_3+n_hidden_2))), 
                'h4':  tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4],stddev=2.0/(n_hidden_3+n_hidden_4))), 
                'out': tf.Variable(tf.truncated_normal([n_hidden_3,num_classes],stddev=2.0/(n_hidden_3)))
            }
            bias_v = {
                'h1': tf.Variable(tf.random_normal([n_hidden_1])),
                'h2': tf.Variable(tf.random_normal([n_hidden_2])),
                'h3': tf.Variable(tf.random_normal([n_hidden_3])),
                'h4': tf.Variable(tf.random_normal([n_hidden_4])),
                'out': tf.Variable(tf.random_normal([num_classes]))
            }
            v_layer1 = tf.add(tf.matmul(input_vv, weight_v['h1']), bias_v['h1'])
            v_layer1 = tf.nn.relu(v_layer1)
            v_layer2 = tf.add(tf.matmul(v_layer1, weight_v['h2']), bias_v['h2'])
            v_layer2 = tf.nn.relu(v_layer2)
            v_layer3 = tf.add(tf.matmul(v_layer2, weight_v['h3']), bias_v['h3'])
            v_layer3 = tf.nn.relu(v_layer3)
            # v_layer4 = tf.add(tf.matmul(v_layer3, weight_v['h4']), bias_v['h4'])
            # v_layer4 = tf.nn.relu(v_layer4)
            v_out_layer = v_layer3# tf.add(tf.matmul(v_layer3, weight_v['out']), bias_v['out'])


            out_layer =tf.reduce_sum(tf.multiply(u_out_layer,v_out_layer)+(v_out_layer+u_out_layer))
            print "_+=++++",tf.shape(out_layer)


            self.scores = out_layer
            '''
            cos distence
            '''
            # self.scores= out_layer/(tf.reduce_sum(tf.sqrt(u_out_layer))*tf.reduce_sum(tf.sqrt(v_out_layer)))#tf.nn.softmax(out_layer)
            # self.scores=tf.maximum(self.scores,tf.to_float(1e-10))

            # Calculate mean loss
            with tf.name_scope('loss'):
                # losses=tf.reduce_mean(tf.square(tf.to_float(self.real_score)-tf.to_float(self.final_score)))
                # losses= -1*(tf.reduce_sum(tf.log(self.scores*self.input_r)))
                # losses =  tf.nn.sigmoid_cross_entropy_with_logits(self.scores,self.input_r)
                losses=tf.reduce_mean(tf.square(self.scores-self.input_r))

                ws = ['h1','h2','h3','out']
                
                for item in ws:
                    l2_loss += tf.nn.l2_loss(weight_u[item])
                    l2_loss += tf.nn.l2_loss(weight_v[item])

                self.loss = losses + l2_reg_lambda * l2_loss
                self.out_loss=tf.sqrt(losses)


            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.train_op= tf.train.RMSPropOptimizer(0.001, 0.99).minimize(self.loss, global_step=self.global_step)
            # self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.loss,global_step=self.global_step)
            self.init=tf.global_variables_initializer()

        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        self.sess=tf.Session(graph=self.model,config=session_conf)
        self.sess.run(self.init)



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
                        self.input_r:R_train[i*batch_size:(i+1)*batch_size]
                    }
                elif (i+1)*batch_size<R.shape[0]:
                    feed_dict={
                        self.input_u:U[(i+1)*batch_size:R.shape[0]],
                        self.input_v:V[(i+1)*batch_size:R.shape[0] ],
                        self.input_r:R[(i+1)*batch_size:R.shape[0] ]
                    }
                _, c,s= self.sess.run([self.train_op,self.loss,self.global_step], feed_dict=feed_dict)
                total_cost += c

            if epoch% self.display_step ==0:
                print('epochs:', '%04d' % (epoch+1), 'cost=', '{:.9f}'.format(total_cost ))



                shuffle_indices = np.random.permutation(np.arange(data_size))
                U_test = U[shuffle_indices[0:data_test_size]]
                V_test=V[shuffle_indices[0:data_test_size]]
                R_test=R[shuffle_indices[0:data_test_size]]

                result=self.predict(U_test,V_test,R_test)
                if result<=best_loss:
                    best_loss=result

                # print "best loss is Test:",result
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
                    self.input_r:R[tb*batch_size:(tb+1)*batch_size]
                }
            elif (tb+1)*batch_size<R.shape[0]:
                feed_dict={
                    self.input_u:U[(tb+1)*batch_size:R.shape[0]],
                    self.input_v:V[(tb+1)*batch_size:R.shape[0] ],
                    self.input_r:R[(tb+1)*batch_size:R.shape[0] ]
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

