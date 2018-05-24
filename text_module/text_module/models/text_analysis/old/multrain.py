import os
import sys
import json
import time
import logging
import dataHelper
import numpy as np
import tensorflow as tf
from text_cnn import TextCNN
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split
from dataHelper import LoadData
from sklearn.metrics import classification_report
from sklearn import metrics

from keras.preprocessing import sequence
logging.getLogger().setLevel(logging.INFO)

np.random.seed(1337)


class CNN_module():
    """docstring for ClassName"""
    def __init__(self, output_dimesion, vocab_size, dropout_rate, emb_dim, max_len, nb_filters, init_W,num_items):
        '''load parameters'''
        self.filter_lengths = [3, 4, 5]
        self.l2_reg_lambda=40
        self.dropout_keep_prob=0.5

        self.evaluate_every=10#10
        self.batch_size = 128#128
        # More than this epoch cause easily over-fitting on our data sets
        self.nb_epoch = 10#5

        self.max_len=max_len
        self.output_dimesion=output_dimesion
        self.vocab_size=vocab_size
        self.dropout_rate=dropout_rate
        self.emb_dim=emb_dim
        self.nb_filters=nb_filters
        self.init_W=init_W
        self.num_items=num_items
        m_l_rates=[0.1,0.05,0.01,0.005,0.001]
    def train(self, X_train, V, item_weight, seed):
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)#sequence of word_idx [[1,2,3],[2,3,4,6],...]
        np.random.seed(seed)
        # X_train = np.random.permutation(X_train)
        np.random.seed(seed)
        # V = np.random.permutation(V)
        np.random.seed(seed)
        # item_weight = np.random.permutation(item_weight)

        logging.info("Train...CNN module")
        self.m_train(X_train,V)
        #history= self.model.fit({'input': X_train, 'output': V},verbose=0, batch_size=self.batch_size, nb_epoch=self.nb_epoch, sample_weight={'output': item_weight})
        return self.train_loss


    def get_projection_layer(self):
        # X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
        # Y = self.model.predict({'input': X_train}, batch_size=len(X_train))['output']
        return self.final_out
    def batch_iter(self,data, batch_size, num_epochs, shuffle=True):
        """Iterate the data batch by batch"""
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int(data_size / batch_size) + 1

        for epoch in range(num_epochs):
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

    def m_train(self,X_train,V):
        '''step 1: split data to test,dev,train'''
        X_trains,x_test,y_trains,y_test = train_test_split(X_train,V,test_size=0.1)
        x_train,x_dev, y_train,y_dev=train_test_split(X_trains,y_trains,test_size=0.1)

        logging.info('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))

        # graph = self.graph
        graph=tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
            sess = tf.Session(config=session_conf)
            sess=tf.Session()
            with sess.as_default():

                cnn = TextCNN(
                    sequence_length=self.max_len,
                    num_classes=self.output_dimesion,
                    vocab_size=self.vocab_size,
                    embedding_size=self.emb_dim,
                    filter_sizes=self.filter_lengths,#feature windows 2,3,4,5
                    num_filters=self.nb_filters,#map
                    l2_reg_lambda=self.l2_reg_lambda,
                    wordVocab=self.init_W,
                    num_items=self.num_items
                    )
                global_step = tf.Variable(0, name="global_step", trainable=False)

                steps=0


                train_op= tf.train.RMSPropOptimizer(0.005, 0.99).minimize(cnn.m_loss, global_step=global_step)


                optimizer1 = tf.train.AdamOptimizer(0.5)
                grads_and_vars1 = optimizer1.compute_gradients(cnn.m_loss)
                train_op1 = optimizer1.apply_gradients(grads_and_vars1, global_step=global_step)
                
                optimizer2 = tf.train.AdamOptimizer(0.01)
                grads_and_vars2 = optimizer2.compute_gradients(cnn.m_loss)
                train_op2 = optimizer2.apply_gradients(grads_and_vars2, global_step=global_step)
                
                optimizer3 = tf.train.AdamOptimizer(0.05)
                grads_and_vars3 = optimizer3.compute_gradients(cnn.m_loss)
                train_op3 = optimizer3.apply_gradients(grads_and_vars3, global_step=global_step)


                sess.run(tf.initialize_all_variables())

                # One training step: train the model with one batch
                def train_step(x_batch, y_batch):
                    y=y_batch[:,0:-1]
                    iy=y_batch[:,-1]
                    # logging.info("&&&&&& shape {} {} {}".format(len(x_batch),y.shape,iy.shape))
                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y,
                        cnn.input_index:iy,
                        cnn.dropout_keep_prob:self.dropout_keep_prob
                        }
                    # logging.info("input shape {} {} {}".format(type(x_batch),y.shape,iy.shape))
                    # if steps<150:
                    #     _, step, loss= sess.run([train_op1, global_step, cnn.m_loss], feed_dict=feed_dict)
                    # elif steps<300:
                    #     _, step, loss= sess.run([train_op2, global_step, cnn.m_loss], feed_dict=feed_dict)
                    # else:
                    #     _, step, loss= sess.run([train_op3, global_step, cnn.m_loss], feed_dict=feed_dict)
                    
                    _, step, loss= sess.run([train_op, global_step, cnn.m_loss], feed_dict=feed_dict)

                    logging.info("the step={} of training ,loss={}".format(step,loss))
                    return loss

                # One evaluation step: evaluate the model with one batch
                def dev_step(x_batch, y_batch):
                    y=y_batch[:,0:-1]
                    iy=y_batch[:,-1]
                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y,
                        cnn.input_index:iy,
                        cnn.dropout_keep_prob: 1.0
                        }
                    step, loss= sess.run([global_step, cnn.m_loss], feed_dict)
                    logging.info("the step={} of dev_step ,loss={}".format(step,loss))
                    return loss

                def test_step(x_batch, y_batch):
                    y=y_batch[:,0:-1]
                    iy=y_batch[:,-1]
                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y,
                        cnn.input_index:iy,
                        cnn.dropout_keep_prob: 1.0
                        }
                    step, loss = sess.run([global_step, cnn.m_loss], feed_dict)
                    logging.info("the step={} of dev_step ,loss={} ".format(step,loss))
                    return loss


                # Training starts here
                train_batches = self.batch_iter(list(zip(x_train,y_train)),self.batch_size,self.nb_epoch)
                logging.info("train_batches type is {}".format(type(train_batches)))
                best_accuracy, best_at_step = 0, 0

                """Step 6: train the cnn model with x_train and y_train (batch by batch)"""
                logging.info("start CNN train: batch_size={},num_epochs={}".format(self.batch_size,self.nb_epoch))
                PP_train=[]
                for train_batch in train_batches:

                   
                    # logging.info("+++++++++++++++train_batches type is {},{}".format(type(train_batch),train_batch.shape))
                    try:
                        x_train_batch, y_train_batch = zip(*train_batch)
                        # logging.info(" type {} {} {}".format(type(x_train_batch),len(x_train_batch),len(x_train_batch[0])))

                    except Exception as err:
                        logging.info("this zip is error;{}".format(err))
                    x_train_batch=np.array(list(x_train_batch))
                    y_train_batch=np.array(list(y_train_batch))

                    # logging.info("training process data type {} {}".format(x_train_batch.shape,y_train_batch.shape))
                    
                    mloss=train_step(x_train_batch,y_train_batch)
                    PP_train.append(mloss)

                    self.train_loss=mloss
                    current_step = tf.train.global_step(sess, global_step)

                    """Step 6.1: evaluate the model with x_dev and y_dev (batch by batch)"""
                    if current_step % self.evaluate_every == 0:
                        dev_batches =self.batch_iter(list(zip(x_dev, y_dev)), self.batch_size, 10)
                        total_dev_correct = 0
                        PP_DEV=[]
                        for dev_batch in dev_batches:
                            try:
                                x_dev_batch, y_dev_batch = zip(*dev_batch)
                            except Exception as err:
                                logging.info("this dev zip is error;{}".format(err))
                            x_dev_batch=np.array(list(x_dev_batch))
                            y_dev_batch=np.array(list(y_dev_batch))
                            dev_loss = dev_step(x_dev_batch, y_dev_batch)
                            total_dev_correct=total_dev_correct+dev_loss
                            PP_DEV.append(dev_loss)

                        dev_accuracy = float(total_dev_correct) / len(y_dev)
                        logging.critical('Accuracy on dev set: {}'.format(dev_accuracy))
                        steps+=self.evaluate_every


                        """Step 6.2: save the model if it is the best based on accuracy of the dev set"""
                        if dev_accuracy >= best_accuracy:
                            best_accuracy, best_at_step = dev_accuracy, current_step
                            logging.critical('Best accuracy {} at step {}'.format(best_accuracy, best_at_step))
                logging.info("train process end")

                """Step 7: predict x_test (batch by batch)"""
                logging.info("start predict ")
                test_batches = self.batch_iter(list(zip(x_test, y_test)), self.batch_size, 1)
                PP=[]
                total_test_loss=0
                for test_batch in test_batches:
                    try:
                        x_test_batch, y_test_batch = zip(*test_batch)
                    except Exception as err:
                        logging.info("this test zip is error;{}".format(err))
                    x_test_batch=np.array(list(x_test_batch))
                    y_test_batch=np.array(list(y_test_batch))
                    mloss = test_step(x_test_batch, y_test_batch)

                    total_test_loss=total_test_loss+mloss
                    PP.append(mloss)

                logging.critical('loss on test set is {} based on the best model'.format(total_test_loss/len(y_test)))

                logging.critical('The processes is completed')

                logging.info("----------------------------------------")
                logging.info("start predict and output result")
                logging.info("-------------------------------------------")
                Y=np.array(list(V))
                y=Y[:,0:-1]
                iy=Y[:,-1]
                feed_dict = {
                    cnn.input_x: X_train,
                    cnn.input_y: y,
                    cnn.input_index:iy,
                    cnn.dropout_keep_prob: 1.0
                    }
                loss,out,p_loss= sess.run([cnn.m_loss,cnn.scores,cnn.p_loss], feed_dict)
                logging.info("the step={} of dev_step ,loss={} ".format(tf.shape(out),loss))
                self.final_out=out
                self.train_loss=loss
                self.p_loss=p_loss
                logging.info("train loss is {},{}".format(self.train_loss,p_loss))
                logging.info("------------------end-------------------------")
                return self.train_loss


if __name__ == '__main__':
    # python3 train.py ./data/consumer_complaints.csv.zip ./parameters.json
    # CNN_module(filename,data_type)
    pass
