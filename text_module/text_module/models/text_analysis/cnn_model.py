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
    def __init__(self, output_dimesion, vocab_size, dropout_rate, emb_dim, max_len, nb_filters, init_W,num_items,lambda_v,lambda_p,lambda_q):
        '''load parameters'''
        self.filter_lengths = [3, 4, 5]
        self.l2_reg_lambda=1
        self.dropout_keep_prob=0.5

        self.evaluate_every=10#10
        self.batch_size = 256
        # More than this epoch cause easily over-fitting on our data sets
        self.nb_epoch =20

        self.max_len=max_len
        self.output_dimesion=output_dimesion
        self.vocab_size=vocab_size
        self.dropout_rate=dropout_rate
        self.emb_dim=emb_dim
        self.nb_filters=nb_filters
        self.init_W=init_W
        self.num_items=num_items

        self.graph=tf.Graph()
        with self.graph.as_default():
            self.cnn = TextCNN(
                    sequence_length=self.max_len,
                    num_classes=self.output_dimesion,
                    vocab_size=self.vocab_size,
                    embedding_size=self.emb_dim,
                    filter_sizes=self.filter_lengths,#feature windows 2,3,4,5
                    num_filters=self.nb_filters,#map
                    l2_reg_lambda=self.l2_reg_lambda,
                    wordVocab=self.init_W,
                    num_items=self.num_items,
                    lambda_v=lambda_v,
                    lambda_p=lambda_p,
                    lambda_q=lambda_q
                    )
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.train_op= tf.train.RMSPropOptimizer(0.0001,0.99).minimize(self.cnn.m_loss, global_step=self.global_step)
            self.init=tf.global_variables_initializer()



        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        self.sess=tf.Session(graph=self.graph,config=session_conf)
        self.sess.run(self.init)





    # One training step: train the model with one batch
    def train_step(self,x_batch, y_batch,type=1):

        y=y_batch[:,0:-1]
        iy=y_batch[:,-1]
        # logging.info("&&&&&& shape {} {} {}".format(len(x_batch),y.shape,iy.shape))
        feed_dict = {
            self.cnn.input_x: x_batch,
            self.cnn.input_y: y,
            self.cnn.input_index:iy,
            self.cnn.dropout_keep_prob:self.dropout_keep_prob
        }
        if type==1:
            _, step, loss= self.sess.run([self.train_op, self.global_step, self.cnn.m_loss], feed_dict=feed_dict)
            logging.info("the step={} of training ,loss={}".format(step,loss))
            return loss
        else:
            loss,l1_p_loss,l2_p_loss=self.sess.run([self.cnn.m_loss,self.cnn.l1_p_loss,self.cnn.l2_p_loss], feed_dict)
            return (loss,l1_p_loss,l2_p_loss)

    # One evaluation step: evaluate the model with one batch
    def dev_step(self,x_batch, y_batch):
        y=y_batch[:,0:-1]
        iy=y_batch[:,-1]
        feed_dict = {
            self.cnn.input_x: x_batch,
            self.cnn.input_y: y,
            self.cnn.input_index:iy,
            self.cnn.dropout_keep_prob: 1.0
        }
        step, loss= self.sess.run([self.global_step, self.cnn.m_loss], feed_dict)
        logging.info("the step={} of dev_step ,loss={}".format(step,loss))
        return loss

    def test_step(self,x_batch, y_batch):
        y=y_batch[:,0:-1]
        iy=y_batch[:,-1]
        feed_dict = {
            self.cnn.input_x: x_batch,
            self.cnn.input_y: y,
            self.cnn.input_index:iy,
            self.cnn.dropout_keep_prob: 1.0
        }
        step, loss = self.sess.run([self.global_step, self.cnn.m_loss], feed_dict)
        logging.info("the step={} of dev_step,loss={} ".format(step,loss))
        return loss



    def train(self, X_train, V, item_weight, seed):
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)#sequence of word_idx [[1,2,3],[2,3,4,6],...]
        np.random.seed(seed)
        # X_train = np.random.permutation(X_train)
        np.random.seed(seed)
        # V = np.random.permutation(V)
        np.random.seed(seed)
        # item_weight = np.random.permutation(item_weight)

        logging.info("Train...CNN module")
        # X_train=np.zeros([len(X_train),self.max_len])

        '''step 1: split data to test,dev,train'''
        X_trains,x_test,y_trains,y_test = train_test_split(X_train,V,test_size=0.1)
        x_train,x_dev, y_train,y_dev=train_test_split(X_trains,y_trains,test_size=0.1)

        logging.info('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))


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

            mloss=self.train_step(x_train_batch,y_train_batch)
            PP_train.append(mloss)


            current_step = tf.train.global_step(self.sess, self.global_step)

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
                    dev_loss = self.dev_step(x_dev_batch, y_dev_batch)
                    total_dev_correct=total_dev_correct+dev_loss
                    PP_DEV.append(dev_loss)

                dev_accuracy = float(total_dev_correct) / len(y_dev)
                logging.critical('Accuracy on dev set: {}'.format(dev_accuracy))


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
            mloss = self.test_step(x_test_batch, y_test_batch)

            total_test_loss=total_test_loss+mloss
            PP.append(mloss)

        logging.critical('loss on test set is {} based on the best model'.format(total_test_loss/len(y_test)))
        logging.critical('The processes is completed')

        self.train_loss,self.l2_p_loss,self.l1_p_loss=self.train_step(x_train,y_train,2)
        logging.info("every loss is {},final loss is {}".format( sum(PP_train)/len(PP_train),self.train_loss))
        logging.info("------------------end-------------------------")
        return self.final_out


    def get_projection_layer(self,X_train,V):

        # X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
        # Y = self.model.predict({'input': X_train}, batch_size=len(X_train))['output']
        logging.info("start predict and output result")
        X_train = sequence.pad_sequences(X_train,self.max_len)#sequence of word_idx [[1,2,3],[2,3,4,6],...]

        Y=np.array(list(V))
        y=Y[:,0:-1]
        iy=Y[:,-1]
        feed_dict = {
            self.cnn.input_x: X_train,
            self.cnn.input_y: y,
            self.cnn.input_index:iy,
            self.cnn.dropout_keep_prob: 1.0
            }
        out,echas,p,p1,p2= self.sess.run([self.cnn.scores,tf.nn.l2_loss(self.cnn.embedded_chars),tf.nn.l2_loss(self.cnn.PP),tf.nn.l2_loss(self.cnn.PP_1),tf.nn.l2_loss(self.cnn.PP_2)], feed_dict)
        print "Test init PP ",p
        print "Test init PP1 ",p1
        print "Test init PP2 ",p2
        print "Test init echas",echas


        self.final_out=np.array(out)
        self.final_out=self.final_out.reshape(y.shape[0],y.shape[1])
        logging.info("{}------------------end-------------------------{}".format(type(out),self.final_out.shape))
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
    def close_session(self):
        self.sess.close()



if __name__ == '__main__':
    # python3 train.py ./data/consumer_complaints.csv.zip ./parameters.json
    # CNN_module(filename,data_type)
    pass
