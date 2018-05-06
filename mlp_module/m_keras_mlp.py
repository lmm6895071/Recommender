'''
Created on Sep 14, 2017
@author: ming
'''

import numpy as np

import theano
import theano.tensor as T
import keras
from keras import backend as K
from keras import initializations
from keras.regularizers import l2, activity_l2
from keras.models import Sequential, Graph, Model
from keras.layers.core import Dense, Lambda, Activation,Reshape,Flatten,Dropout,Merge
from keras.layers import Embedding, Input ,merge 
from keras.constraints import maxnorm
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
# from evaluate import evaluate_model
from time import time
import sys
import argparse
import multiprocessing as mp
import logging

class  Keras_MLP_Moudle():
    """docstring for  Keras_MLP_Moudle"""
    def __init__(self,num_items,num_users,epochs=100,batch_size=256,layers=[64,32,16,8],reg_layers=[0,0,0,0],lr=0.001,learner='adam',verbose=1):
        self.epochs=epochs
        self.batch_size=batch_size
        self.layers=layers
        self.reg_layers=reg_layers
        self.lr=lr
        self.learner=learner
        self.verbose=verbose
        self.num_items=num_items
        self.num_users=num_users
        self.model=self.get_model(self.layers,self.reg_layers)
         # Build model
        # if learner.lower() == "adagrad": 
        #     self.model.compile(optimizer=Adagrad(lr=lr), loss='binary_crossentropy')
        # elif learner.lower() == "rmsprop":
        #     self.model.compile(optimizer=RMSprop(lr=lr), loss='binary_crossentropy')
        # elif learner.lower() == "adam":
        #     self.model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy')
        # else:
        self.model.compile(optimizer=SGD(lr=lr), loss='sigmoid_cross_entropy_with_logits')    
        

    def init_normal(self,shape, name=None):
        return initializations.normal(shape, scale=0.01, name=name)

    def get_model(self, layers = [20,10], reg_layers=[0,0]):
        assert len(layers) == len(reg_layers)
        num_layer = len(layers) #Number of layers in the MLP
        # Input variables
        user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
        item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

        MLP_Embedding_User = Embedding(input_dim = self.num_users, output_dim = layers[0]/2, name = 'user_embedding',
                                      init = self.init_normal, W_regularizer = l2(reg_layers[0]), input_length=1)
        MLP_Embedding_Item = Embedding(input_dim = self.num_items, output_dim = layers[0]/2, name = 'item_embedding',
                                      init = self.init_normal, W_regularizer = l2(reg_layers[0]), input_length=1)   
        
        # Crucial to flatten an embedding vector!
        user_latent = Flatten()(MLP_Embedding_User(user_input))
        item_latent = Flatten()(MLP_Embedding_Item(item_input))
        
        # The 0-th layer is the concatenation of embedding layers
        vector = merge([user_latent, item_latent], mode = 'concat')
        
        # MLP layers
        for idx in xrange(1, num_layer):
            layer = Dense(layers[idx], W_regularizer= l2(reg_layers[idx]), activation='relu', name = 'layer%d' %idx)
            vector = layer(vector)
            
        # Final prediction layer
        prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name = 'prediction')(vector)
        
        model = Model(input=[user_input, item_input], 
                      output=prediction)
        
        return model

    def train(self,user_input,item_input,ratings):
        loss=0
        for epoch in xrange(self.epochs):
            t1 = time()
            # Training        
            hist = model.fit([ user_input, item_input ], #input
                              ratings , # labels 
                             batch_size=self.batch_size, nb_epoch=1, verbose=0, shuffle=True)
            t2 = time()
            # Evaluation
            if epoch %self.verbose == 0:
                # (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
                loss =  hist.history['loss'][0]
                print('Iteration %d [%.1f s]: loss = %.4f [%.1f s]' 
                      % (epoch,  t2-t1,  loss, time()-t2))
        return loss

    def predict(self,test_user,test_item,tets_ratings):
        hist=self.model.predict([test_user,test_item],ratings)
        print hist.keys()
        print hist.history['loss']
        return hist
        

if __name__ == '__main__':
    pass