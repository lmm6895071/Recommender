#-*- coding:utf-8 -*-

import re
import os
import sys
import numpy as np
from numpy import *
import gensim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn import preprocessing
from tensorflow.contrib import learn
dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))

import logging
logging.getLogger().setLevel(logging.INFO)

class LoadData(object):
    def clean_str(self,s):
        """Clean sentence"""
        s = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", s)
        s = re.sub(r"\'s", " \'s", s)
        s = re.sub(r"\'ve", " \'ve", s)
        s = re.sub(r"n\'t", " n\'t", s)
        s = re.sub(r"\'re", " \'re", s)
        s = re.sub(r"\'d", " \'d", s)
        s = re.sub(r"\'ll", " \'ll", s)
        s = re.sub(r",", " , ", s)
        s = re.sub(r"!", " ! ", s)
        s = re.sub(r"\(", " \( ", s)
        s = re.sub(r"\)", " \) ", s)
        s = re.sub(r"\?", " \? ", s)
        s = re.sub(r"\s{2,}", " ", s)
        s = re.sub(r'\S*(x{2,}|X{2,})\S*',"xxx", s)
        s = re.sub(r'[^\x00-\x7F]+', "", s)
        return s.strip().lower()
    """docstring for LoadData"""
    def __init__(self,filename,data_type,count=1500):
        super(LoadData, self).__init__()
        self.fname=filename
        if filename == 'ch_waimai2_corpus.txt':
            count=4000
        self.pos_count=count
        self.neg_count=count
        self.nclasses=2
        self.vocab_size=0
        self.vocab_processor=0
        self.data_type=data_type
        # self.enc = preprocessing.OneHotEncoder()

    def getWordVector(self,word,size):
        vec = np.zeros(size).reshape(1,size)
        try:
            vec = self.model[word]
        except Exception as err:
            vec=  2* np.random.random_sample((size)) - 1  #[-1,1]
        return vec 
    def buildWordVector(self,words,size=400):
        vec = np.zeros(size).reshape((1,size))
        count = 0
        for word in words:
            try:
                vec += self.model[word].reshape((1,size))
                count += 1
            except KeyError:
                continue
        if count != 0:
            vec /= count
        return vec
    def word2Vec_raw(self,x_raw,size=400):
        # self.model = gensim.models.Word2Vec.load(dirname+"/../word2vec/wiki.ch.text.model")
        self.model = gensim.models.Word2Vec.load(dirname+"/../../../word2vec/model_hotel_waimai_wiki/wiki.ch.text.model")
        self.max_document_length = max([len(x.split('\t')) for x in x_raw])
        logging.info('The maximum length of all sentences: {}'.format((self.max_document_length)))
        results=[]
        for ws in x_raw:
            ws=ws.split("\t")
            X=[]
            index_temp=0
            for w in ws:
                # w=self.clean_str(w)
                # if w== ' ' or len(w)<2:
                #     continue
                X.append(self.getWordVector(w,size))
                index_temp=index_temp+1
            logging.info("index_stemp={},max={},len(ws)={}".format(index_temp,self.max_document_length,len(ws)))
            for idx in range(self.max_document_length-len(ws)):
                X.append( 2* np.random.random_sample((size)) - 1)
                index_temp=index_temp+1
            logging.info("this is padding  process {}".format(index_temp))
            logging.info("this X type is {},results type is {}\n".format(len(X),len(results)))
            results.append(X)
        results = np.array(results)

        # results= scale(results)
        logging.info("the results like that {}".format(results.shape))
        # results=results.reshape(-1,self.max_document_length*size)
        # logging.info("the results like that {}".format(results.shape))

        return results
    def init_wordvector(self,x_raw,size=400):
        self.model = gensim.models.Word2Vec.load(dirname+"/../../../word2vec/model_hotel_waimai_wiki/wiki.ch.text.model")
        self.max_document_length = max([len(x.split('\t')) for x in x_raw])
        logging.info('The maximum length of all sentences: {}'.format((self.max_document_length)))

        result=np.zeros(((self.vocab_size)+1,size))
        count=0
        for word, i in self.vocab_processor.vocabulary_._mapping.items():
            try:
                result[i]=self.model[word]#self.model.getWordvector(word)
                count=count+1
            except Exception as e:
                result[i]=2* np.random.random_sample((size)) - 1  #[-1,1]np.random.normal(mean, 0.1, size)
        logging.info("the number of pretraing word2vector is {}".format(count))
        self.wordVocab=result

    def bag_raw(self,x_raw):
        
        """Step 1: pad each sentence to the same length and map each word to an id"""
        max_document_length = max([len(x.split('\t')) for x in x_raw])
        logging.info('The maximum length of all sentences: {}'.format((max_document_length)))
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        X = np.array(list(vocab_processor.fit_transform(x_raw)))
        self.vocab_size = len(vocab_processor.vocabulary_)
        self.vocab_processor= vocab_processor
        logging.info("the total vocabulary is %d",(self.vocab_size))
        return X

    def train_data(self):
        path = os.path.join(dirname,"../testData")
        #fname = "ch_waimai2_corpus.txt"
        # fname = "ch_hotel_corpus.txt"
        infile = open(path + "/"+self.fname)
        posD=[]
        negD=[]
        rdata1=[]
        rdata2=[]
        for line in infile.readlines():
            if len(line)<4:
                pass
            if line[0:3] == "neg":
                negD.append(line[4:].strip().strip("\n"))
            else:
                posD.append(line[4:].strip().strip("\n"))

        logging.info("pos counts:%d"%len(posD))
        logging.info("neg counts:%d"%len(negD))
        if len(posD)<self.pos_count or len(negD)<self.neg_count:
        	self.pos_count=len(posD)
        	self.neg_count=len(negD)


        if self.pos_count > 0 :
            shuffleArray = range(len(posD))
            np.random.shuffle(shuffleArray)
            for ii in xrange(self.pos_count):
                rdata1.append(posD[shuffleArray[ii]])
        else:
            rdata1 = posD

        if self.neg_count > 0:
            shuffleArray = range(len(negD))
            np.random.shuffle(shuffleArray)
            for ii in xrange(self.neg_count):
                rdata2.append(negD[shuffleArray[ii]])
        else:
            rdata2 = negD

        y = np.concatenate((np.ones(len(rdata1)), np.zeros(len(rdata2))))
        X = np.concatenate((rdata1, rdata2))
        X_vec = []
        if self.data_type == 'random':
            XX = self.bag_raw(X)
            self.init_wordvector(X,256)
            for item in XX:
                X_vec.append(tuple(item.tolist()))
        elif self.data_type=='word2vector':
            X_vec = self.word2Vec_raw(X,256)

        self.pos = X_vec[0:len(rdata1)]
        self.neg = X_vec[0:len(rdata2)]
        X_train,X_test,y_train,y_test = train_test_split(X_vec,y,test_size=0.2)


        X_train=np.array(X_train)

        # y_train=np.array(y_train)
        # y_train=y_train.reshape(len(y_train),1)

        y_train=self.trans_ont_hot(y_train)

        X_test=np.array(X_test)

        # y_test=np.array(y_test)
        # y_test=y_test.reshape(len(y_test),1)
        y_test=self.trans_ont_hot(y_test)

        self.X_test=X_test
        self.y_test=y_test

        logging.info("train shape is {}".format(X_train.shape))
        return  (X_train,y_train)
    def trans_ont_hot(self,ls,c=2):
        result=np.zeros((len(ls),c))
        for index in range(len(ls)):
            result[index][ls[index]]=1
        return result
    def test_data(self):
        return (self.X_test,self.y_test)
    def next_batch(self,count=50):
    	logging.info("next batch ---#---pos=%d,neg=%d"%(self.pos_count,self.neg_count))
        shuffleArray = range(self.pos_count)
        np.random.shuffle(shuffleArray)

        pos=[]
        neg=[]
        for ii in xrange(count/2):
            pos.append(self.pos[shuffleArray[ii]])
        shuffleArray = range(self.neg_count)
        np.random.shuffle(shuffleArray)
        for ii in xrange(count/2):
            neg.append(self.neg[shuffleArray[ii]])

        y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))
        X = np.concatenate((pos, neg))
        X=np.array(X)

        # y=np.array(y)
        # y=y.reshape(len(y),1)
        y=self.trans_ont_hot(y,self.nclasses)

        return  (X,y)
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


