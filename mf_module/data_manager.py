'''
Created on April 9, 2018

@author: ming
'''

import os
import sys
import cPickle as pickl
import numpy as np

from operator import itemgetter
from scipy.sparse.csr import csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import preprocessing
import random
import re
from tensorflow.contrib import learn



class Data_Factory():

    def load(self, path):
        R = pickl.load(open(path + "/ratings.all", "rb"))
        print "Load preprocessed rating data - %s" % (path + "/ratings.all")
        return R
    def save(self, path,R):
        if not os.path.exists(path):
            os.makedirs(path)
        print "Saving preprocessed rating data - %s" % (path + "/ratings.all")
        pickl.dump(R, open(path + "/ratings.all", "wb"))
        print "Done!"

    def read_rating(self, path):
        results = []
        if os.path.isfile(path):
            raw_ratings = open(path, 'r')
        else:
            print "Path (preprocessed) is wrong!"
            sys.exit()
        index_list = []
        rating_list = []
        all_line = raw_ratings.read().splitlines()
        #data format is:  len i:r i:r

        for line in all_line:
            tmp = line.split()
            num_rating = int(tmp[0])
            if num_rating > 0:
                tmp_i, tmp_r = zip(*(elem.split(":") for elem in tmp[1::]))
                index_list.append(np.array(tmp_i, dtype=int))
                rating_list.append(np.array(tmp_r, dtype=float))
            else:
                index_list.append(np.array([], dtype=int))
                rating_list.append(np.array([], dtype=float))

        results.append(index_list)
        results.append(rating_list)

        return results


    def split_data(self, ratio, R):
        print "Randomly splitting rating data into training set (%.1f) and test set (%.1f)..." % (1 - ratio, ratio)
        train = []
        for i in xrange(R.shape[0]):            #R.shape[0]  the number of rows(also the number of users)
            user_rating = R[i].nonzero()[1]     #R[i].nonzero() is a list [user_idex,item_index];eg: [[0,0],[0,4]]
            np.random.shuffle(user_rating)      #user_rating is the i-th user click item's list
            train.append((i, user_rating[0]))   #tuple (i-th user_index,item_index); every user have a item;
        #print train
        remain_item = set(xrange(R.shape[1])) - set(zip(*train)[1])# zip(*train)[1] unzip train

        for j in remain_item:
            item_rating = R.tocsc().T[j].nonzero()[1]         #R.tocsc() according to column;
            np.random.shuffle(item_rating)
            train.append((item_rating[0], j))

        rating_list = set(zip(R.nonzero()[0], R.nonzero()[1]))
        total_size = len(rating_list)
        remain_rating_list = list(rating_list - set(train))
        random.shuffle(remain_rating_list)

        num_addition = int((1 - ratio) * total_size) - len(train)
        if num_addition < 0:
            print 'this ratio cannot be handled'
            sys.exit()
        else:
            train.extend(remain_rating_list[:num_addition])
            tmp_test = remain_rating_list[num_addition:]
            random.shuffle(tmp_test)
            valid = tmp_test[::2]
            test = tmp_test[1::2]

            trainset_u_idx, trainset_i_idx = zip(*train)
            trainset_u_idx = set(trainset_u_idx)
            trainset_i_idx = set(trainset_i_idx)
            if len(trainset_u_idx) != R.shape[0] or len(trainset_i_idx) != R.shape[1]:
                print "Fatal error in split function. Check your data again or contact authors"
                sys.exit()

        print "Finish constructing training set and test set"
        return train, valid, test

    def generate_train_valid_test_file_from_R(self, path, R, ratio):
        '''
        Split randomly rating matrix into training set, valid set and test set with given ratio (valid+test)
        and save three data sets to given path.
        Note that the training set contains at least a rating on every user and item.

        Input:
        - path: path to save training set, valid set, test set
        - R: rating matrix (csr_matrix)
        - ratio: (1-ratio), ratio/2 and ratio/2 of the entire dataset (R) will be training, valid and test set, respectively
        '''
        train, valid, test = self.split_data(ratio, R)
        print "Save training set and test set to %s..." % path
        if not os.path.exists(path):
            os.makedirs(path)

        R_lil = R.tolil() #R.todense()
        user_ratings_train = {}
        item_ratings_train = {}
        for i, j in train:
            if user_ratings_train.has_key(i):
                user_ratings_train[i].append(j)
            else:
                user_ratings_train[i] = [j]

            if item_ratings_train.has_key(j):
                item_ratings_train[j].append(i)
            else:
                item_ratings_train[j] = [i]

        user_ratings_valid = {}
        item_ratings_valid = {}
        for i, j in valid:
            if user_ratings_valid.has_key(i):
                user_ratings_valid[i].append(j)
            else:
                user_ratings_valid[i] = [j]

            if item_ratings_valid.has_key(j):
                item_ratings_valid[j].append(i)
            else:
                item_ratings_valid[j] = [i]

        user_ratings_test = {}
        item_ratings_test = {}
        for i, j in test:
            if user_ratings_test.has_key(i):
                user_ratings_test[i].append(j)
            else:
                user_ratings_test[i] = [j]

            if item_ratings_test.has_key(j):
                item_ratings_test[j].append(i)
            else:
                item_ratings_test[j] = [i]

        f_train_user = open(path + "/train_user.dat", "w")
        f_valid_user = open(path + "/valid_user.dat", "w")
        f_test_user = open(path + "/test_user.dat", "w")

        formatted_user_train = []
        formatted_user_valid = []
        formatted_user_test = []

        for i in xrange(R.shape[0]):
            if user_ratings_train.has_key(i):
                formatted = [str(len(user_ratings_train[i]))]#user's click counts of the item;
                formatted.extend(["%d:%.1f" % (j, R_lil[i, j])
                                  for j in sorted(user_ratings_train[i])])
                formatted_user_train.append(" ".join(formatted))#formatted_user_train format:  counts item_index:r1 item_index:r2 item_index:r3 ...
            else:
                formatted_user_train.append("0")

            if user_ratings_valid.has_key(i):
                formatted = [str(len(user_ratings_valid[i]))]
                formatted.extend(["%d:%.1f" % (j, R_lil[i, j])
                                  for j in sorted(user_ratings_valid[i])])
                formatted_user_valid.append(" ".join(formatted))
            else:
                formatted_user_valid.append("0")

            if user_ratings_test.has_key(i):
                formatted = [str(len(user_ratings_test[i]))]
                formatted.extend(["%d:%.1f" % (j, R_lil[i, j])
                                  for j in sorted(user_ratings_test[i])])
                formatted_user_test.append(" ".join(formatted))
            else:
                formatted_user_test.append("0")

        f_train_user.write("\n".join(formatted_user_train))
        f_valid_user.write("\n".join(formatted_user_valid))
        f_test_user.write("\n".join(formatted_user_test))

        f_train_user.close()
        f_valid_user.close()
        f_test_user.close()
        print "\ttrain_user.dat, valid_user.dat, test_user.dat files are generated."
        print "\torder by user_index, data format:  len(item) item1:rate1 item2:rate2 ...."
        f_train_item = open(path + "/train_item.dat", "w")
        f_valid_item = open(path + "/valid_item.dat", "w")
        f_test_item = open(path + "/test_item.dat", "w")

        formatted_item_train = []
        formatted_item_valid = []
        formatted_item_test = []

        for j in xrange(R.shape[1]):
            if item_ratings_train.has_key(j):
                formatted = [str(len(item_ratings_train[j]))]
                formatted.extend(["%d:%.1f" % (i, R_lil[i, j])
                                  for i in sorted(item_ratings_train[j])])
                formatted_item_train.append(" ".join(formatted))
            else:
                formatted_item_train.append("0")

            if item_ratings_valid.has_key(j):
                formatted = [str(len(item_ratings_valid[j]))]
                formatted.extend(["%d:%.1f" % (i, R_lil[i, j])
                                  for i in sorted(item_ratings_valid[j])])
                formatted_item_valid.append(" ".join(formatted))
            else:
                formatted_item_valid.append("0")

            if item_ratings_test.has_key(j):
                formatted = [str(len(item_ratings_test[j]))]
                formatted.extend(["%d:%.1f" % (i, R_lil[i, j])
                                  for i in sorted(item_ratings_test[j])])
                formatted_item_test.append(" ".join(formatted))
            else:
                formatted_item_test.append("0")

        f_train_item.write("\n".join(formatted_item_train))
        f_valid_item.write("\n".join(formatted_item_valid))
        f_test_item.write("\n".join(formatted_item_test))

        f_train_item.close()
        f_valid_item.close()
        f_test_item.close()
        print "\ttrain_item.dat, valid_item.dat, test_item.dat files are generated."
        print "\torder by item_index, data format:  len(user) user1:rate1 user2:rate2 ...."
        print "Done!"

    def preprocess(self, path_rating,min_rating):
        '''
        Preprocess rating and document data.

        Input:
            - path_rating: path for rating data (data format - user_id::item_id::rating)
            - min_rating: users who have less than "min_rating" ratings will be removed (default = 1)

        Output:
            - R: rating matrix (csr_matrix: row - user, column - item)
        '''
        # Validate data paths
        if os.path.isfile(path_rating):
            raw_ratings = open(path_rating, 'r')
            print "Path - rating data: %s" % path_rating
        else:
            print "Path(rating) is wrong!"
            sys.exit()



        print "Preprocessing rating data..."
        print "\tCounting # ratings of each user and removing users having less than %d ratings..." % min_rating
        # 1st scan rating file to check # ratings of each user
        all_line = raw_ratings.read().splitlines()
        tmp_user = {}  #user rating counts;
        for line in all_line:
            tmp = line.split('::')
            u = tmp[0]
            i = tmp[1]
            # if (i in tmp_id_plot):
            if (u not in tmp_user):
                tmp_user[u] = 1
            else:
                tmp_user[u] = tmp_user[u] + 1

        raw_ratings.close()

        # 2nd scan rating file to make matrix indices of users and items
        # with removing users and items which are not satisfied with the given
        # condition
        raw_ratings = open(path_rating, 'r')
        all_line = raw_ratings.read().splitlines()
        userset = {}
        itemset = {}
        user_idx = 0
        item_idx = 0

        user = []#id_index;0,1,2,...
        item = []#item_index;0,1,2,3,...
        rating = []#float ratings

        for line in all_line:
            tmp = line.split('::')
            u = tmp[0]
            if u not in tmp_user:#temp_user who  ratings;
                continue
            i = tmp[1]
            # An user will be skipped where the number of ratings of the user
            # is less than min_rating.
            if tmp_user[u] >= min_rating:
                if u not in userset:
                    userset[u] = user_idx
                    user_idx = user_idx + 1

                if (i not in itemset):# and (i in tmp_id_plot):#tmp_id_plot item who have review info
                    itemset[i] = item_idx
                    item_idx = item_idx + 1
            else:
                continue

            if u in userset and i in itemset:
                u_idx = userset[u]
                i_idx = itemset[i]

                user.append(u_idx)
                item.append(i_idx)
                rating.append(float(tmp[2]))

        raw_ratings.close()

        R = csr_matrix((rating, (user, item)))
        #csr_matrix according rows;


        # mingfile =open("userIDS.txt","w")
        # for i in userset.keys():
        #     mingfile.write(str(i)+"\n")
        # mingfile.close()

        # mingfile=open("movieIDS.txt","w")
        # for i in itemset.keys():
        #     mingfile.write(str(i)+"\n")
        # mingfile.close()


        print "Finish preprocessing rating data - # user: %d, # item: %d, # ratings: %d" % (R.shape[0], R.shape[1], R.nnz)

        return R

if __name__ == '__main__':
    data= Data_Factory()
    u=[0,0,1,1,2]
    v=[0,1,2,3,0]
    dt=[3,3,2,4,5]
    R=csr_matrix((dt,(u,v)))

    print R[0]
    print "aa", R[1]
    S=csr_matrix((dt,(v,u)))
    print R, S

    print R[1]
    print S[1]

    # print R.shape
    # print R.tolil()
    # print R[0].nonzero()
    # train = []
    # for i in xrange(R.shape[0]):            #R.shape[0]  the number of rows(also the number of users)
    #     user_rating = R[i].nonzero()[1]     #R[i].nonzero() is a list [user_idex,item_index];eg: [[0,0],[0,4]]
    #     np.random.shuffle(user_rating)      #user_rating is the i-th user click item's list
    #     train.append((i, user_rating[0]))   #tuple (i-th user_index,item_index); every user have a item;
    # print train
    # remain_item = set(xrange(R.shape[1])) - set(zip(*train)[1])# zip(*train)[1] unzip train
    # print zip(*train),";",zip(*train)[1]
    # print remain_item
