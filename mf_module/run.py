'''
Created on Dec 9, 2015

@author: donghyun
'''
import argparse
import sys
from data_manager import Data_Factory

parser = argparse.ArgumentParser()

# Option for pre-processing data
parser.add_argument("-c", "--do_preprocess", type=bool,
                    help="True or False to preprocess raw data for ConvMF (default = False)", default=False)
parser.add_argument("-r", "--raw_rating_data_path", type=str,
                    help="Path to raw rating data. data format - user id::item id::rating")

parser.add_argument("-m", "--min_rating", type=int,
                    help="Users who have less than \"min_rating\" ratings will be removed (default = 1)", default=1)
parser.add_argument("-t", "--split_ratio", type=float,
                    help="Ratio: 1-ratio, ratio/2 and ratio/2 of the entire dataset (R) will be training, valid and test set, respectively (default = 0.2)", default=0.2)

# Option for pre-processing data and running ConvMF
parser.add_argument("-d", "--data_path", type=str,
                    help="Path to training, valid and test data sets")
parser.add_argument("-a", "--aux_path", type=str, help="Path to R, D_all sets")

# Option for running ConvMF
parser.add_argument("-k", "--dimension", type=int,
                    help="Size of latent dimension for users and items (default: 50)", default=100)
parser.add_argument("-u", "--lambda_u", type=float,
                    help="Value of user regularizer")
parser.add_argument("-v", "--lambda_v", type=float,
                    help="Value of item regularizer")

parser.add_argument("-P", "--lambda_p", type=float,
                    help="Value of l2_loss regularizer")
parser.add_argument("-Q", "--lambda_q", type=float,
                    help="Value of l1_loss regularizer")

parser.add_argument("-n", "--max_iter", type=int,
                    help="Value of max iteration (default: 200)", default=500)


parser.add_argument("-F","--flag",type=str,help="class flag",default="PMF")
parser.add_argument("-G","--momentum_flag",type=int,help="momentum_flag",default=1)


args = parser.parse_args()
do_preprocess = args.do_preprocess
data_path = args.data_path
aux_path = args.aux_path
if data_path is None:
    sys.exit("Argument missing - data_path is required")
if aux_path is None:
    sys.exit("Argument missing - aux_path is required")


data_factory = Data_Factory()

if do_preprocess:
    path_rating = args.raw_rating_data_path
    min_rating = args.min_rating
    split_ratio = args.split_ratio

    print "=================================Preprocess Option Setting================================="
    print "\tsaving preprocessed aux path - %s" % aux_path
    print "\tsaving preprocessed data path - %s" % data_path
    print "\trating data path - %s" % path_rating
    print "\tmin_rating: %d\n\t split_ratio: %.1f"% (min_rating, split_ratio)
    print "==========================================================================================="
    

    R =data_factory.preprocess(path_rating, min_rating)
    data_factory.save(aux_path, R)
    data_factory.generate_train_valid_test_file_from_R(data_path, R, split_ratio)
else:
    methods = args.flag
    dimension = args.dimension
    lambda_u = args.lambda_u
    lambda_v = args.lambda_v
    lambda_p=args.lambda_p
    lambda_q=args.lambda_q

    max_iter = args.max_iter
    momentum_flag=args.momentum_flag

    if lambda_u is None:
        sys.exit("Argument missing - lambda_u is required")
    if lambda_v is None:
        sys.exit("Argument missing - lambda_v is required")

    print "===================================ConvMF Option Setting==================================="
    print "\t approach -%s"%methods
    print "\taux path - %s" % aux_path
    print "\tdata path - %s" % data_path
    print "\tdimension: %d\n\tlambda_u: %.4f\n\tlambda_v: %.4f\n\tmax_iter: %d\n\t" \
        % (dimension, lambda_u, lambda_v, max_iter)
    print "==========================================================================================="

    R = data_factory.load(aux_path)
    train_user = data_factory.read_rating(data_path + '/train_user.dat')
    train_item = data_factory.read_rating(data_path + '/train_item.dat')
    valid_user = data_factory.read_rating(data_path + '/valid_user.dat')
    test_user = data_factory.read_rating(data_path + '/test_user.dat')


    if methods=="PMF":
        from models.PMF import PMF
        PMF(max_iter=max_iter, lambda_u=lambda_u, lambda_v=lambda_v, dimension=dimension, 
            train_user=train_user, train_item=train_item, valid_user=valid_user, test_user=test_user, R=R)
    elif methods == "BiasMF":
        from models.BiasMF import BiasMF
        BiasMF(max_iter=max_iter, lambda_u=lambda_u, lambda_v=lambda_v, dimension=dimension, 
            train_user=train_user, train_item=train_item, valid_user=valid_user, test_user=test_user, R=R,momentum_flag=momentum_flag)
    elif methods == "BiasMF_Constant":
        from models.BiasMF_Constant import BiasMF_Constant
        BiasMF_Constant(max_iter=max_iter, lambda_u=lambda_u, lambda_v=lambda_v, dimension=dimension, 
            train_user=train_user, train_item=train_item, valid_user=valid_user, test_user=test_user, R=R)
    elif methods == "WNMF":
        from models.WNMF import WNMF
        WNMF(max_iter=max_iter, train_user=train_user, train_item=train_item,valid_user=valid_user,test_user=test_user,R=R,dimension=dimension)

    elif methods == "JMF-S":
        from models.JMF_S import JMF_S
        print "######### Test start lambda_u={},lambda_v={},lambda_p=-,lambda_q={}############".format(lambda_u,lambda_v,lambda_q)
        JMF_S(max_iter=max_iter, lambda_u=lambda_u, lambda_v=lambda_v, dimension=dimension, train_user=train_user, 
            train_item=train_item, valid_user=valid_user, test_user=test_user, R=R,lambda_p=lambda_p,lambda_q=lambda_q,momentum_flag=momentum_flag)

    elif methods == "JONMF-P":
        from models.JONMF_P import JONMF_P
        print "######### Test start lambda_u={},lambda_v={},lambda_p={},lambda_q={}############".format(lambda_u,lambda_v,lambda_p,lambda_q)
        JONMF_P(max_iter=max_iter,lambda_u=lambda_u, lambda_v=lambda_v, dimension=dimension, train_user=train_user,
                train_item=train_item, valid_user=valid_user, test_user=test_user, R=R,lambda_p=lambda_p,lambda_q=lambda_q)
    elif methods =="JMF-Double":
        from models.JMF_Double import JMF_Double
        JMF_Double(max_iter=max_iter,lambda_u=lambda_u, lambda_v=lambda_v, dimension=dimension, train_user=train_user, 
               train_item=train_item, valid_user=valid_user, test_user=test_user, R=R,lambda_p=lambda_p,lambda_q=lambda_q)        
    print "###### method {} end lambda_u={},lambda_v={},lambda_p={},lambda_q={} ########".format(methods,lambda_u,lambda_v,lambda_p,lambda_q)

   