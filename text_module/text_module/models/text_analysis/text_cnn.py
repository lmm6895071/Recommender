import numpy as np
import tensorflow as tf
import logging
import os
import sys

logging.getLogger().setLevel(logging.INFO)

dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))

class TextCNN(object):

	def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda,wordVocab,num_items,lambda_v,lambda_p,lambda_q=1):
		# Placeholders for input, output and dropout
		logging.info("the sequence_length is:%d",sequence_length)
		self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
		self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
		self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

		'''
		this is the index of items;
		item unique vector
		formula is: [PP_1*PP_2+diag(PP)]*e_i^T
		PP_1   d*r
		PP_2   r*d
		PP     d*d
		e_i    1*d
		'''
		r=3 #

		self.input_index=tf.placeholder(tf.int32,[None],name='index_of_item')

		# self.PP_1=tf.Variable(tf.random_normal([num_items,embedding_size,r], mean=0.0, stddev=0.01, dtype=tf.float32, seed=1234, name="PP_1"))#np.sqrt(2/embedding_size)
		# self.PP_2=tf.Variable(tf.random_normal([num_items,r,embedding_size], mean=0.0, stddev=0.01, dtype=tf.float32, seed=1235, name="PP_2"))#np.sqrt(2/embedding_size)
		# self.PP_1=tf.get_variable(
		# 		'PP_1',
		# 		shape=[num_items,embedding_size,r],
		# 		initializer=tf.contrib.layers.xavier_initializer())
		# self.PP_2=tf.get_variable(
		# 		'PP_2',
		# 		shape=[num_items,r,embedding_size],
		# 		initializer=tf.contrib.layers.xavier_initializer())
		# self.PP=tf.get_variable(
		# 		'PP',
		# 		shape=[embedding_size],
		# 		initializer=tf.contrib.layers.xavier_initializer())

		self.PP_1=tf.Variable(tf.truncated_normal([num_items,embedding_size,r],stddev=np.sqrt(1/embedding_size), dtype=tf.float32, ),name="PP_1")#np.sqrt(6/embedding_size)
		self.PP_2=tf.Variable(tf.truncated_normal([num_items,r,embedding_size],stddev=np.sqrt(1/embedding_size), dtype=tf.float32, ),name="PP_2")

		self.PP=tf.Variable(tf.truncated_normal([embedding_size], stddev=np.sqrt(1/embedding_size), dtype=tf.float32, seed=12322, name="diag_P"))#np.sqrt(2/embedding_size)

		# self.PP_1=tf.nn.l2_normalize(self.PP_1,1)
		# self.PP_2=tf.nn.l2_normalize(self.PP_2,1)


		# Keeping track of l2 regularization loss (optional)
		l2_loss = tf.constant(0.0)
		l2_p_loss = tf.constant(0.0)
		l1_p_loss = tf.constant(0.0)
		# Embedding layer
		with tf.name_scope('embedding'):#tf.device('/cpu:0'),

			pp1=tf.nn.embedding_lookup(self.PP_1,self.input_index)
			pp2=tf.nn.embedding_lookup(self.PP_2,self.input_index)
			# WS = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='WS')
			WS = tf.Variable(tf.to_float(wordVocab), name='WS')
			e_embedded_chars = tf.nn.embedding_lookup(WS, self.input_x)  #find vertor for word
			p = (tf.matmul(pp1,pp2)+tf.reshape(tf.tile(tf.diag(self.PP),(tf.shape(pp1)[0],1)),[-1,embedding_size,embedding_size]))

			self.embedded_chars =tf.nn.tanh(tf.matmul(e_embedded_chars,p))

			# self.embedded_chars =tf.matmul(e_embedded_chars,p)

			self.embedded_chars=tf.nn.l2_normalize(self.embedded_chars,1)
			logging.info("embedded_chars shape is {}".format(self.embedded_chars))
			# elif data_type=='word2vector':
			# 	self.input_x = tf.placeholder(tf.float32,[None,sequence_length,embedding_size],name='input_x')
			#   WS=self.get_wordVec(sequence_length,embedding_size)
			# 	self.embedded_chars =(self.input_x)#tf.Variable(self.input_x, name='WS')#this vocab_size is the vecotr of data
			self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

		logging.info("input X shape is {}".format(self.input_x))

		# Create a convolution + maxpool layer for each filter size
		logging.info(self.embedded_chars)
		logging.info("this embedded_chars_expanded size:{}".format(self.embedded_chars_expanded))
		logging.info((self.embedded_chars_expanded))

		pooled_outputs = []
		for i, filter_size in enumerate(filter_sizes):
			with tf.name_scope('conv-maxpool-%s' % filter_size):
				# Convolution Layer
				filter_shape = [filter_size, embedding_size, 1, num_filters] # kenrel 3*50,input=1,output=32|100
				logging.info(filter_shape)
				W = tf.Variable(tf.truncated_normal(filter_shape, stddev=np.sqrt(embedding_size)), name='W')
				b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
				logging.info("++++W++++:%s",str(tf.shape(W)))
				logging.info(W)
				logging.info("++++b++++:%s",str(tf.shape(b)))

				conv = tf.nn.conv2d(
					tf.reshape(self.embedded_chars_expanded,[-1,sequence_length,embedding_size,1]),
					# self.embedded_chars_expanded,
					W,
					strides=[1, 1, 1, 1],
					padding='VALID',
					name='conv')

				# Apply nonlinearity
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
				logging.info("pooling input shape:{}".format(conv))
				# Maxpooling over the outputs
				pooled = tf.nn.max_pool(
					h,
					ksize=[1, sequence_length - filter_size + 1, 1, 1],
					strides=[1, 1, 1, 1],
					padding='VALID',
					name='pool')
				pooled_outputs.append(pooled)

		# Combine all the pooled features
		logging.info("Combine all the pooled features")
		num_filters_total = num_filters * len(filter_sizes)
		self.h_pool = tf.concat(pooled_outputs,3)
		self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

		# Add dropout
		with tf.name_scope('dropout'):
			self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

		# Final (unnormalized) scores and predictions
		logging.info("-------------output layer------")
		with tf.name_scope('output'):
			W = tf.get_variable(
				'W',
				shape=[num_filters_total, num_classes],
				initializer=tf.contrib.layers.xavier_initializer())
			b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
			l2_loss += tf.nn.l2_loss(W)
			l2_loss += tf.nn.l2_loss(b)

			l2_p_loss +=tf.nn.l2_loss(self.PP_1)
			l2_p_loss+=tf.nn.l2_loss(self.PP_2)
			l2_p_loss+=tf.nn.l2_loss(self.PP)

			# l2_loss+=tf.reduce_sum(tf.abs(self.PP_1))
			# l2_loss+=tf.reduce_sum(tf.abs(self.PP_2))
			# l2_loss+=tf.reduce_sum(tf.abs(self.PP))

			l1_p_loss=tf.reduce_sum(tf.abs(self.PP_1))
			l1_p_loss+=tf.reduce_sum(tf.abs(self.PP_2))
			l1_p_loss+=tf.reduce_sum(tf.abs(self.PP))
			self.l1_p_loss=l1_p_loss
			self.l2_p_loss= l2_p_loss


			self.scores = tf.nn.relu(tf.nn.xw_plus_b(self.h_drop, W, b, name='scores'))
			self.predictions = tf.argmax(self.scores, 1, name='predictions')

		# Calculate mean cross-entropy loss
		with tf.name_scope('loss'):
			losses = tf.nn.softmax_cross_entropy_with_logits(labels = self.input_y, logits = self.scores) #  only named arguments accepted
			self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
		with tf.name_scope("m_loss"):
			# losses=tf.reduce_mean(tf.square(self.input_y-self.scores))
			losses=lambda_v*tf.nn.l2_loss(self.input_y-self.scores)
			# self.m_loss=losses   # +l2_reg_lambda*l2_loss
			self.m_loss=losses+l2_loss*lambda_p +l1_p_loss*lambda_q



		# Accuracy
		with tf.name_scope('accuracy'):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
		with tf.name_scope('num_correct'):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.num_correct = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='num_correct')
