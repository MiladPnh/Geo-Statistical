import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from numpy import loadtxt
import numpy as np
from pylab import rcParams

# reading the data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
X_train = mnist.train.images
X_test = mnist.test.images

# creating the noise matrix
n_rows = X_test.shape[0]
n_cols = X_test.shape[1]
mean = 1
std = 0.35
noise = np.random.normal(mean, std, (n_rows, n_cols))
X_test_noisy = X_test + noise

# Deciding how many nodes each layer should have
n_nodes_inpl = 784  #encoder
n_nodes_hl1  = 32  #encoder
n_nodes_hl2  = 32  #decoder
n_nodes_outl = 784  #decoder
hidden_1_layer_vals = {
'weights':tf.Variable(tf.random_normal([n_nodes_inpl,n_nodes_hl1])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))  }
hidden_2_layer_vals = {
'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))  }
output_layer_vals = {
'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_outl])), 'biases':tf.Variable(tf.random_normal([n_nodes_outl])) }
input_layer = tf.placeholder('float', [None, 784])
layer_1 = tf.nn.sigmoid(
       tf.add(tf.matmul(input_layer,hidden_1_layer_vals['weights']), hidden_1_layer_vals['biases']))
layer_2 = tf.nn.sigmoid(
       tf.add(tf.matmul(layer_1,hidden_2_layer_vals['weights']), hidden_2_layer_vals['biases']))
output_layer = tf.matmul(layer_2,output_layer_vals['weights']) + output_layer_vals['biases']
output_true = tf.placeholder('float', [None, 784])

# define our cost function
meansq =    tf.reduce_mean(tf.square(output_layer - output_true))
# define our optimizer
learn_rate = 0.1   # how fast the model should learn
optimizer = tf.train.AdagradOptimizer(learn_rate).minimize(meansq)


# initialising
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 300
hm_epochs =100
tot_images = X_train.shape[0]

for epoch in range(hm_epochs):
    epoch_loss = 0
    for i in range(int(tot_images/batch_size)):
        epoch_x = X_train[ i*batch_size : (i+1)*batch_size ]
        _, c = sess.run([optimizer, meansq],\
               feed_dict={input_layer: epoch_x, \
               output_true: epoch_x})
        epoch_loss += c
    print('Epoch', epoch, '/', hm_epochs, 'loss:',epoch_loss)


#Validation
# any_image = X_test_noisy[100]
# output_any_image = sess.run(output_layer,\
#                    feed_dict={input_layer:[any_image]})


