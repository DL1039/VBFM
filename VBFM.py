import pandas as pd
import numpy as np
import scipy.io
import tensorflow as tf
import math

display_step = 10
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:,k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k*mini_batch_size:(k+1)*mini_batch_size]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:,(num_complete_minibatches*mini_batch_size)+1:]
        mini_batch_Y = shuffled_Y[:,(num_complete_minibatches*mini_batch_size)+1:]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


#Load NN Data
mat=scipy.io.loadmat(r'/home/vm/Python/NeuralNetwork/NNData.mat')
print(mat.keys())

TrainData=mat["TrainData"]
print(TrainData.shape)
Train_x=np.transpose(TrainData[:,0:5])
Train_y=np.transpose(TrainData[:,5:])

print(Train_x.shape)
print(Train_y.shape)
#mini_batch_size=64
mini_batches = random_mini_batches(Train_x, Train_y)

x = tf.placeholder(tf.float32, shape=[None, 5])
W1 = tf.get_variable("W1", shape=[5,10], initializer = tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable("b1", shape=[10], initializer =  tf.constant_initializer(0.01)) #tf.zeros_initializer())
W2 = tf.get_variable("W2", [10,1], initializer = tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable("b2", [1], initializer = tf.constant_initializer(0.01)) #tf.zeros_initializer())
parameters = {"W1": W1,"b1": b1,"W2": W2,"b2": b2}

#W = tf.get_variable("weights", shape=[5, 10],initializer=tf.contrib.layers.xavier_initializer()) #tf.truncated_normal_initializer(stddev=0.01) #tf.glorot_uniform_initializer()
#b = tf.get_variable("bias", shape=[10],initializer=tf.constant_initializer(0.01))#tf.zeros_initializer()

A1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
y = tf.nn.relu(tf.matmul(A1, W2) + b2)
y_ = tf.placeholder(tf.float32, [None, 1])


cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_))
#cross_entropy = tf.reduce_mean(tf.square(y_-y))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

# Evaluate model
accuracy=tf.losses.mean_squared_error(y_,y)

init = tf.global_variables_initializer()

#Create a saver object which will save all the variables
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for step in range(491):
     
        #  batch_xs= np.transpose(mini_batches[step][0])
        #  batch_ys= np.transpose(mini_batches[step][1])

        #  sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

        #  if step % display_step == 0 or step == 1:
        #     loss, acc = sess.run([cross_entropy, accuracy], feed_dict={x: batch_xs, y_:batch_ys})
        #     print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))
    
        sess.run(train_step,feed_dict={x:np.transpose(Train_x), y_:np.transpose(Train_y)})
        if step % display_step == 0 or step == 1:
            loss, acc = sess.run([cross_entropy, accuracy], feed_dict={x:np.transpose(Train_x), y_:np.transpose(Train_y)})
            print("Step " + str(step) + ", Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))
    
    saved_path = saver.save(sess, './my_model',global_step=491)    
    
print("Optimization Finished!")

#########################################################################

# delete the current graph
tf.reset_default_graph()

# import the graph from the file
imported_graph = tf.train.import_meta_graph('my_model-491.meta')

# list all the tensors in the graph
#for tensor in tf.get_default_graph().get_operations():
#    print (tensor.name)



# run the session
#with tf.Session() as sess:
    # restore the saved vairable
    #imported_graph.restore(sess, './my_model-491')
    # print the loaded variable
    #weight, bias = sess.run(['weights:0','bias:0'])
    #print('W = ', weight)
    #print('b = ', bias)


