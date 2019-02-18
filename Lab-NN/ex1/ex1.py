import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import sys
import argparse

# global flags for convenience
FLAGS=None

# Parameters
NUM_PIXELS = 784
NUM_CLASSES = 10
BATCH_SIZE = 1000
TRAIN_STEPS = 1000

NUM_H_1 = 100
NUM_H_2 = 50


def train_and_test(_):

    # Check if log_dir exists, if so delete contents
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

#################################################################################
############################    YOUR CODE HERE   ################################


    # define placeholders for batch of training images and labels
    x = tf.placeholder(tf.float32, shape=(None, NUM_PIXELS), name="input_images") # X is the training Data standing for the Images vectors a dimension 784 = 28x28 pixels
    y = tf.placeholder(tf.float32, shape=(None, NUM_CLASSES), name="input_labels")# y is the True labels of the images , dimension 10 which refer to the digits 

    # define variables for weights and biases of the three fully connected layers
    W1 = tf.Variable(tf.truncated_normal(shape=(NUM_PIXELS, NUM_H_1), stddev=0.1), name='trainable_weights1')
    b1 = tf.Variable(tf.constant(0.1, shape=(NUM_H_1,)), name='bias1')
    W2 = tf.Variable(tf.truncated_normal(shape=(NUM_H_1, NUM_H_2), stddev=0.1), name='trainable_weights2')
    b2 = tf.Variable(tf.constant(0.1, shape=(NUM_H_2,)), name='bias2')
    W3 = tf.Variable(tf.truncated_normal(shape=(NUM_H_2, NUM_CLASSES), stddev=0.1), name='trainable_weights3')
    b3 = tf.Variable(tf.constant(0.1, shape=(NUM_CLASSES,)), name='bias3')    

    # computation graph
    h1 = tf.matmul(x, W1) + b1
    h1 = tf.nn.relu(h1) 
    h2 = tf.matmul(h1, W2) + b2
    h2 = tf.nn.relu(h2) 
    h3 = tf.matmul(h2, W3) + b3     

    # define loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=h3))

    # make the loss a "summary" to visualise it in tensorboard
    tf.summary.scalar('loss', loss)

    # define the optimizer and what is optimizing
    optimizer = tf.train.GradientDescentOptimizer(0.5) # 0.5 is learning rate 
    # tell the optimizer that we want we want to optimize the minimum function
    train_step = optimizer.minimize(loss)

    # measure accuracy on the batch and make it a summary for tensorboard
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(h3, 1))
    # we can get the position of entry of maximum
    # , we check if both positions are right or not 
    # vector of true and false values 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
    tf.summary.scalar('accuracy', accuracy)

    # create session
    sess = tf.InteractiveSession()

    # merge summaries for tensorboard
    merged = tf.summary.merge_all() # becuase we have many summaries we merge them all in one node
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)

    # initialize variables
    tf.global_variables_initializer().run()
    # training iterations: fetch training batch and run
    # we create a loop of train set iterations 
    # 60K images in training set , so batch size of 100 is fine so choosing 1000 iteration 
    # makes it efficient to check all the examples atleast once 
    # training iterations: fetch training batch and run
    for i in range(1, TRAIN_STEPS):
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        # Specify 2 nodes  , merge and train steps . Feed dictionary is what is used to provide inputs
        # the kyvalue pairs . we feed batch axis to placeholder x and same geos for y 
        # we only care about the summary train value
        summary_train, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys})
        train_writer.add_summary(summary_train, i)

    # after training fetch test set and measure accuracy and print it to screen
    batch_xs, batch_ys = mnist.test.next_batch(10000)
    batch_Txs, batch_Tys = mnist.train.next_batch(100)
    train_accuracy = sess.run(accuracy,feed_dict={x:batch_Txs,y:batch_Tys})
    test_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
    print('Train accuracy: %.4f'%train_accuracy)
    print('Test accuracy: %.4f' % test_accuracy)

###################################################################################        
if __name__ == '__main__':

    # use nice argparse module to aprte cli arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data_dir', help='Directory for training data')
    parser.add_argument('--log_dir', type=str, default='./log_dir', help='Directory for Tensorboard event files')
    FLAGS, unparsed = parser.parse_known_args()
    # app.run is a simple wrapper that parses flags and sends them to main function
    tf.app.run(main=train_and_test, argv=[sys.argv[0]] + unparsed)
