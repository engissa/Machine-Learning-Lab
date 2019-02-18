import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import initializers

import sys
import argparse

# global flags for convenience
FLAGS=None

# Parameters
NUM_PIXELS = 28
NUM_CLASSES = 10
BATCH_SIZE = 200
TRAIN_STEPS = 1000
NUM_FILTERS_1 = 3
NUM_FILTERS_2 = 3


def classifier_model(): # linear stack of 
    
################################################################################
############################    YOUR CODE HERE   ################################

    # Define a Sequential model
    model = models.Sequential()
    # The first two layers are convolutional layers. For the first layer, we must specify the input shape.
    model.add(layers.Conv2D(NUM_FILTERS_1, 3, strides=(2,2), activation='relu', padding='same', input_shape=(28,28,1)))  # we have to add 2d convolutional layer 
    # a conv layer is defined by the number of filters 
    # second dimension we have to choose a stride of 2 : amount of shift that we are using for shift 
    # papdding same = output is cropped , output should be 28*28 
    # specify the kind of activate : RELU
    # first layer of input dimension 
    # we initialize the biases to 0 
    # we set kernel initializer 

    model.add(layers.Conv2D(NUM_FILTERS_2, 3, strides=(2,2), activation='relu', padding='same')) 
    # also a convolutional layer 
    # 3x3 filters , 2x2 strides 
    # we don't specify the input shape
    # The final layer is a dense, 1 dimensional layer. We must therefore first flatten the result of the
    # previous layer
    model.add(layers.Flatten()) # we want to reduce to 1 dimension to be able to have a dense layer of 1 dim
    # convert 2-3 layer dimension into a vextor 
    model.add(layers.Dense(10))
    return model
################################################################################



def train_and_test(_):

    # Check if log_dir exists, if so delete contents
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, reshape=False)

#################################################################################
############################    YOUR CODE HERE   ################################


    # define placeholders for batch of training images and labels
    x = tf.placeholder(tf.float32, shape=(None, NUM_PIXELS, NUM_PIXELS, 1), name='input_images')
    y = tf.placeholder(tf.float32, shape=(None, NUM_CLASSES), name='input_labels')# is a matrix of batch size by number of classes 
    tf.summary.image('input_images',x)
    # create model
    my_net = classifier_model() # we define the model in my_net we have to create a sequential model
    
    print (my_net.summary())
    # we use the model as funciton , we apply x to the model to get h
    # use model on input image batch to compute logits
    h = my_net(x) # we feed our model by x to get predicted value

    # define loss function softmax , performs both softmax and we apply reduce mean 
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=h))

    # make the loss a "summary" to visualise it in tensorboard
    tf.summary.scalar('loss', loss)

    # define the optimizer and what is optimizing
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train_step = optimizer.minimize(loss)

    # measure accuracy on the batch and make it a summary for tensorboard
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(h,1)) # compare y with h , it gives a boolean vector

    # we use the cast to convert from True / False to number to be able to reduce the mean 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) # average number of predictions

    tf.summary.scalar('accuracy', accuracy)
    
    # create session
    sess = tf.InteractiveSession()

    # merge summaries for tensorboard
    merged = tf.summary.merge_all()
    
    train_writer = tf.summary.FileWriter(FLAGS.log_dir+'/train', sess.graph)

    # initialize variables
    tf.global_variables_initializer().run() # tell Keras to run initializers that we defined in the layers
    
    # training iterations: fetch training batch and run
    for i in range(TRAIN_STEPS):
        # we fetch a batch , we read a set of images from the disk and places them in the tensor
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        summar_train, _ = sess.run([merged,train_step],feed_dict={x: batch_xs, y: batch_ys})
        train_writer.add_summary(summar_train, i)

    test_xs, test_ys = mnist.test.next_batch(BATCH_SIZE) # we choose next batch of Test images 
    test_Txs, test_Tys = mnist.train.next_batch(BATCH_SIZE) # we choose next batch of Test images 

    # after training fetch test set and measure accuracy and print it to screen
    accuracy_value = sess.run(accuracy, feed_dict={x: test_xs, y: test_ys})
    accuracy_Tvalue = sess.run(accuracy, feed_dict={x: test_Txs, y: test_Tys})

    # tf.summary.scalar('accuracy1', accuracy_value)
    # train_writer.add_summary(summar_train)
    print (accuracy_value)
    print (accuracy_Tvalue)


###################################################################################


if __name__ == '__main__':

    # use nice argparse module to aprte cli arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data_dir', help='Directory for training data')
    parser.add_argument('--log_dir', type=str, default='./log_dir', help='Directory for Tensorboard event files')
    FLAGS, unparsed = parser.parse_known_args()
    # app.run is a simple wrapper that parses flags and sends them to main function
    tf.app.run(main=train_and_test, argv=[sys.argv[0]] + unparsed)
