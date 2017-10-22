#!/usr/bin/python

# https://www.tensorflow.org/tutorials/layers
# A Guide to TF Layers: Building a Convolutional Neural Network 

# source ~/tensorflow/bin/activate
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import sys

import cv2


tf.logging.set_verbosity(tf.logging.INFO)
pSize = 128

def my_hom_cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]

    #HOM Images are 128x128, and have two channels
    input_layer = tf.reshape(features["x"], [-1, pSize, pSize, 2])

  # Convolutional Layer #1
  # Computes 32 features using a 3x3 filter with ReLU activation.
  # Padding is added to preserve width and height.
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=64,kernel_size=[3, 3], padding="same",activation=tf.nn.relu) #None
    conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], padding="same",activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(inputs=conv3,filters=64,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    conv5 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3], padding="same",activation=tf.nn.relu)
    conv6 = tf.layers.conv2d(inputs=conv5,filters=128,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=2)

    conv7 = tf.layers.conv2d(inputs=pool3, filters=128, kernel_size=[3, 3], padding="same",activation=tf.nn.relu)
    conv8 = tf.layers.conv2d(inputs=conv7,filters=128,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)

    # Dense Layer
    # Densely connected layer with 1024 neurons
    conv2_flat = tf.reshape(conv8, [-1, 128 * 128 * 2])
    # Add dropout operation; 0.5 probability that element will be kept
    dropout = tf.layers.dropout(inputs=conv2_flat, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    fully_connect = tf.layers.dense(inputs=dropout, units=1024, activation=None) #activation=None
    predictions = tf.layers.dense(inputs=fully_connect, units=8, activation=None)

    #predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions = predictions)
    
    loss = tf.losses.mean_squared_error(labels=labels, predictions = predictions)
   
    #loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=predictions, name="softmax_tensor")

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    #eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["coord"])}#predictions=predictions["classes"])}
    
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "mean_square_error": tf.metrics.mean_squared_error(labels=labels, predictions = predictions)}#predictions=predictions["classes"])}

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops, predictions = predictions)


def load_data(filename):

    filedata = open(filename,'r') 
    lines = filedata.readlines() 

    patchSize = 128
    cornersLabel = np.zeros(8)
    data = []#list()
    labels = []#list()
    num_inst = 0
    for l in lines:
        if num_inst == 5000:
            break
        ls = l.split(' ', 10)
        filenameI1 = ls[0] 
        filenameI2 = ls[1]
        for p in range(0, 8):
            cornersLabel[p] = float(ls[ p + 2])

        I1 = cv2.imread(filenameI1, 0)
        I2 = cv2.imread(filenameI2, 0)

        ch1 = np.asarray(I1)#.flatten()
        ch2 = np.asarray(I2)#.flatten()

        imgs = np.swapaxes(np.stack([ch1, ch2]),0,2)

        if len(labels) == 0:
            labels = cornersLabel
            data = imgs[None,...]# 4th dimension
        else:
            labels = np.vstack((labels, cornersLabel))
            data = np.append(data, imgs[None,...], axis=0)
        num_inst = num_inst + 1

    return np.asarray(data, dtype=np.float32), np.asarray(labels, dtype=np.float32)

def main(argv):

    if len(argv) < 2:
        print("Error: not enought parameters!\Select mode (train, test) with [0|1]\npython create_dataset.py [0|1]")
        sys.exit(1)

    mode = int(argv[1])
    
    # Load training and eval data
    data, labels = load_data('data/train_data_list.txt')
    # Normalize data image between[0,1]
    data =  np.divide(data, 255.0)

    print('data shape : ' + str(data.shape))
    print('label shape : ' + str(labels.shape))

    num_inst = data.shape[0]
    # 75% train 25% val
    num_eval = int(round(float(num_inst) * 0.25))
    eval_data = data[-num_eval:]
    eval_labels = labels[-num_eval:]
    train_data = data[0:num_inst-num_eval]
    train_labels = labels[0:num_inst-num_eval]
    # TODO define batchs (memory!)
   
    print('train data shape : ' + str(train_data.shape))
    print('train label shape : ' + str(train_labels.shape))
    print('test data shape : ' + str(eval_data.shape))
    print('test labels shape : ' + str(eval_labels.shape))
    """
    # verbose
    for i in range(len(train_labels)):
        print('label: ' + str(i))
        print('(' + str(train_labels[i][0]) + ', ' + str(train_labels[i][1]) + ') - ('+
            str(train_labels[i][2]) + ', ' + str(train_labels[i][3]) + ') - (' +
            str(train_labels[i][4]) + ', ' + str(train_labels[i][5]) + ') - (' +
            str(train_labels[i][6]) + ', ' + str(train_labels[i][7]) + ')')

        # Display the images from data!
        mat_1 = np.asmatrix(train_data[i,:,:,0])#, dtype = np.uint8)
        mat_2 = np.asmatrix(train_data[i,:,:,1])#, dtype = np.uint8)
        print(mat_1)
        print(mat_2)
        cv2.imshow('I1', mat_1)
        cv2.imshow('I2', mat_2)
        cv2.waitKey(0)
    """
    # Create the Estimator
    my_hom_classifier = tf.estimator.Estimator(model_fn=my_hom_cnn_model_fn, model_dir="/home/mondejar/homography_cnn/models/my_hom_convnet_model")

    if mode == 0:
        # Train the model
        # Shuffle the data?
        train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data}, y=train_labels, batch_size = 100,num_epochs=None,shuffle=True)
        my_hom_classifier.train(input_fn=train_input_fn, steps=20000)#, hooks=[logging_hook])

    elif mode == 1:
        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data}, y=eval_labels, num_epochs=1,shuffle=False)
        eval_results = my_hom_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)
        predictions = list(my_hom_classifier.predict(input_fn=eval_input_fn))
        
        for i in range(len(predictions)):
            print("Predicted corner: ", np.around(predictions[i] * pSize))
            print("GT corner: ", np.around(eval_labels[i] * pSize))

            # Display 
            ev0 = np.multiply(eval_data[i,:,:,0], 255.0)
            ev1 = np.multiply(eval_data[i,:,:,1], 255.0)

            mat_1 = np.asmatrix(ev0, dtype = np.uint8)#eval_data[i,:,:,0])#, dtype = np.uint8)
            mat_2 = np.asmatrix(ev1, dtype = np.uint8)#eval_data[i,:,:,1])

            mat_1 = cv2.normalize(mat_1, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            mat_2 = cv2.normalize(mat_2, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            mat_2 = cv2.cvtColor(mat_2,cv2.COLOR_GRAY2RGB)
            for p in range(0, 4):	
                #ground truth corner
                gt_corner = int(round(eval_labels[i][0 + (p*2)] * pSize)), int(round(eval_labels[i][1 + (p*2)] * pSize))
                cv2.circle(mat_2, (gt_corner[0], gt_corner[1]), 5, [10, 255, 20], -1)

                #predicted corner
                p_corner = int(round(predictions[i][0 + (p*2)] * pSize)), int(round(predictions[i][1 + (p*2)] * pSize))
                cv2.circle(mat_2, (p_corner[0], p_corner[1]), 5, [200, 50, 20], -1)

            cv2.imshow('I1', mat_1)
            cv2.imshow('I2', mat_2)
            cv2.waitKey(0)

 
if __name__ == "__main__":
    tf.app.run()