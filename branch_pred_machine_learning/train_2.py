#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 09:08:39 2018

@author: Siyuan
"""
import tensorflow as tf

import numpy as np

def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 1, 3, 1])
    
    conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5,5],
            padding="same",
            activation=tf.nn.relu)
    
    pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[1,3],
            strides=1)
    
    pool1_flat = tf.reshape(pool1, [-1, 32])
    dense = tf.layers.dense(inputs=pool1_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    logits = tf.layers.dense(inputs=dropout, units=2)
    
    predictions = {
    "classes": tf.argmax(input=logits, axis=1),
    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    eval_metric_ops = {
    "accuracy": tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    
"""
filename_queue = tf.train.string_input_producer(["SHORT_MOBILE-36.bt9.trace.csv"])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[0], [1], [0], [9]]
PC, taken, target, opType = tf.decode_csv(
    value, record_defaults=record_defaults)
features = tf.stack([PC, target, opType])
features = tf.cast(features, tf.float32)
taken = tf.cast(taken, tf.float32)
"""
from numpy import genfromtxt
my_data = genfromtxt("SHORT_MOBILE-36.bt9.trace.csv", delimiter=',', dtype=np.float32)
data = my_data[:,[0,2,3]]
taken = my_data[:,1].astype(int)

my_data_eval = genfromtxt("SHORT_MOBILE-3.bt9.trace.csv", delimiter=',', dtype=np.float32)
data_eval = my_data_eval[:,[0,2,3]]
taken_eval = my_data_eval[:,1].astype(int)

br_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="./model/")

tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": data},
    y=taken,
    batch_size=100,
    num_epochs=None,
    shuffle=True)
br_classifier.train(
    input_fn=train_input_fn,
    steps=20000,
    hooks=[logging_hook])

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": data_eval},
    y=taken_eval,
    num_epochs=1,
    shuffle=False)
eval_results = br_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)


"""
with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(1200):
    # Retrieve a single instance:
    example, label = sess.run([features, taken])

  coord.request_stop()
  coord.join(threads)
"""

