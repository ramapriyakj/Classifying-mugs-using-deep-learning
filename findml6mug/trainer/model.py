#!/usr/bin/env python
"""This file contains all the model information: the training steps, the batch size and the model iself."""

import tensorflow as tf


def get_training_steps():
    """Returns the number of batches that will be used to train your solution.
    It is recommended to change this value."""
    return 3000


def get_batch_size():
    """Returns the batch size that will be used by your solution.
    It is recommended to change this value."""
    return 20


def getModel(features):
    # Input Layer (a batch of images that have 64x64 pixels and are RGB colored (3)
    input_layer = tf.reshape(features["x"], [-1, 64, 64, 3])

    model = tf.layers.Conv2D(128, 3, activation=tf.nn.relu)(input_layer)
    model = tf.layers.Conv2D(128, 3, activation=tf.nn.relu)(model)
    model = tf.layers.MaxPooling2D(2, 2)(model)

    model = tf.layers.Conv2D(32, 1, activation=tf.nn.relu)(model)
    model = tf.layers.Conv2D(32, 1, activation=tf.nn.relu)(model)
    model = tf.layers.MaxPooling2D(2, 2)(model)

    model = tf.layers.Conv2D(32, 2, activation=tf.nn.relu)(model)
    model = tf.layers.Conv2D(32, 2, activation=tf.nn.relu)(model)
    model = tf.layers.MaxPooling2D(2, 2)(model)

    model = tf.layers.Conv2D(32, 1, activation=tf.nn.relu)(model)
    model = tf.layers.Conv2D(32, 1, activation=tf.nn.relu)(model)
    model = tf.layers.MaxPooling2D(2, 2)(model)

    model = tf.layers.Flatten()(model)

    model = tf.layers.Dense(512, activation=tf.nn.relu)(model)
    model = tf.layers.Dropout(0.4)(model)

    model = tf.layers.Dense(128, activation=tf.nn.relu)(model)
    model = tf.layers.Dropout(0.1)(model)

    model = tf.layers.Dense(64, activation=tf.nn.relu)(model)
    model = tf.layers.Dropout(0.1)(model)

    model = tf.layers.Dense(4)(model)

    return model


def solution(features, labels, mode):
    """Returns an EstimatorSpec that is constructed using the solution that you have to write below."""
    #TODO: Code of your solution
    logits = getModel(features)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        #TODO: return tf.estimator.EstimatorSpec with prediction values of all classes
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        #TODO: Let the model train here
        #TODO: return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        optimizer = tf.contrib.opt.AdamWOptimizer(0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}

    if mode == tf.estimator.ModeKeys.EVAL:
        #TODO: eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=classes)}
        #TODO: return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

