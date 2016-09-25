import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0, add_second_conv_layer=False):

        # Placeholders for input, output and dropout (which you need to implement!!!!)
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1) # TODO expand_dims?

        # embedded_chars_expanded dims: [batch_size=?, sequence_length=sequence_length (56), embedding_size=64, channels=1]
            
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]  #height, width, input channel depth, output channel depth
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Problem 1: Activation
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # h dimension = [batch_size=?, senquence_length=sequence_length-filter_size+1, 1, num_filters]
                # inputs to the pooling layer
                layer_to_pool = h
                convd_sequence_length = sequence_length - filter_size + 1
                
                if add_second_conv_layer:
                    # Convolution Layer
                    filter_shape2 = [filter_size, 1, num_filters, num_filters]  # TODO not confident in this
                    W2 = tf.Variable(tf.truncated_normal(filter_shape2, stddev=0.1), name="W2")
                    b2 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b2")
                    conv2 = tf.nn.conv2d(
                        h,
                        W2,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv2")
                    # Apply nonlinearity
                    # Problem 1: Activation
                    h2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name="relu2")
                    # Update inputs to the pooling layer
                    layer_to_pool = h2
                    convd_sequence_length = convd_sequence_length - filter_size + 1

                # Maxpooling over the outputs
                # size of the pooled layer output: "[batch_size, 1, 1, num_filters]"
                pooled = tf.nn.max_pool(
                    layer_to_pool,
                    ksize=[1, convd_sequence_length, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
            
        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Problem 2: Regularization
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_pool_flat, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
