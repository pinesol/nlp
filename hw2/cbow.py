
import tensorflow as tf
import numpy as np

# Takes in a sequence of words (a movie review) as a list of one-hot vectors,
# gets their embeddings, takes their average, and puts them through a hidden layer (relu), and then
# a softmax


class CBOW(object):
    # TODO add dropout?
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size):
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        
        # Embedding layer
        # TODO why is this put on the CPU explicitly??
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # Embedding matrix, initialized with uniform random values between -1 and 1.
            E = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="E")
            embeddings = tf.nn.embedding_lookup(E, self.input_x)
            # This adds another dimension at the end
            # TODO why did text_cnn do this? i'm not doing it
#            embeddings_expanded = tf.expand_dims(embeddings), -1)

        with tf.name_scope("hidden"):
            #_expanded TODO
            # TODO consider using 'sum'?
            mean_embedding = tf.reduce_mean(embeddings, 1, name="mean")

        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(mean_embedding, self.dropout_keep_prob)
            
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[embedding_size, num_classes],
                initializer=tf.contrib.layers.xavier_initializer()) # TODO what is this xavier initializer?
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
            predictions = tf.argmax(scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(scores, self.input_y)
            self.loss = tf.reduce_mean(losses)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
