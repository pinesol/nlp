#! /usr/bin/env python

import datetime
import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib import learn
import time

import data
import cbow

# Program infra flags
tf.flags.DEFINE_string("exp_name",
                       datetime.datetime.now().strftime('%Y%m%d%H%M%S'),
                       "The name of the experiment. Defaults to timestamp") 
tf.flags.DEFINE_string("use_pickle", True, "If set, data is loaded from a precomputed file.")
tf.flags.DEFINE_boolean("test", False, "If true, the model is run on a much smaller dataset. Overrides some flags.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 64, "Dimensionality of character embedding (default: 64)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout probability (default: 0.5)")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Initial learning rate. Defaults to 0.001")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 8, "Number of training epochs (default: 8)")
tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on val set after this many steps (default: 200)") # TODO try evaling every step
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 1000)")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
    print("")

if FLAGS.test:
    TEST_DATA_LEN = 1000
    print('Testing mode is on, only using the first {} data points'.format(TEST_DATA_LEN))
    TEST_NUM_EPOCHS = 1
    print('Testing mode is on: only doing {} epochs'.format(TEST_NUM_EPOCHS))
    FLAGS.num_epochs = TEST_NUM_EPOCHS
    FLAGS.exp_name = 'test'
    vocab, reviews, labels = data.load(use_pickle=FLAGS.use_pickle, max_reviews=TEST_DATA_LEN)
else:
    vocab, reviews, labels = data.load(use_pickle=FLAGS.use_pickle)
    
vocab_id_reviews = data.make_vocab_id_reviews(vocab, reviews)
x_train, x_val, y_train, y_val = data.shuffle_split_data(vocab_id_reviews, labels)


# Training
# ==================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cbow = cbow.CBOW(sequence_length=x_train.shape[1],
                         num_classes=2,
                         vocab_size=len(vocab) + 1, # +1 is for the padding word (0).
                         embedding_size=FLAGS.embedding_dim)
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(cbow.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", FLAGS.exp_name, timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cbow.loss)
        acc_summary = tf.scalar_summary("accuracy", cbow.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

        # Val summaries
        val_summary_op = tf.merge_summary([loss_summary, acc_summary])
        val_summary_dir = os.path.join(out_dir, "summaries", "val")
        val_summary_writer = tf.train.SummaryWriter(val_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cbow.input_x: x_batch,
              cbow.input_y: y_batch,
              cbow.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cbow.loss, cbow.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def val_step(x_batch, y_batch):
            """
            Evaluates model on a val set
            """
            feed_dict = {
              cbow.input_x: x_batch,
              cbow.input_y: y_batch,
              cbow.dropout_keep_prob: 1  # don't do dropout on validation
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, val_summary_op, cbow.loss, cbow.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            val_summary_writer.add_summary(summaries, step)
            
        # Generate batches
        batches = data.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        current_step = None
        # Training loop. For each batch...
        for batch in batches:
            if len(batch) == 0: # TODO I hope the existence of zero-sized batches isn't a bug...
                print 'empty batch, skipping...' # TODO it stopped here? why? or was it done as step 5000?
                continue
            x_batch, y_batch = zip(*batch) 
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                val_step(x_val, y_val)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
        # Eval and checkpoint if it wasn't done on the last step
        if current_step % FLAGS.evaluate_every != 0:
            print("\nEvaluation:")
            val_step(x_val, y_val)
        if current_step % FLAGS.checkpoint_every != 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))
        train_summary_writer.flush()
        val_summary_writer.flush()
