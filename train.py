#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from data_helpers import *
from cbow import TextCBoW
from pathlib import Path
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 200, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

# Creates a flag for ngrams
bi_Gram_flag = False
# TODO: max_document_length should be changed to a global parameter
max_document_length = 250

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparatopn
# ==================================================

# Loads the data
print("Loading Data...")
train_positive_reviews, train_negative_reviews = get_train_sets()

# Loads the vocabulary
print("Loading Vocabulary...")
vocabulary = load_vocabulary("data/vocab_unigrams_no_counts/part-00000")

# Label the data
print("Labeling Data...")
x_train, y_train = label_data(train_positive_reviews, train_negative_reviews)

if bi_Gram_flag:
    print("BI GRAMS ON")
    bi_grams = parse_ngrams(x_train, 2)
    bigram_vocab = load_vocabulary("data/vocab_bigrams_no_counts/part-00000", 5000)
    print("Replacing with NOOV...")
    reviews_noov_file = Path("data/reviews_noov.p")
    if reviews_noov_file.is_file():
        x_train_reviews_noov = pickle.load(open("data/reviews_noov.p", "rb"))
    else:
        x_train_reviews_noov = set_noov(bi_grams, bigram_vocab)
        # x_train_reviews_oov = set_oov_tag(x_train, vocabulary)
        pickle.dump(x_train_reviews_noov, open("data/reviews_noov.p", "wb"))
    print("End replacing with NOOV")

# Replace the words not in vocabulary for 'oov' tag
print("Replacing with OOV...")
reviews_oov_file = Path("data/reviews_oov.p")
if reviews_oov_file.is_file():
    x_train_reviews_oov = pickle.load(open("data/reviews_oov.p", "rb"))
else:
    x_train_reviews_oov = set_oov(x_train, vocabulary)
    # x_train_reviews_oov = set_oov_tag(x_train, vocabulary)
    pickle.dump(x_train_reviews_oov, open("data/reviews_oov.p", "wb"))
print("End replacing with OOV")

x_train_revs = x_train_reviews_oov

if bi_Gram_flag:
    x_train_revs = [x_train_reviews_oov[i] + " " + x_train_reviews_noov[i] for i in range(len(x_train))]
    max_document_length *= 2


print("Splits Data Train and Dev")
x_train_list, x_dev_list = split_train_validation(x_train_revs, y_train)

# Gets the Train Reviews
x_train_reviews_preidx = x_train_list[0]
x_train_labels = x_train_list[1]

# Creates the indexes

vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x_train_reviews = np.array(list(vocab_processor.fit_transform(x_train_reviews_preidx)))

# Gets the Dev Reviews and Save them into pickle files
x_dev_reviews_preidx = x_dev_list[0]
x_dev_labels = x_dev_list[1]

# Fits the dev_reviews
x_dev_reviews = np.array(list(vocab_processor.transform(x_dev_reviews_preidx)))

# Saves the dev set
pickle.dump(x_dev_reviews, open("data/x_dev_reviews.p", "wb"))
pickle.dump(x_dev_labels, open("data/x_dev_labels.p", "wb"))

print("Sequence Length: {:d}".format(x_train_reviews.shape[1]))
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(x_train_reviews), len(x_dev_labels)))

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cbow = TextCBoW(
            sequence_length=x_train_reviews.shape[1],
            num_classes=2,
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            n_hidden=256,
            dropout_keep_prob=FLAGS.dropout_keep_prob,
            l2_reg_lambda=FLAGS.l2_reg_lambda
        )

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # TODO: Change the Optimizer
        optimizer = tf.train.AdamOptimizer(1e-3)
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
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cbow.loss)
        acc_summary = tf.scalar_summary("accuracy", cbow.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

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

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cbow.input_x: x_batch,
              cbow.input_y: y_batch,
              cbow.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cbow.loss, cbow.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
            return loss, accuracy

        # Generate batches
        batches = batch_iter(
            list(zip(x_train_reviews, x_train_labels)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...

        best_step = None
        best_loss = 100000
        break_count = 0

        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)

            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                loss, accuracy = dev_step(x_dev_reviews, x_dev_labels, writer=dev_summary_writer)
                if loss < best_loss:
                    break_count = 0
                    best_loss = loss
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                else:
                    break_count += 1

                if break_count == 15:
                    break
                print("")
