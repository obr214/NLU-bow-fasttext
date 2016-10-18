#! /usr/bin/env python

import os
import tensorflow as tf
import numpy as np
import pickle
from pathlib import Path
from data_helpers import *
from tensorflow.contrib import learn


# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:

    # Creates a flag for ngrams
    bi_Gram_flag = False

    # Loads the data
    print("Loading Data...")
    test_positive_reviews, test_negative_reviews = get_test_sets()

    # Loads the vocabulary
    print("Loading Vocabulary...")
    vocabulary = load_vocabulary("data/vocab_unigrams_no_counts/part-00000")

    # Label the data
    print("Labeling Data...")
    x_test, y_test = label_data(test_positive_reviews, test_negative_reviews)

    if bi_Gram_flag:
        print("BI GRAMS ON")
        bi_grams = parse_ngrams(x_test, 2)
        bigram_vocab = load_vocabulary("data/vocab_bigrams_no_counts/part-00000", 5000)
        print("Replacing with NOOV...")
        reviews_noov_file = Path("data/reviews_test_noov.p")
        if reviews_noov_file.is_file():
            x_test_reviews_noov = pickle.load(open("data/reviews_test_noov.p", "rb"))
        else:
            x_test_reviews_noov = set_noov(bi_grams, bigram_vocab)
            # x_train_reviews_oov = set_oov_tag(x_train, vocabulary)
            pickle.dump(x_test_reviews_noov, open("data/reviews_test_noov.p", "wb"))
        print("End replacing with NOOV")

    # Replace the words not in vocabulary for 'oov' tag
    print("Replacing with OOV...")
    reviews_oov_file = Path("data/reviews_test_oov.p")
    if reviews_oov_file.is_file():
        x_test_reviews_oov = pickle.load(open("data/reviews_test_oov.p", "rb"))
    else:
        x_test_reviews_oov = set_oov(x_test, vocabulary)
        # x_train_reviews_oov = set_oov_tag(x_train, vocabulary)
        pickle.dump(x_test_reviews_oov, open("data/reviews_test_oov.p", "wb"))
    print("End replacing with OOV")

    x_test_revs = x_test_reviews_oov

    if bi_Gram_flag:
        x_test_revs = [x_test_reviews_oov[i] + " " + x_test_reviews_noov[i] for i in range(len(x_test))]

    #Fits the Test Reviews
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test_reviews = np.array(list(vocab_processor.transform(x_test_revs)))

    y_test = np.argmax(y_test, axis=1)
else:
    x_test_reviews = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = batch_iter(list(x_test_reviews), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if x_dev_labels is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
