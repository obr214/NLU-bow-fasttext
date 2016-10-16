import tensorflow as tf


class TextCBoW(object):

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, n_hidden, dropout_keep_prob,l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout (which you need to implement!!!!)

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        # self.dropout_keep_prob  = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.dropout_keep_prob = tf.constant(0.5, name="dropout_keep_prob")
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            E = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="E")
            self.embedded_chars = tf.nn.embedding_lookup(E, self.input_x)

            print("embedded_chars: {}".format(self.embedded_chars.get_shape()))
            self.embedded_chars_reduced = tf.reduce_mean(self.embedded_chars, 1)
            print("embedded_chars_reduced: {}".format(self.embedded_chars_reduced.get_shape()))

            self.W = tf.Variable(tf.truncated_normal([embedding_size, n_hidden], stddev=0.1), name="W")
            self.b = tf.Variable(tf.random_normal([n_hidden]), name="b")

            print("W {}".format(self.W.get_shape()))

            # self.output_layer = tf.add(tf.matmul(self.embedded_chars_reduced, self.W), self.b)
            self.output_layer = tf.nn.xw_plus_b(self.embedded_chars_reduced, self.W, self.b, name="scores")
            # self.output_layer = tf.nn.tanh(self.output_layer, name="tanh")
            self.output_layer = tf.nn.relu(self.output_layer, name="relu")

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[n_hidden, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b( self.output_layer, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
