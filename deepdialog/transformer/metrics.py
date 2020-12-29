import tensorflow as tf

# copy from https://github.com/tensorflow/models/blob/master/official/transformer/utils/metrics.py

# PAD 部分は loss や accuracy の計算に入れないこと、
# label smoothing と言って正解データを単純な one-hot ではなく、
# 少しなまらせたものにするなどの工夫がされている。

def padded_cross_entropy_loss(logits, labels, smoothing, vocab_size):
    """Calculate cross entropy loss while ignoring padding.
    Args:
      logits: Tensor of size [batch_size, length_logits, vocab_size]
      labels: Tensor of size [batch_size, length_labels]
      smoothing: Label smoothing constant, used to determine the on and off values
      vocab_size: int size of the vocabulary
    Returns:
      Returns the cross entropy loss and weight tensors: float32 tensors with
        shape [batch_size, max(length_logits, length_labels)]
    """
    # Check!!!
    # with tf.name_scope("loss", values=[logits, labels]):
    with tf.name_scope("loss"):
        logits, labels = _pad_tensors_to_same_length(logits, labels)

        # Calculate smoothing cross entropy
        # Check!!!!
        # with tf.name_scope("smoothing_cross_entropy", values=[logits, labels]):
        with tf.name_scope("smoothing_cross_entropy"):
            confidence = 1.0 - smoothing
            # Check!!!!
            # low_confidence = (1.0 - confidence) / tf.to_float(vocab_size - 1)
            low_confidence = (1.0 - confidence) / tf.cast(vocab_size - 1,dtype=tf.float32)
            soft_targets = tf.one_hot(
                tf.cast(labels, tf.int32),
                depth=vocab_size,
                on_value=confidence, #確率?を1ではなく、0.95とする。
                off_value=low_confidence) #確率?を0ではなく、0.0..1とする。
            # Check!!!
            # xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            xentropy = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=soft_targets)

            # Calculate the best (lowest) possible value of cross entropy, and
            # subtract from the cross entropy loss.
            # 交差エントロピー誤差はH(p,q)=-sigma_x p(x)log(q(x))で求まり、
            # best possbile value of cross entropyはpとqが同じ時、つまり、
            # p=(confidence, low_confidence, low_confidence, ... , low_confidence)
            # q=(confidence, low_confidence, low_confidence, ... , low_confidence)
            #　なので、これで交差エントロピー誤差を求めたものがnormalizing_constant
            # normalizing_constanを引く理由は以下の通り。
            #---> Then, a small problem is that cross entropy loss value becomes large. Even when the model is 100% accurate, the loss is not zero because of the label smoothing. So, we just subtract the "normalizing" constant value from the cross entropy value. Then, loss will be close to zero as the model becomes accurate. This does not affect to backward propagation, but it just make it clear to debug if the loss gets stuck or converged toward an optimal point.
            normalizing_constant = -(
                # Check!!!
                # confidence * tf.log(confidence) + tf.to_float(vocab_size - 1) *
                confidence * tf.math.log(confidence) + tf.cast(vocab_size - 1,dtype=tf.float32) *
                # Check!!!
                # low_confidence * tf.log(low_confidence + 1e-20))
                low_confidence * tf.math.log(low_confidence + 1e-20))
            xentropy -= normalizing_constant

        # weightsはラベルが0以外の箇所をTrueにして1にする。つまり、0以外は全て1になる。
        # Check!!!
        # weights = tf.to_float(tf.not_equal(labels, 0))
        weights = tf.cast(tf.not_equal(labels, 0),dtype=tf.float32)
        return xentropy * weights, weights


def padded_accuracy(logits, labels):
    """Percentage of times that predictions matches labels on non-0s."""
    # Check!!!
    # with tf.variable_scope("padded_accuracy", values=[logits, labels]):
    with tf.compat.v1.variable_scope("padded_accuracy", values=[logits, labels]):
        logits, labels = _pad_tensors_to_same_length(logits, labels)
        #Check!!!
        # weights = tf.to_float(tf.not_equal(labels, 0))
        weights = tf.cast(tf.not_equal(labels, 0),dtype=tf.float32)
        # Check!!!
        # outputs = tf.to_int32(tf.argmax(logits, axis=-1))
        outputs = tf.cast(tf.argmax(logits, axis=-1),dtype=tf.int32)
        # Check!!!
        # padded_labels = tf.to_int32(labels)
        padded_labels = tf.cast(labels,dtype=tf.int32)
        # Check!!!
        # return tf.to_float(tf.equal(outputs, padded_labels)), weights
        return tf.cast(tf.equal(outputs, padded_labels),dtype=tf.float32), weights


def _pad_tensors_to_same_length(x, y):
    """Pad x and y so that the results have the same length (second dimension)."""
    with tf.name_scope("pad_to_same_length"):
        x_length = tf.shape(x)[1]
        y_length = tf.shape(y)[1]

        max_length = tf.maximum(x_length, y_length)

        x = tf.pad(x, [[0, 0], [0, max_length - x_length], [0, 0]])
        y = tf.pad(y, [[0, 0], [0, max_length - y_length]])
        return x, y
