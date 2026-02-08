import tensorflow as tf

def NLL(y_true, y_pred):
    mu = y_pred[0]
    cov = y_pred[1]
    eps = 1e-6
    cov = tf.clip_by_value(cov, eps, float('inf'))  # ensure positivity
    nll = 0.5 * tf.math.log(2*3.14* cov) + 0.5 * tf.square(y_true - mu) / cov
    return tf.reduce_mean(nll)

