# Gedamu
# center for future Media
#  Domain Adaptation loss correlation loss
from functools import partial
import tensorflow as tf

import grl_ops
import utils

slim = tf.contrib.slim


def MMD(x, y, kernel=utils.gaussian_kernel_matrix):
    with tf.name_scope("MMD"):
        cost = tf.reduce_mean(kernel(x, x))
        cost += tf.reduce_mean(kernel(y, y))
        cost -= 2 * tf.reduce_mean(kernel(x,y))

        cost = tf.where(cost > 0, cost, 0, name= 'Value')
        return cost

def mmdLoss(source, target, weight, scope):

    sigma = [
        1e-6
    ]
    gassuian_kernel = partial(
        utils.gaussian_kernel_matrix, sigma = tf.constant(sigma))
    loss_value = MMD(source, target, kernel=gassuian_kernel)
    loss_value = tf.maximum(1e-4, loss_value) * weight
    asset_op = tf.Assert(tf.is_finite(loss_value), [loss_value])

    with tf.control_dependencies([asset_op]):
        tag = "MMd Loss"
        if scope:
            tag = tag + scope
        tf.Summary.scalar(tag, loss_value)
        tf.losses.add_loss(loss_value)
        return loss_value

def correlation(source, target, weight, scope=None):
    with tf.name_scope("corr-loss"):
        source -= tf.reduce_mean(source, 0)
        target -= tf.reduce_mean(target, 0)

        source = tf.nn.l2_normalize(source, 1)
        target = tf.nn.l2_normalize(target, 1)

        source_corr = tf.matmul(tf.transpose(source), source)
        target_corr = tf.matmul(tf.transpose(target), target)

        corr_loss = tf.reduce_mean(tf.sqrt(source_corr-target_corr)) * weight

        assert_op = tf.Assert(tf.is_finite(corr_loss), [corr_loss])
        with tf.control_dependencies([assert_op]):
            tag = "Correlation"
            if scope:
                tag = scope + tag
            tf.Summary.scala(tag, corr_loss)
            tf.losses.add_loss(corr_loss)
        return  corr_loss

