#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/4/30
"""

import tensorflow as tf


def histogram_fixed_width(values,
                          value_range,
                          nbins=100,
                          dtype=tf.int32,
                          name=None):
    """Return histogram of values.
    Given the tensor `values`, this operation returns a rank 1 histogram counting
    the number of entries in `values` that fell into every bin.  The bins are
    equal width and determined by the arguments `value_range` and `nbins`.
    Args:
      values:  Numeric `Tensor`.
      value_range:  Shape [2] `Tensor` of same `dtype` as `values`.
        values <= value_range[0] will be mapped to hist[0],
        values >= value_range[1] will be mapped to hist[-1].
      nbins:  Scalar `int32 Tensor`.  Number of histogram bins.
      dtype:  dtype for returned histogram.
      name:  A name for this operation (defaults to 'histogram_fixed_width').
    Returns:
      A 1-D `Tensor` holding histogram of values.
    Examples:
    ```python
    # Bins will be:  (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
    nbins = 5
    value_range = [0.0, 5.0]
    new_values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]
    with tf.default_session() as sess:
      hist = tf.histogram_fixed_width(new_values, value_range, nbins=5)
      variables.global_variables_initializer().run()
      sess.run(hist) => [2, 1, 1, 0, 2]
    ```
    """
    with tf.name_scope(name, 'histogram_fixed_width',
                       [values, value_range, nbins]) as scope:
        values = tf.convert_to_tensor(values, name='values')
        values = tf.reshape(values, [-1])
        value_range = tf.convert_to_tensor(value_range, name='value_range')
        nbins = tf.convert_to_tensor(nbins, dtype=tf.int32, name='nbins')
        nbins_float = tf.cast(nbins, values.dtype)

        # Map tensor values that fall within value_range to [0, 1].
        scaled_values = tf.truediv(values - value_range[0],
                                   value_range[1] - value_range[0],
                                   name='scaled_values')

        # map tensor values within the open interval value_range to {0,.., nbins-1},
        # values outside the open interval will be zero or less, or nbins or more.
        indices = tf.floor(nbins_float * scaled_values, name='indices')

        # Clip edge cases (e.g. value = value_range[1]) or "outliers."
        indices = tf.cast(tf.clip_by_value(indices, 0, nbins_float - 1), tf.int32)

        # TODO(langmore) This creates an array of ones to add up and place in the
        # bins.  This is inefficient, so replace when a better Op is available.
        # return tf.unsorted_segment_sum(
        #     tf.ones_like(indices, dtype=dtype),
        #     indices,
        #     nbins,
        #     name=scope)
        # return indices
        counts = tf.one_hot(indices, depth=nbins, dtype=tf.int32)
        res = tf.reduce_sum(counts, axis=0)
        return res


def cumsum(tensor_arr):
    values = tf.split(tensor_arr, tensor_arr.get_shape()[0], 0)
    out = []
    prev = tf.zeros_like(values[0], dtype=tf.int32)
    for val in values:
        s = prev + val
        out.append(s)
        prev = s
    res = tf.concat(out, axis=0)
    return res


def tf_abs(val):
    val = tf.cond(val < 0, lambda: tf.multiply(val, -1), lambda: val)
    return val
