#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/4/27
"""

from __future__ import print_function

import os

import tensorflow as tf
from google.protobuf import text_format
from tensorflow.python.framework import graph_io

from root_dir import MODELS_DIR

filename = os.path.join(MODELS_DIR, "gaze_opt_b2.m")

with open(filename + '.pbtxt', 'r') as f:
    graph_def = tf.GraphDef()
    file_content = f.read()
    text_format.Merge(file_content, graph_def)
    graph_io.write_graph(graph_def,
                         os.path.dirname(filename),
                         os.path.basename(filename) + '.pb',
                         as_text=False)

print('Converted %s.pbtxt to %s.pb' % (filename, filename))
