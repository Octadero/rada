"""
    Projector realisation for data visualisation.
    Author: Volodymyr Pavliukevych.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

# Create randomly initialized embedding weights which will be trained.
first_D = 23998 # Number of items (size).
second_D = 11999 # Number of items (size).

DATA_DIR = ''
LOG_DIR = DATA_DIR + 'embedding/'

first_rada_input = np.loadtxt(DATA_DIR + 'result_' + str(first_D) + '/rada_full_packed.tsv', delimiter='\t')
second_rada_input = np.loadtxt(DATA_DIR + 'result_' + str(second_D) + '/rada_full_packed.tsv', delimiter='\t')

first_embedding_var = tf.Variable(first_rada_input, name='politicians_embedding_' + str(first_D))
second_embedding_var = tf.Variable(second_rada_input, name='politicians_embedding_' + str(second_D))

saver = tf.train.Saver()
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    saver.save(session, os.path.join(LOG_DIR, "model.ckpt"), 0)

config = projector.ProjectorConfig()

# You can add multiple embeddings. Here we add only one.
first_embedding = config.embeddings.add()
second_embedding = config.embeddings.add()

first_embedding.tensor_name = first_embedding_var.name
second_embedding.tensor_name = second_embedding_var.name

# Link this tensor to its metadata file (e.g. labels).
first_embedding.metadata_path = os.path.join(DATA_DIR, '../rada_full_packed_labels.tsv')
second_embedding.metadata_path = os.path.join(DATA_DIR, '../rada_full_packed_labels.tsv')

first_embedding.bookmarks_path = = os.path.join(DATA_DIR, '../result_23998/bookmarks.txt')
second_embedding.bookmarks_path = = os.path.join(DATA_DIR, '../result_11999/bookmarks.txt')

# Use the same LOG_DIR where you stored your checkpoint.
summary_writer = tf.summary.FileWriter(LOG_DIR)

# The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
# read this file during startup.
projector.visualize_embeddings(summary_writer, config)
