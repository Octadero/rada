""" Rada Auto Encoder Example.

Simple auto-encoder to compress high dimensional vector to a lower latent space.

References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.

Author: 
    Volodymyr Pavliukevych / Octadero
    website: https://octadero.com/

OpenData:
    Web site: http://data.rada.gov.ua/open
    Faces: http://w1.c1.rada.gov.ua/pls/radan_gs09/zal_frack_ank?kod=0 ... 500
    Vote list: http://w1.c1.rada.gov.ua/pls/radan_gs09/ns_golos?g_id=345
    Politisians: http://w1.c1.rada.gov.ua/pls/site2/p_deputat_list
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Read CSV Data
file = np.load('rada_full.npz')
rada_input = file['arr_0']
print("Shape of input vector: ", np.shape(rada_input))

# Training Parameters
learning_rate = 0.01
num_steps = 1000000
display_step = 1000
track_step = 100

# Network Parameters
num_input = 23998 # Input shape
num_hidden_1 = 110 # 1st layer num features
num_hidden_2 = 80 # 2nd layer num features
num_hidden_3 = 60 # 3rd layer num features 
log_path = '/tmp/autoencoder/23998_453x'+str(num_hidden_1)+'x'+str(num_hidden_2)+'x'+str(num_hidden_3)+'/'
# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])

# Building the encoder
def encoder(x):
    with tf.variable_scope('encoder', reuse=False):
        with tf.variable_scope('layer_1', reuse=False):
            w1 = tf.Variable(tf.random_normal([num_input, num_hidden_1]), name="w1")
            b1 = tf.Variable(tf.random_normal([num_hidden_1]), name="b1")
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, w1), b1))
            tf.summary.histogram("histogram-w1", w1)
            tf.summary.histogram("histogram-b1", b1)

        with tf.variable_scope('layer_2', reuse=False):
            w2 = tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2]), name="w2")
            b2 = tf.Variable(tf.random_normal([num_hidden_2]), name="b2")
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w2), b2))
            tf.summary.histogram("histogram-w2", w2)
            tf.summary.histogram("histogram-b2", b2)
        
        with tf.variable_scope('layer_3', reuse=False):
            w2 = tf.Variable(tf.random_normal([num_hidden_2, num_hidden_3]), name="w2")
            b2 = tf.Variable(tf.random_normal([num_hidden_3]), name="b2")
            layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, w2), b2))
            tf.summary.histogram("histogram-w2", w2)
            tf.summary.histogram("histogram-b2", b2)    
            return layer_3

# Building the decoder
def decoder(x):
    with tf.variable_scope('decoder', reuse=False):
        with tf.variable_scope('layer_1', reuse=False):
            w1 = tf.Variable(tf.random_normal([num_hidden_3, num_hidden_2]), name="w1")
            b1 = tf.Variable(tf.random_normal([num_hidden_2]), name="b1")
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, w1), b1))
            tf.summary.histogram("histogram-w1", w1)
            tf.summary.histogram("histogram-b1", b1)

        with tf.variable_scope('layer_2', reuse=False):
            w1 = tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1]), name="w1")
            b1 = tf.Variable(tf.random_normal([num_hidden_1]), name="b1")
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w1), b1))
            tf.summary.histogram("histogram-w1", w1)
            tf.summary.histogram("histogram-b1", b1)
            
        with tf.variable_scope('layer_3', reuse=False):
            w2 = tf.Variable(tf.random_normal([num_hidden_1, num_input]), name="w2")
            b2 = tf.Variable(tf.random_normal([num_input]), name="2")
            layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, w2), b2))
            tf.summary.histogram("histogram-w2", w2) 
            tf.summary.histogram("histogram-b2", b2)           
            return layer_3

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
tf.summary.scalar("loss", loss)
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
# Using only 30% of available GPU memory.
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
config = tf.ConfigProto(device_count={'CPU' : 2, 'GPU' : 1}, allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
with tf.Session(config=config) as sess:

    # Run the initializer
    sess.run(init)
    # Save Graph for TensorBoard representation
    summary_writer = tf.summary.FileWriter(log_path, graph=sess.graph)
    summaries = tf.summary.merge_all()

    # add Saver
    saver = tf.train.Saver(tf.global_variables())
    episode_number = 1
    # Try to load saved model
    try:
        ckpt = tf.train.get_checkpoint_state(log_path)
        model_checkpoint_path = ckpt.model_checkpoint_path
        saver.restore(sess, model_checkpoint_path)
        print("Restore from: ", log_path, "model: ", model_checkpoint_path)
    except Exception as error: 
        print(error)
        print("There is no saved model to load.\n Starting new session.")
    else:
        print("loaded model: {}".format(model_checkpoint_path))
        saver = tf.train.Saver(tf.global_variables())
        episode_number = int(model_checkpoint_path.split('-')[-1])

    # Training
    for step in range(episode_number, num_steps+1):        
        # Run optimization op (backprop) and cost op (to get loss value)
        _, l, summ = sess.run([optimizer, loss, summaries], feed_dict={X: rada_input})

        if step % track_step == 0 or step == 1:
            summary_writer.add_summary(summ, global_step=step)
        # Display logs per step
        if step % display_step == 0 or step == 1:
            saver.save(sess, log_path + 'model.ckpt', step)
            print('Step %i: Minibatch Loss: %f' % (step, l))

    # Extract packed vector
    packed_layer_value = sess.run(encoder_op, feed_dict={X: rada_input})
    np.savetxt("rada_full_packed.tsv", packed_layer_value, delimiter="\t")

    # Ploting result
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig = np.empty((155 * n,155 * n))
    canvas_recon = np.empty((155 * n, 155 * n))
    for i in range(n):
        batch_x = rada_input[:n]
        # Encode and decode the digit image
        g = sess.run(decoder_op, feed_dict={X: batch_x})
        
        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * 155:(i + 1) * 155, j * 155:(j + 1) * 155] = \
                np.append(batch_x[j], [0]*27).reshape([155, 155])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * 155:(i + 1) * 155, j * 155:(j + 1) * 155] = \
                np.append(g[j], [0]*27).reshape([155, 155])

print("Original Images")
plt.figure(figsize=(n, n))
plt.imshow(canvas_orig, origin="upper", cmap="gray")
plt.savefig('canvas_orig.png')

print("Reconstructed Images")
plt.figure(figsize=(n, n))
plt.imshow(canvas_recon, origin="upper", cmap="gray")
plt.savefig('canvas_recon.png')