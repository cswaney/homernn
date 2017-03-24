import tensorflow as tf
import numpy as np
import h5py as h5
import collections
import time

"""Start by importing the dataset. Find the unique text characters, and create
   a mapping between characters and their integer representations."""

def load_data():

    odyssey = open('./data/odyssey.txt')
    text_lines = odyssey.readlines()
    odyssey.close()
    iliad = open('./data/iliad.txt')
    text_lines.extend(iliad.readlines())
    iliad.close()

    text_lines = [line.rstrip('\n') for line in text_lines]
    for line in text_lines:
        if len(line) == 0:
            text_lines.remove(line)
        if line[0:4] == 'BOOK':
            text_lines.remove(line)
        if line[0:4] == '----':
            text_lines.remove(line)
        if line == '\n':
            text_lines.remove(line)
    text_chars = []
    for line in text_lines:
        text_chars.extend(line)
    chars = list(set(text_chars))
    chars.sort()
    chars_map = {}
    for i in range(0, len(chars)):
        chars_map[chars[i]] = i

    with h5.File('./data/homer.hdf5', 'r') as hdf:
        if hdf.get('homer') is not None:
            text_data = hdf['homer'][:]
            return text_data, chars, chars_map

print("Loading text data...")
_, chars, chars_map = load_data()
print("Done.")

print("Building model...")
num_neurons = 512
num_chars = 67
num_samples = 5000
batch_size = 100
test_size = 1
batch_length = 100
test_length = 1
input_size = 1
lr = 3.00
learning_rate = 10.0 ** -lr
temperature = 0.20
ckpt_dir = './checkpoints/rnn_lr={:.2f}.ckpt'.format(lr)

keep_pl = tf.placeholder(tf.float32)
cell = tf.nn.rnn_cell.BasicLSTMCell(num_neurons, state_is_tuple=True)
cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_pl)
cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 2, state_is_tuple=True)
weights = tf.Variable(tf.truncated_normal([num_neurons, num_chars], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[num_chars]))

"""SAMPLING: Run a single decoder input ([1 x 1 x 1]) through the RNN cell. From
   the output ([1 x 256]), sample a multinomial ([1 x 1]). Replace the input
   with the multinomial and repeat."""

test_pl = tf.placeholder(tf.float32, [test_size, test_length, input_size])
test_inputs = tf.unpack(test_pl, axis=1)
state_pl =  ((tf.placeholder(tf.float32, [test_size, num_neurons]), tf.placeholder(tf.float32, [test_size, num_neurons])),
             (tf.placeholder(tf.float32, [test_size, num_neurons]), tf.placeholder(tf.float32, [test_size, num_neurons])))

def multinomial(outputs, state, temperature):
    outputs = tf.reshape(tf.pack(outputs, axis=1), [test_size, num_neurons])
    logits = tf.matmul(outputs / temperature, weights) + bias
    return tf.multinomial(logits, 1), state

with tf.variable_scope('decoders') as scope:
    outputs_test, state = tf.nn.seq2seq.rnn_decoder(decoder_inputs=test_inputs,
                                                    initial_state=state_pl,
                                                    cell=cell)
    sample, state = multinomial(outputs_test, state, temperature)
print("Done.")

"""Sample a fixed number of characters and convert to human-readable format."""

print("Sampling... ")
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, ckpt_dir)

x = np.zeros((test_size, test_length, input_size), dtype=int)
zero_state = np.zeros((test_size, num_neurons))
s = ((zero_state, zero_state), (zero_state, zero_state))
text = chars[x.item()]
start = time.time()
for _ in range(0, num_samples):
    o, s = sess.run([sample, state], feed_dict={test_pl:x, state_pl:s, keep_pl: 1.00})  # [1 x 1]
    text += chars[o.item()]
    x = o.reshape((test_size, test_length, input_size))  # [1 x 1 x 1]
stop = time.time()
print("Done.\n")
print("SAMPLE={}\n".format(text))
print("(sampling time={:3f}).\n".format(stop - start))
