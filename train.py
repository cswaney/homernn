import tensorflow as tf
import numpy as np
import h5py as h5
import collections
import time
import sys

# NOTE: Basically the same approach works for word-level, but need embedding...

# TODO: batch test output: sample N characters at a time...
# TODO: make sure data keeps spaces between words, after sentences...
# TODO: use FLAGS or args to set the model parameters, etc...
# TODO: "..." and "done" on the same line?

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

class TextData():

    def __init__(self, data, chars, chars_map):
        self.data = data

    def get_batch(self, batch_size, as_chars=False):
        idx = np.random.randint(low=0, high=len(self.data), size=batch_size)
        codes = self.data[idx][:, :-1].reshape(batch_size, self.data.shape[1] - 1, 1)
        labels = self.data[idx][:, 1:].reshape(batch_size, self.data.shape[1] - 1)
        if as_chars:
            codes_chars = []
            for line in codes:
                codes_chars.append([chars[c] for c in line.ravel()])
            labels_chars = []
            for line in labels:
                labels_chars.append([chars[c] for c in line.ravel()])
            return codes_chars, labels_chars
        else:
            return codes, labels

print("Loading text data...")
data, chars, chars_map = load_data()
text_data = TextData(data, chars, chars_map)
print("Done.")

print("Building model...")
num_neurons = 512
num_chars = 67
num_samples = 100
batch_size = 100
test_size = 1
batch_length = 100
test_length = 1
input_size = 1
lr = float(sys.argv[1])
learning_rate = 10.0 ** -lr
keep_prob = 0.50
max_steps = 30000
print_steps = 100
save_steps = 100
temperature = 1.00
ckpt_dir = './checkpoints/rnn_lr={:.2f}.ckpt'.format(lr)
log_dir = './logs/rnn_lr_{:.2f}'.format(lr)

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
state_pl =  ((tf.placeholder(tf.float32, [test_size, num_neurons]),
              tf.placeholder(tf.float32, [test_size, num_neurons])),
             (tf.placeholder(tf.float32, [test_size, num_neurons]),
              tf.placeholder(tf.float32, [test_size, num_neurons])))

def multinomial(outputs, state, temperature):
    outputs = tf.reshape(tf.pack(outputs, axis=1), [test_size, num_neurons])
    logits = tf.matmul(outputs / temperature, weights) + bias
    return tf.multinomial(logits, 1), state

with tf.variable_scope('decoders') as scope:
    outputs_test, state = tf.nn.seq2seq.rnn_decoder(decoder_inputs=test_inputs,
                                                    initial_state=state_pl,
                                                    cell=cell)
    sample, state = multinomial(outputs_test, state, temperature)

"""TRAINING: Train a RNN cell from decoder input batches ([100 x 100 x 1]). The
   model shares parameters across testing and training. Training outputs aren't
   called during testing, and vice-a-versa."""

train_pl = tf.placeholder(tf.float32, [batch_size, batch_length, input_size])
train_inputs = tf.unpack(train_pl, axis=1)
zero_state_train = cell.zero_state(batch_size, tf.float32)
labels_pl = tf.placeholder(tf.int64, [batch_size, batch_length])

def xentropy(outputs, labels):
    labels = tf.reshape(labels, (-1,))
    outputs = tf.reshape(tf.pack(outputs, axis=1), (-1, num_neurons))
    logits = tf.matmul(outputs, weights) + bias
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
    loss = tf.reduce_mean(losses)
    tf.scalar_summary('mean_xentropy', loss)
    return tf.reduce_mean(losses)

def train(loss, learning_rate):
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    return optimizer.minimize(loss)

with tf.variable_scope('decoders', reuse=True) as scope:
    outputs_train, _ = tf.nn.seq2seq.rnn_decoder(decoder_inputs=train_inputs,
                                                 initial_state=zero_state_train,
                                                 cell=cell)
    loss = xentropy(outputs_train, labels_pl)
    train_op = train(loss, learning_rate)
init_op = tf.initialize_all_variables()
summarizer = tf.merge_all_summaries()
print("Done.")

print("Beginning training...")
sess = tf.Session()
writer = tf.train.SummaryWriter(log_dir, sess.graph)
saver = tf.train.Saver()
sess.run(init_op)
steps = 0
while (steps < max_steps):
    x, y = text_data.get_batch(batch_size)
    start = time.time()
    feed_dict = {train_pl:x, labels_pl:y, keep_pl:keep_prob}
    _, batch_loss, summary = sess.run([train_op, loss, summarizer], feed_dict=feed_dict)
    stop = time.time()
    if (steps == 0):
        min_loss = batch_loss
    if (steps % print_steps == 0) or (steps == 0):
        print('batch={:d}, batch_loss={:.2f}, batch_time={:.4f}s'.format(steps, batch_loss, stop - start))
        writer.add_summary(summary, steps)
        writer.flush()
    if (steps % save_steps == 0) and (batch_loss < min_loss):
        min_loss = batch_loss
        print("new best loss achieved: saving model checkpoint to {}".format(ckpt_dir))
        saver.save(sess, ckpt_dir)
    steps += 1
print("Done.")
