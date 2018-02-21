#-*- coding: UTF-8 -*-
import collections
import numpy as np
import tensorflow as tf
import pickle
import os
import re

PATH5 =r'C:\Users\Pan\Desktop\poem\data\jueju5_utf8'
PATH7 =r'C:\Users\Pan\Desktop\poem\data\jueju7_utf8'

#将所有诗切分成句子
def read_poems(path):
    files = os.listdir(path)
    poems = []
    for file in files:
        poem_path = path + '\\' + file
        with open(poem_path, "r", encoding = 'utf-8') as f:
            for line in f:
                try:
                    line = line.strip(u'\n')
                    temp = re.split('[，。？！]',line)[:-1]
                    for sen in temp:
                        poems.append('['+sen+']')
                except Exception as e:
                    pass     
    return poems

#返回一个list，包含所有的字
def read_id2word_map():
    words = None
    with open('data\\id2word_map.pkl', 'rb') as f:
        words = pickle.load(f)
    return words

#返回一个dict
def read_word2id_map():
    word2id_map = None
    with open('data\\word2id_map.pkl', 'rb') as f:
        word2id_map = pickle.load(f)
    return word2id_map

poems = read_poems(PATH5)
print('共有%s个句子 ' % len(poems))


words = read_id2word_map()
word2id_map = read_word2id_map()

# 把句子中的字索引成id的形式
to_num = lambda word: word2id_map.get(word, len(words))
poems_vector = [list(map(to_num, poem)) for poem in poems]


# 每次取64首诗进行训练
batch_size = 64
# 1次epoch需要训练n_chunk次
n_chunk = len(poems_vector) // batch_size

class DataSet(object):
    def __init__(self, data_size):
        self._data_size = data_size
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._data_index = np.arange(data_size)

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        if start + batch_size > self._data_size:
            np.random.shuffle(self._data_index)
            self._epochs_completed = self._epochs_completed + 1
            self._index_in_epoch = batch_size
            full_batch_features, full_batch_labels = self.data_batch(0, batch_size)
            return full_batch_features, full_batch_labels
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            full_batch_features ,full_batch_labels = self.data_batch(start, end)
            if self._index_in_epoch == self._data_size:
                self._index_in_epoch = 0
                self._epochs_completed = self._epochs_completed + 1
                np.random.shuffle(self._data_index)
            return full_batch_features,full_batch_labels

    def data_batch(self,start,end):
        batches = []
        for i in range(start,end):
            batches.append(poems_vector[self._data_index[i]])

        length = max(map(len,batches))

        xdata = np.full((end - start, length), len(words), np.int32)
        for row in range(end - start):
            xdata[row,:len(batches[row])] = batches[row]
        ydata = np.copy(xdata)
        ydata[:,:-1] = xdata[:, 1:]
        return xdata,ydata

#---------------------------------------RNN--------------------------------------#
input_data = tf.placeholder(tf.int32, [batch_size, None])
output_targets = tf.placeholder(tf.int32, [batch_size, None])
# 定义RNN
def neural_network(model = 'lstm', rnn_size = 128, num_layers = 2):
    if model == 'rnn':
        cell_fun = tf.contrib.rnn.BasicRNNCell
    elif model == 'gru':
        cell_fun = tf.contrib.rnn.GRUCell
    elif model == 'lstm':
        cell_fun = tf.contrib.rnn.BasicLSTMCell
    cell = cell_fun(rnn_size, state_is_tuple = True)
    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple = True)
    initial_state = cell.zero_state(batch_size, tf.float32)
    with tf.variable_scope('rnnlm'):
        softmax_w = tf.get_variable("softmax_w", [rnn_size, len(words)])
        softmax_b = tf.get_variable("softmax_b", [len(words)])
        embedding = tf.get_variable("embedding", [len(words), rnn_size])
        inputs = tf.nn.embedding_lookup(embedding, input_data)
    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state = initial_state, scope = 'rnnlm')
    output = tf.reshape(outputs, [-1, rnn_size])
    logits = tf.matmul(output, softmax_w) + softmax_b
    probs = tf.nn.softmax(logits)
    return logits, last_state, probs, cell, initial_state

def load_model(sess, saver, ckpt_path):
    latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
    if latest_ckpt:
        print ('resume from', latest_ckpt)
        saver.restore(sess, latest_ckpt)
        return int(latest_ckpt[latest_ckpt.rindex('-') + 1:])
    else:
        print ('building model from scratch')
        sess.run(tf.global_variables_initializer())
        return -1

#训练
def train_neural_network():
    logits, last_state, _, _, _ = neural_network()
    targets = tf.reshape(output_targets, [-1])
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets], \
        [tf.ones_like(targets, dtype = tf.float32)], len(words))
    cost = tf.reduce_mean(loss)
    learning_rate = tf.Variable(0.0, trainable = False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    Session_config = tf.ConfigProto(allow_soft_placement = True)
    Session_config.gpu_options.allow_growth = True

    trainds = DataSet(len(poems_vector))

    with tf.Session(config = Session_config) as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables())
        last_epoch = load_model(sess, saver, 'sentence_model/')

        for epoch in range(last_epoch + 1, 101):
            sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** epoch)))
            #sess.run(tf.assign(learning_rate, 0.01))

            all_loss = 0.0
            for batche in range(n_chunk):
                x,y = trainds.next_batch(batch_size)
                train_loss, _, _ = sess.run([cost, last_state, train_op], feed_dict={input_data: x, output_targets: y})

                all_loss = all_loss + train_loss

                if batche % 50 == 1:
                    print(epoch, batche, 0.002 * (0.97 ** epoch),train_loss)
            if epoch % 20 == 0:
                saver.save(sess, 'sentence_model/poetry.module', global_step = epoch)
            print (epoch,' Loss: ', all_loss * 1.0 / n_chunk)

train_neural_network()
