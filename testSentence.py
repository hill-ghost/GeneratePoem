#-*- coding: UTF-8 -*-
import collections
import numpy as np
import tensorflow as tf
import pickle
import os
import re
from topic import *

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

def getKeyword():
    category, topics, keywords = getCategory_Topics_Keywords()
    return random_keywords(keywords[0][0])#这里是模拟，后期要更改

def getFirstSentence():
    return gen_firstSentence()

poems = read_poems(PATH5)
print('共有%s个句子 ' % len(poems))


words = read_id2word_map()
word2id_map = read_word2id_map()

# 把句子中的字索引成id的形式
to_num = lambda word: word2id_map.get(word, len(words))
poems_vector = [list(map(to_num, poem)) for poem in poems]


# 每次取1首诗进行训练
batch_size = 1
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

# RNN

# 输入占位符
input_data = tf.placeholder(tf.int32, [batch_size, None])
# 输出占位符
output_targets = tf.placeholder(tf.int32, [batch_size, None])
# 定义RNN
def neural_network(model='lstm', rnn_size=128, num_layers=2):
    if model == 'rnn':
        cell_fun = tf.contrib.rnn.BasicRNNCell
    elif model == 'gru':
        cell_fun = tf.contrib.rnn.GRUCell
    elif model == 'lstm':
        cell_fun = tf.contrib.rnn.BasicLSTMCell

    cell = cell_fun(rnn_size, state_is_tuple = True)
    #实例化由两层网络构成的递归神经网络
    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple = True)
    #初始化状态，全设为0
    initial_state = cell.zero_state(batch_size, tf.float32)
    #定义权重参数
    with tf.variable_scope('rnnlm'):
        softmax_w = tf.get_variable("softmax_w", [rnn_size, len(words)])
        softmax_b = tf.get_variable("softmax_b", [len(words)])
        embedding = tf.get_variable("embedding", [len(words), rnn_size])#随机初始化向量矩阵
        inputs = tf.nn.embedding_lookup(embedding, input_data)#查找

    #递归神经网络每个细胞的输出
    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
    output = tf.reshape(outputs,[-1, rnn_size])

    #模型实际输出
    logits = tf.matmul(output, softmax_w) + softmax_b
    probs = tf.nn.softmax(logits)
    return logits, last_state, probs, cell, initial_state

# 生成古诗

# 使用训练完成的模型
def gen_firstSentence():
    def to_word(weights):
        t = np.cumsum(weights)#当前元素是前面所有元素的累加
        s = np.sum(weights)#累加所有元素
        m = np.random.rand(1)*s
        sample = int(np.searchsorted(t, m))#二分查找法找出随机数在t中的位置
        return words[sample]

    _, last_state, probs, cell, initial_state = neural_network()
    #允许自动分配设备
    Session_config = tf.ConfigProto(allow_soft_placement = True)
    # 使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加
    Session_config.gpu_options.allow_growth = True

    with tf.Session(config = Session_config) as sess:
        sess.run(tf.global_variables_initializer())
        #创建saver
        list = [softmax_w,softmax_b]
        saver = tf.train.Saver(list)
        #saver.restore(sess, 'model/poetry.module-99')
        #从checkpoints文件中恢复变量
        ckpt = tf.train.get_checkpoint_state('./sentence_model/')
        checkpoint_suffix = ""
        if tf.__version__ > "0.12":
            checkpoint_suffix = ".index"
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path + checkpoint_suffix):
            #print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            #恢复变量
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            return None

        state_ = sess.run(cell.zero_state(1, tf.float32))
        #开始生成诗歌
        x = np.array([list(map(word2id_map.get, '['))])
        [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
        word = to_word(probs_)
        #word = words[np.argmax(probs_)]
        poem = getKeyword()
        for word in poem:
            x = np.zeros((1,1))#生成一行一列零矩阵
            x[0,0] = word2id_map.get(word,len(words)-1)
            [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
            word = to_word(probs_)

        while word != ']':
            poem += word
            x = np.zeros((1,1))#生成一行一列零矩阵
            x[0,0] = word2id_map.get(word,len(words)-1)
            [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
            word = words[np.argmax(probs_)]
        return poem

if __name__ == '__main__':
    print(gen_firstSentence())

