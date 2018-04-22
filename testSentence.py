#-*- coding: UTF-8 -*-
import collections
import numpy as np
import tensorflow as tf
import pickle
import os
import re
from tools import *

RESTORE_PATH5 =r'./sentence_model/poem5/'
RESTORE_PATH7 =r'./sentence_model/poem7/'

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

def getKeyword(row, col):
    category, topics, keywords = getCategory_Topics_Keywords()
    return random_keywords(keywords[row][col])#这里是模拟，后期要更改

def to_word(weights):
    t = np.cumsum(weights)#当前元素是前面所有元素的累加
    s = np.sum(weights)#累加所有元素
    m = np.random.rand(1)*s
    sample = int(np.searchsorted(t, m))#二分查找法找出随机数在t中的位置
    return words[sample]


words = read_id2word_map()
word2id_map = read_word2id_map()

# 每次取1首诗进行训练
batch_size = 1

# RNN
g1 = tf.Graph()
with g1.as_default():
    # 输入占位符
    input_data = tf.placeholder(tf.int32, [batch_size, None])
    # 输出占位符
    output_targets = tf.placeholder(tf.int32, [batch_size, None])
# 定义RNN
def neural_network(model='lstm', rnn_size=128, num_layers=2):
    with g1.as_default():
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
        with tf.variable_scope('rnnlm',reuse=tf.AUTO_REUSE):
            softmax_w = tf.get_variable("softmax_w", [rnn_size, len(words)])
            softmax_b = tf.get_variable("softmax_b", [len(words)])
            embedding = tf.get_variable("embedding", [len(words), rnn_size])#随机初始化向量矩阵
            inputs = tf.nn.embedding_lookup(embedding, input_data)#查找
        with tf.variable_scope('',reuse=tf.AUTO_REUSE):
            #递归神经网络每个细胞的输出
            outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
        output = tf.reshape(outputs,[-1, rnn_size])

        #模型实际输出
        logits = tf.matmul(output, softmax_w) + softmax_b
        probs = tf.nn.softmax(logits)
    return logits, last_state, probs, cell, initial_state

# 使用训练完成的模型
def gen_firstSentence(word_num,row,col):
    if word_num == 7:
        restore_path = RESTORE_PATH7
    else:
        restore_path = RESTORE_PATH5

    _, last_state, probs, cell, initial_state = neural_network()
    #允许自动分配设备
    Session_config = tf.ConfigProto(allow_soft_placement = True)
    # 使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加
    Session_config.gpu_options.allow_growth = True

    with tf.Session(graph = g1, config = Session_config) as sess:
        sess.run(tf.global_variables_initializer())
        #创建saver
        saver = tf.train.Saver()
        #从checkpoints文件中恢复变量
        ckpt = tf.train.get_checkpoint_state(restore_path)
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
        poem = getKeyword(row, col)
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
    print(gen_firstSentence(5, 0, 0))
    print(gen_firstSentence(7, 0, 0))

