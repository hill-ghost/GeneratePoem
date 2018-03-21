#-*- coding: UTF-8 -*-
import tensorflow as tf
import pickle
import numpy as np

RESTORE_PATH5 =r'./poem_model/poem5/'
RESTORE_PATH7 =r'./poem_model/poem7/'
 
def word2id(sentence):
    word2id_map = None
    with open('data\\word2id_map.pkl', 'rb') as f:
        word2id_map = pickle.load(f)
    ids = []
    for word in sentence:
        ids.append(word2id_map.get(word,len(word2id_map)-1))
    return ids

def id2word(ids):
    words = None
    with open('data\\id2word_map.pkl', 'rb') as f:
        words = pickle.load(f)
    sentence = []
    for one in ids:
        sentence.append(words[one])
    return sentence

def str2list(sentence,word_num):
    list = []
    for word in sentence:
        list.append(word)
    while len(list) < word_num+2:
        list.append("_")
        list.append("_")
    print(list)
    return list

def genNextSentence(sentence,word_num):
    batch_size = 1
    num_encoder_symbols = 4477
    num_decoder_symbols = 4477
    embedding_size = 256
    hidden_size = 256
    num_layers = 2

    if word_num == 5:
        restore_path = RESTORE_PATH5
        sequence_length = 7
    else:
        restore_path = RESTORE_PATH7
        sequence_length = 9

    encoder_inputs = tf.placeholder(dtype=tf.int32, shape=[batch_size, sequence_length])
    decoder_inputs = tf.placeholder(dtype=tf.int32, shape=[batch_size, sequence_length])

    targets = tf.placeholder(dtype=tf.int32, shape=[batch_size, sequence_length])
    weights = tf.placeholder(dtype=tf.float32, shape=[batch_size, sequence_length])

    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

    with tf.variable_scope('',reuse=tf.AUTO_REUSE):
        results, states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
            tf.unstack(encoder_inputs, axis=1),
            tf.unstack(decoder_inputs, axis=1),
            cell,
            num_encoder_symbols,
            num_decoder_symbols,
            embedding_size,
            feed_previous=True,
        )
    logits = tf.stack(results, axis=1)
    pred = tf.argmax(logits, axis=2)
    poem_result = []
    saver = tf.train.Saver()
    with tf.Session() as sess:
        module_file = tf.train.latest_checkpoint(restore_path)
        saver.restore(sess, module_file)
        lastSentence = str2list(sentence,word_num)
        poem_result.append(lastSentence[0:word_num])
        for i in range(3):
            encoder_input = word2id(lastSentence)
            encoder_input = np.asarray([np.asarray(encoder_input)])
            decoder_input = np.zeros([1, sequence_length])
            pred_value = sess.run(pred, feed_dict={encoder_inputs: encoder_input, decoder_inputs: decoder_input})
            sentence = id2word(pred_value[0])
            lastSentence = sentence
            if word_num == 5:
                lastSentence[5] = '_'
            else:
                lastSentence[7] = '_'
            print(sentence)
            poem_result.append(sentence[0:word_num])
    return poem_result
