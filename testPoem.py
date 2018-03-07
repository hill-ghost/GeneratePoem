#-*- coding: UTF-8 -*-
import tensorflow as tf
import pickle
import numpy as np

def word2id(sentence):
    word2id_map = None
    with open('data\\word2id_map.pkl', 'rb') as f:
        word2id_map = pickle.load(f)
    ids = []
    for word in sentence:
        ids.append(word2id_map[word])
    return ids

def id2word(ids):
    words = None
    with open('data\\id2word_map.pkl', 'rb') as f:
        words = pickle.load(f)
    sentence = []
    for one in ids:
        sentence.append(words[one])
    return sentence

def str2list(sentence):
    list = []
    for word in sentence:
        list.append(word)
    list.append("_")
    list.append("_")
    print(list)
    return list

def genNextSentence(sentence):
    batch_size = 1
    sequence_length = 7
    num_encoder_symbols = 2943
    num_decoder_symbols = 2943
    embedding_size = 256
    hidden_size = 256
    num_layers = 2

    encoder_inputs = tf.placeholder(dtype=tf.int32, shape=[batch_size, sequence_length])
    decoder_inputs = tf.placeholder(dtype=tf.int32, shape=[batch_size, sequence_length])

    targets = tf.placeholder(dtype=tf.int32, shape=[batch_size, sequence_length])
    weights = tf.placeholder(dtype=tf.float32, shape=[batch_size, sequence_length])

    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

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

    saver = tf.train.Saver()
    with tf.Session() as sess:
        module_file = tf.train.latest_checkpoint('./poem_model/')
        saver.restore(sess, module_file)
        encoder_input = word2id(str2list(sentence))

        encoder_input = np.asarray([np.asarray(encoder_input)])
        decoder_input = np.zeros([1, sequence_length])
        print('encoder_input : ', encoder_input)
        print('decoder_input : ', decoder_input)
        pred_value = sess.run(pred, feed_dict={encoder_inputs: encoder_input, decoder_inputs: decoder_input})
        print(pred_value)
        sentence = id2word(pred_value[0])
        print(sentence)
    return sentence
