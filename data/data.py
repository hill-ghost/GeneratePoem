#-*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
import collections
import pickle
import os
import re

PATH5 =r'C:\Users\Pan\Desktop\poem\data\jueju5_utf8'
PATH7 =r'C:\Users\Pan\Desktop\poem\data\jueju7_utf8'

#获得序对
def read_poems(path):
    files = os.listdir(path)
    poems = []
    sum_sens = []
    for file in files:
        poem_path = path + '\\' + file
        with open(poem_path, "r", encoding = 'utf-8') as f:
            for line in f:
                try:
                    line = line.strip(u'\n')
                    temp = re.split('[，。？！]',line)[:-1]
                    poems.append(line)
                    sum_sens.append(temp)
                except Exception as e:
                    pass
    last_sen = []
    next_sen = []
    target_sen = []
    print('共有%d首诗'%(len(sum_sens)))
    for i in range(len(sum_sens)):
        sum_sen = sum_sens[i]
        last_sen.append(sum_sen[0]+'_'+'_')
        next_sen.append('['+sum_sen[1]+']')
        target_sen.append(sum_sen[1]+']'+'_')

        last_sen.append(sum_sen[1]+'_'+'_')
        next_sen.append('['+sum_sen[2]+']')
        target_sen.append(sum_sen[2]+']'+'_')

        last_sen.append(sum_sen[2]+'_'+'_')
        next_sen.append('['+sum_sen[3]+']')
        target_sen.append(sum_sen[3]+']'+'_')
        
    return poems,last_sen,next_sen,target_sen

def gen_dict(poems):
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    all_words.append(']')
    all_words.append('[')
    all_words.append('_')
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x:-x[1])#counter.items()返回的是一个list[字符,字符统计次数]
    words,_ = zip(*count_pairs)
    words = words[:len(words)] + ('',)
    print('共有%d个不同的字'%(len(words)))
    with open('id2word_map.pkl', 'wb') as f:
        pickle.dump(words, f)
    word2id_map = dict(zip(words, range((len(words)))))
    with open('word2id_map.pkl', 'wb') as f:
        pickle.dump(word2id_map, f)
    return words,word2id_map

def word2id(sentences,words,word2id_map):
    to_num = lambda word: word2id_map.get(word, len(words))
    poems_id = [list(map(to_num, sentence)) for sentence in sentences]
    return poems_id  
    
def preprocess_data():
    poems5,last_sen5,next_sen5,target_sen5 = read_poems(PATH5)
    poems7,last_sen7,next_sen7,target_sen7 = read_poems(PATH7)

    poems = poems5 + poems7
    words,word2id_map = gen_dict(poems)

    last_sen_id5 = word2id(last_sen5,words,word2id_map)
    next_sen_id5 = word2id(next_sen5,words,word2id_map)
    target_sen_id5 = word2id(target_sen5,words,word2id_map)

    last_sen_id7 = word2id(last_sen7,words,word2id_map)
    next_sen_id7 = word2id(next_sen7,words,word2id_map)
    target_sen_id7 = word2id(target_sen7,words,word2id_map)

    data_len5 = len(last_sen_id5)
    print('五言绝句共有%d个序对'%(data_len5))

    data_len7 = len(last_sen_id7)
    print('七言绝句共有%d个序对'%(data_len7))

    last_id5 = np.zeros([data_len5, 7], dtype=np.int32)
    next_id5 = np.zeros([data_len5, 7], dtype=np.int32)
    target_id5 = np.zeros([data_len5, 7], dtype=np.int32)

    last_id7 = np.zeros([data_len7, 9], dtype=np.int32)
    next_id7 = np.zeros([data_len7, 9], dtype=np.int32)
    target_id7 = np.zeros([data_len7, 9], dtype=np.int32)

    for i in range(data_len5):
        last_id5[i] = np.array(last_sen_id5[i])
        next_id5[i] = np.array(next_sen_id5[i])
        target_id5[i] = np.array(target_sen_id5[i])

    for i in range(data_len7):
        last_id7[i] = np.array(last_sen_id7[i])
        next_id7[i] = np.array(next_sen_id7[i])
        target_id7[i] = np.array(target_sen_id7[i])

    np.save('last_id5.npy',last_id5)
    np.save('next_id5.npy',next_id5)
    np.save('target_id5.npy',target_id5)

    np.save('last_id7.npy',last_id7)
    np.save('next_id7.npy',next_id7)
    np.save('target_id7.npy',target_id7)
    
if __name__ == '__main__':
    preprocess_data()
