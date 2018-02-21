#-*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
import collections
import pickle
import os
import re

path1 =r'C:\Users\Pan\Desktop\poem\data\jueju5_utf8'
path2 =r'C:\Users\Pan\Desktop\poem\data\jueju7_utf8'

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
        target_sen.append(sum_sen[1]+']'+'_')
        last_sen.append(sum_sen[2]+'_'+'_')
        next_sen.append('['+sum_sen[3]+']')
        target_sen.append(sum_sen[1]+']'+'_')
        
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
    poems,last_sen,next_sen,target_sen = read_poems(path1)

    words,word2id_map = gen_dict(poems)
    last_sen_id = word2id(last_sen,words,word2id_map)
    next_sen_id = word2id(next_sen,words,word2id_map)
    target_sen_id = word2id(target_sen,words,word2id_map)
    data_len = len(last_sen_id)
    print('共有%d个序对'%(data_len))
    last_id = np.zeros([data_len, 7], dtype=np.int32)
    next_id = np.zeros([data_len, 7], dtype=np.int32)
    target_id = np.zeros([data_len, 7], dtype=np.int32)

    for i in range(data_len):
        last_id[i] = np.array(last_sen_id[i])
        next_id[i] = np.array(next_sen_id[i])
        target_id[i] = np.array(target_sen_id[i])

    np.save('last_id.npy',last_id)
    np.save('next_id.npy',next_id)
    np.save('target_id.npy',target_id)
    
if __name__ == '__main__':
    preprocess_data()
    
