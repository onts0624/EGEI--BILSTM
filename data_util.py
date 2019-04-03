# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:46:58 2019

@author: dell
"""
import time
import numpy as np
import collections
import os
from random import randint
def read_words(conf):
    words = []
    word_data=[]
    for file in os.listdir(conf.data_dir_seq):
        print(file)
        with open(os.path.join(conf.data_dir_seq, file), 'r') as f:
            for line in f.readlines():
                tokens = line.split()
                if(len(tokens)<conf.max_seq_length):
                    words.extend(tokens)
                    word_data.append(tokens)
                # NOTE Currently, only sentences with a fixed size are chosen
                # to account for fixed convolutional layer size.
                #if len(tokens) == conf.context_size-2:
                    #words.extend((['<pad>']*(conf.filter_h/2)) + ['<s>'] + tokens + ['</s>'])
    return words,word_data

def read_labels(conf):
    labels = []
    labels_data=[]
    for file in os.listdir(conf.data_dir_label):
        print(file)
        with open(os.path.join(conf.data_dir_label, file), 'r') as f:
            for line in f.readlines():
                tokens = line.split()
                if(len(tokens)<conf.max_seq_length):
                    labels.extend(tokens)
                    labels_data.append(tokens)
                #print(words)
                # NOTE Currently, only sentences with a fixed size are chosen
                # to account for fixed convolutional layer size.
                #if len(tokens) == conf.context_size-2:
                    #words.extend((['<pad>']*(conf.filter_h/2)) + ['<s>'] + tokens + ['</s>'])
    return labels,labels_data

def index_words(words, conf):
    word_counter = collections.Counter(words).most_common(conf.vocab_size-2)
    word_to_idx = {'<pad>': 0,'<unk>': 1}
    idx_to_word = {0: '<pad>',1: '<unk>'}
    for i,_ in enumerate(word_counter):
        word_to_idx[_[0]] = i+2
        idx_to_word[i+2] = _[0]
    #data = []
    #for word in words:
    #    idx = word_to_idx.get(word)
    #    idx = idx if idx else word_to_idx['<unk>']
    #    print(idx)
    #    data.append(idx)
    words_embedding_index=np.identity(conf.vocab_size)
    return word_to_idx, idx_to_word,words_embedding_index

def identity(length):
    return np.identity(length)
#单位矩阵函数

def index_labels(words, conf):
    label_counter = collections.Counter(words).most_common(conf.vocab_size-1)
    label_to_idx = {'<pad>': 0}
    idx_to_label = {0: '<pad>'}
    for i,_ in enumerate(label_counter):
        label_to_idx[_[0]] = i+1
        idx_to_label[i+1] = _[0]
    #data = []
    #for word in words:
    #    idx = word_to_idx.get(word)
    #    idx = idx if idx else word_to_idx['<unk>']
    #    print(idx)
    #    data.append(idx)
    labels_embedding_index=np.identity(len(label_to_idx))
    return label_to_idx, idx_to_label,labels_embedding_index

def get_batch(x_batches, y_batches, batch_idx):
    x, y = x_batches[batch_idx], y_batches[batch_idx]
    batch_idx += 1
    if batch_idx >= len(x_batches):
        batch_idx = 0
    return x, y.reshape(-1,1), batch_idx

def data_to_index(index,data,conf,source):
    line_num=np.array(data)
    print(line_num.shape)
    data_ids=np.zeros((line_num.shape[0], conf.max_seq_length), dtype='int32')
    if(source=="seq"):
        der=conf.data_dir_seq
    else:
        der=conf.data_dir_label
    data_line_counter=0
    f1 = open('word_test.txt','w')
    for file in os.listdir(der):
        with open(os.path.join(der, file), 'r') as f:
            for line in f.readlines():
                tokens = line.split()
                data_index_counter=0
                if(len(tokens)>conf.max_seq_length-1):
                    continue
                for word in tokens:
                    try:
                        f1.write(word+' ')
                        data_ids[data_line_counter][data_index_counter]=index[word]
                    except KeyError:
                        data_ids[data_line_counter][data_index_counter]=index['<unk>']
                    data_index_counter=data_index_counter+1
                data_line_counter=data_line_counter+1
                f1.write('\n')
            print(data_line_counter)
    return data_ids

def prepare_data(conf):
    words,data_words = read_words(conf)
    labels,data_labels = read_labels(conf)
    word_to_idx, idx_to_word,words_embedding_index = index_words(words, conf)
    label_to_idx, idx_to_label,labels_embedding_index = index_labels(labels, conf)
    seq_ids = np.zeros((len(data_words), conf.max_seq_length), dtype='int32')
    labels_ids = np.zeros((len(data_labels),conf.max_seq_length), dtype='int32')
    seq_ids=data_to_index(word_to_idx,data_words,conf,"seq")
    labels_ids=data_to_index(label_to_idx,data_labels,conf,"label")
    return word_to_idx,idx_to_word,label_to_idx, idx_to_label,seq_ids,labels_ids,words_embedding_index,labels_embedding_index
    
def del_all_flags():
    FLAGS = tf.flags.FLAGS #FLAGS保存命令行参数的数据
    FLAGS._parse_flags() #将其解析成字典存储到FLAGS.__flags中
    keys_list = [keys for keys in FLAGS.__flags]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

def getTrainBatch(labels_ids,seq_ids,conf):
    labels=np.zeros([conf.batch_size,conf.max_seq_length])
    arr=np.zeros([conf.batch_size,conf.max_seq_length])
    for i in range(conf.batch_size):
            num=randint(2*conf.batch_size,labels_ids.shape[0])
            labels[i]=labels_ids[num-1:num]
            arr[i]=seq_ids[num-1:num]
    arr=arr.astype(np.int)
    labels=labels.astype(np.int)
    return arr,labels

def getTestBatch(labels_ids,seq_ids,conf):
    labels=np.zeros([conf.batch_size,conf.max_seq_length])
    arr=np.zeros([conf.batch_size,conf.max_seq_length])
    for i in range(conf.batch_size):
            num=i+1
            labels[i]=labels_ids[num-1:num]
            arr[i]=seq_ids[num-1:num]
    arr=arr.astype(np.int)
    labels=labels.astype(np.int)
    return arr,labels


if __name__ == '__main__':
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    np.set_printoptions(threshold=np.inf)  
    flags = tf.app.flags
    #del_all_flags()
    flags.DEFINE_integer("vocab_size", 2000, "Maximum size of vocabulary")
    flags.DEFINE_integer("embedding_size", 200, "Embedding size of each token")
    flags.DEFINE_integer("max_seq_length",40, "Embedding size of each token")
    flags.DEFINE_integer("filter_size", 64, "Depth of each CNN layer")
    flags.DEFINE_integer("num_layers", 10, "Number of CNN layers")
    flags.DEFINE_integer("block_size", 5, "Size of each residual block")
    flags.DEFINE_integer("filter_h", 5, "Height of the CNN filter")
    flags.DEFINE_integer("context_size", 20, "Length of sentence/context")
    flags.DEFINE_integer("batch_size", 1, "Batch size of data while training")
    flags.DEFINE_integer("epochs", 50, "Number of epochs")
    flags.DEFINE_integer("num_sampled", 1, "Sampling value for NCE loss")
    flags.DEFINE_integer("learning_rate", 1.0, "Learning rate for training")
    flags.DEFINE_integer("momentum", 0.99, "Nestrov Momentum value")
    flags.DEFINE_integer("grad_clip", 0.1, "Gradient Clipping limit")
    flags.DEFINE_integer("num_batches", 0, "Predefined: to be calculated")
    flags.DEFINE_string("ckpt_path", "ckpt", "Path to store checkpoints")
    flags.DEFINE_string("summary_path", "logs", "Path to store summaries")
    #flags.DEFINE_string("data_dir", "data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled", "Path to store data")
    flags.DEFINE_string("data_dir_seq", "data/seq", "Path to store data of seq")
    flags.DEFINE_string("data_dir_label", "data/label", "Path to store data of label")
    flags.DEFINE_integer("iterations", 100000, "Number of iterations")
    print("flags complete")
    FLAGS = tf.flags.FLAGS
    word_to_idx,idx_to_word,label_to_idx, idx_to_label,seq_ids,labels_ids,words_embedding_index,labels_embedding_index=prepare_data(FLAGS)
    print(seq_ids.shape)
    print(len(labels_ids))
    #for i in range(FLAGS.iterations):
      #  nextBatch,nextBatchLabels=getTrainBatch(labels_ids,seq_ids,FLAGS);
       # print(nextBatch)
       # print(nextBatchLabels)
    