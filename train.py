# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 12:05:43 2019

@author: dell
"""
import os,time
import tensorflow as tf
import numpy as np
from random import randint
import data_util
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
np.set_printoptions(threshold=np.inf)  
flags = tf.app.flags
flags.DEFINE_integer("vocab_size", 2000, "Maximum size of vocabulary")
flags.DEFINE_integer("embedding_size", 100, "Embedding size of each token")
flags.DEFINE_integer("max_seq_length", 40, "size of each seq")
flags.DEFINE_integer("hidden_size",150, "lstm hidden Embedding size of each token")
flags.DEFINE_integer("num_layers", 3, "Number of lstm layers")
flags.DEFINE_integer("block_size", 5, "Size of each residual block")
flags.DEFINE_float('dropout', 0.5, 'dropout rate during training')
flags.DEFINE_integer("filter_h", 5, "Height of the CNN filter")
flags.DEFINE_integer("context_size", 20, "Length of sentence/context")
flags.DEFINE_integer("batch_size", 1, "Batch size of data while training")
flags.DEFINE_integer("iterations", 100000, "Number of iterations")
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
flags.DEFINE_string("openfile", "open3-200.txt", "Path to open")
flags.DEFINE_string("closefile", "colse3-200.txt", "Path to close")




def lstm_cell(hidden_size,keep_prob):
        cell = tf.contrib.rnn.LSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse)
        return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    
    
def main(_):
    conf = tf.flags.FLAGS
    tf.reset_default_graph()
    input_data=tf.placeholder(tf.int64,[conf.batch_size,conf.max_seq_length])
    labels=tf.placeholder(tf.int64,[conf.batch_size,conf.max_seq_length])
    word_to_idx,idx_to_word,label_to_idx, idx_to_label,seq_ids,labels_ids,words_embedding_index,labels_embedding_index=data_util.prepare_data(conf)
    print(len(word_to_idx))
    print(len(label_to_idx))
    print(len(idx_to_word))
    print(len(idx_to_label))
    embedding= tf.get_variable("embedding", [conf.vocab_size,conf.embedding_size],dtype=tf.float32)
    data=tf.nn.embedding_lookup(embedding,input_data)
    #data=tf.tensordot(data,embedding,1)
    #data=tf.reshape(data,[conf.batch_size,conf.max_seq_length,conf.embedding_size])
    #data=tf.cast(data, dtype=tf.float32)
    print(data)
    data_labels=tf.nn.embedding_lookup(labels_embedding_index,labels)
    print("placeholder compelete")
    lstm_Cell_fw = tf.contrib.rnn.MultiRNNCell([lstm_cell(conf.hidden_size,conf.dropout) for _ in range(conf.num_layers)],state_is_tuple=True)
    lstm_Cell_bw = tf.contrib.rnn.MultiRNNCell([lstm_cell(conf.hidden_size,conf.dropout) for _ in range(conf.num_layers)],state_is_tuple=True)
    value,_=tf.nn.bidirectional_dynamic_rnn(lstm_Cell_fw,lstm_Cell_bw,data, dtype=tf.float32)
    print(value)    
    print(tf.shape(value))
    value = tf.concat(value,axis=2)
    print(value)
    value=tf.cast(value, dtype=tf.float32)
    print(value)
    print(len(label_to_idx))
    weight= tf.get_variable("softmax_w", [conf.hidden_size * 6, len(label_to_idx)])
    bias = tf.get_variable("softmax_b", [len(label_to_idx)])
    value_mae=tf.slice(value,[0,0,0],[conf.batch_size,conf.max_seq_length-1,conf.hidden_size*2])
    value_ushiro=tf.slice(value,[0,1,0],[conf.batch_size,conf.max_seq_length-1,conf.hidden_size*2])
    value_mae=tf.concat([tf.zeros([conf.batch_size,1,conf.hidden_size*2],dtype=tf.float32),value_mae],1)
    value_ushiro=tf.concat([value_ushiro,tf.zeros([conf.batch_size,1,conf.hidden_size*2],dtype=tf.float32)],1)
    value = tf.concat([value_mae,value,value_ushiro],2)
    #value = tf.cast(value,tf.float32)
    #prediction=tf.nn.softmax((tf.tensordot(value,weight,1)+bias))
    prediction=tf.tensordot(value,weight,1)+bias
    prediction_softmax=tf.nn.softmax(prediction)
    word_pre=
    """
    softmax的部分会在下面的loss里由tf.nn.softmax........里完成）
    """
    print(value)
    print(prediction)
    guess=tf.argmax(prediction,2)
    correctPred=tf.equal(tf.argmax(prediction,2),labels)
    accuracy=tf.reduce_mean(tf.cast(correctPred,tf.float32))
    #tv = tf.trainable_variables()#得到所有可以训练的参数，即所有trainable=True 的tf.Variable/tf.get_variable
    #regularization_lost = 0.001* tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ]) #0.001是lambda超参数
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=data_labels))
    #l2_loss=loss+regularization_lost
    optimizer=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    #optimizer_l2=tf.train.AdamOptimizer(learning_rate=0.001).minimize(l2_loss)
    optimizer_null=tf.train.AdamOptimizer(learning_rate=0).minimize(loss)#用于dev
    sess=tf.Session()
    saver=tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    ckpt=tf.train.get_checkpoint_state(conf.ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess,ckpt.model_checkpoint_path)
        print(ckpt.model_checkpoint_path)
    else:
        print("checkpoint read flase")
    f = open(conf.closefile,'w')
    f2 = open(conf.openfile,'w')
    nextBatch,nextBatchLabels=data_util.getTrainBatch(labels_ids,seq_ids,conf);
    batch_total_error_count=0
    guess_total_error_count=0
    nicety_guess_error_count=0
    for i in range(conf.iterations):
        nextBatch,nextBatchLabels=data_util.getTrainBatch(labels_ids,seq_ids,conf);
        print(i)
        print(nextBatch)
        print("true label:")
        print(nextBatchLabels)
        time.sleep(5)
        now_prediction,now_loss,now_accuracy,now_guess,_=sess.run([prediction,loss,accuracy,guess,optimizer],{input_data:nextBatch,labels:nextBatchLabels})
            #    if(i+1)%checkpoint_steps==0:
                #        saver.save(sess,checkpoint_dir+'model.ckpt',global_step=i+1)

        print("now_guess:")
        print(now_guess)        
        print("now_loss:")
        print(now_loss)
        f.write(str(i))
        for a in now_guess:
            guess_line=[]
            for x in a:
                if(x!=0):
                    guess_line.append(idx_to_label[x])
            f.write(str(guess_line))
            f.write('\n')
            if(i%100==0):
                f2.write(str(i))
                f2.write(str(nextBatch))
                f2.write('\n')
                f2.write("true label:")
                f2.write(str(nextBatchLabels))
                f2.write("now_guess:")
                f2.write(str(now_guess))
                print("save")
        f.write('\n')
        



if __name__ == '__main__':
    tf.app.run()

        
