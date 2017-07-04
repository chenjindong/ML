#!/home/gdmlab/anaconda3/bin/python
#coding:utf-8

class Hyperparams:
    
    # data path
    source_data_path = 'data/letters_source.txt'
    target_data_path = 'data/letters_target.txt'
    checkpoint = 'model/trained_model'

    # training
    learning_rate = 0.001 
    batch_size = 128

    # model
    epochs = 60 # number of epoch
    rnn_size = 50   # number of neural in rnn cell
    num_layers = 2  # number of layer
    embedding_size = 15 # embedding size



