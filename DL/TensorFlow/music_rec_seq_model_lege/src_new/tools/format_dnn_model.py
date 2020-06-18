# -*- coding: utf-8 -*-
# @File    : format_dnn_model.py
# @Author  : letian
# @Mail    : @.com
# @Time    : 2019/12/3 11:30


import sys
import tensorflow as tf
import numpy as np


def dump_embedding(out_path, model, layer_name):
    embedding_layer = model.get_layer(layer_name)
    weights = embedding_layer.get_weights()

    np.savetxt(out_path, weights[0], fmt='%.6f')


def redefine_model(out_path, model, layer_name):
    layer = model.get_layer(layer_name)
    print(layer.get_weights())

    model_export = tf.keras.Model(inputs=model.input, outputs=layer)

    tf.saved_model.save(model_export, out_path)


def print_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items()) == 0:
            return

        for layer, g in f.items():
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for p_name in g.keys():
                param = g[p_name]
                print("      {}: {}".format(p_name, param.shape))
    finally:
        f.close()


def print_structure_2(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items()) == 0:
            return

        for layer, g in f.items():
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for p_name in g.keys():
                param = g[p_name]
                subkeys = param.keys()
                for k_name in param.keys():
                    print("      {}/{}: {}".format(p_name, k_name, param.get(k_name)[:]))
    finally:
        f.close()


if __name__ == "__main__":
    # seq_dnn_model = tf.keras.models.load_model(sys.argv[1])
    seq_dnn_model = tf.saved_model.load(sys.argv[1])
    #np.set_printoptions(threshold=np.inf)
    embedding_list = seq_dnn_model.embedding_label.embeddings.numpy().tolist()

    with open(sys.argv[2], "w+") as out_file:
        for embedding in embedding_list:
            out_file.write(" ".join([str(v) for v in embedding]) + "\n")

    print("done")

    #print(seq_dnn_model.summary())

    #import h5py

    #f = h5py.File(sys.argv[1], 'r')

    #d = f['embedding_8']

    #print(list(f.keys()))

    #print_structure_2(sys.argv[1])

    #dump_embedding(sys.argv[2], seq_dnn_model, "embedding_8")
    # redefine_model(sys.argv[2], seq_dnn_model, "dense_1")
