import sys
import codecs
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import reader

from sklearn.manifold import TSNE
 
 
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("model_dir", "", "The directory of the model file to evaluate.")
    
checkpoint_file = tf.train.latest_checkpoint(FLAGS.model_dir)  

with tf.Session() as sess:
    print('loading the embedding matrix from {}'.format(checkpoint_file))
    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
    saver.restore(sess, checkpoint_file)
    embedding_var = None
    for var in tf.all_variables():
        if var.name == 'Model/embedding:0':
            embedding_var = var
            break
    if not embedding_var:
        print("Couldn't find the embedding matrix!")
        exit(1)
    embedding = sess.run([embedding_var])[0]
    print('loading the training data to get the vocabulary...')
    raw_data = reader.ptb_raw_data('data')
    _, _, _, word_to_id = raw_data

    id_to_words = {id:word for word,id in word_to_id.iteritems()}
    words = [id_to_words[i] for i in sorted(id_to_words.keys())]

    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    tvals = tsne.fit_transform(embedding[:1000,:])
 
    plt.scatter(tvals[:, 0], tvals[:, 1])
    for label, x, y in zip(words[:1000], tvals[:, 0], tvals[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()
 
 
