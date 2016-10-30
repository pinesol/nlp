
import tensorflow as tf
import reader
from scipy import spatial

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

    # The unkwown word index
    unk = word_to_id['<unk>']
    
    def similarity(word1, word2):
        e1 = embedding[word_to_id.get(word1, unk)]
        e2 = embedding[word_to_id.get(word2, unk)]
        sim = 1 - spatial.distance.cosine(e1, e2)
        print("similarity({}, {}) = {}".format(word1, word2, sim))
        return sim

    score = 0
    score += similarity('a', 'an') > similarity('a', 'document')
    score += similarity('in', 'of') > similarity('in', 'picture')
    score += similarity('nation', 'country') > similarity('nation', 'end')
    score += similarity('films', 'movies') > similarity('films', 'almost')
    score += similarity('workers', 'employees') > similarity('workers', 'movies')
    score += similarity('institutions', 'organizations') > similarity('institutions', 'big')
    score += similarity('assets', 'portfolio') > similarity('assets', 'down')
    score += similarity("'", ',') > similarity("'", 'quite')
    score += similarity('finance', 'acquisition') > similarity('finance', 'seems')
    score += similarity('good', 'great') > similarity('good', 'minutes')
    print('Score: {}/10'.format(score))
