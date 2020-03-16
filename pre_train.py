import tensorflow as tf

from model import SSLNLI
from tqdm import tqdm
from data_load import get_batch, process_file_snli
from utils import save_variable_specs
import os
from hparams import Hparams
import math
import random
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import pickle

def evaluate(sess, eval_features):
    total_loss = 0.0
    num_eval_batches = 0
    dev_batch = get_batch(eval_features, hp.batch_size, shuffle=False)
    for features in dev_batch:
        num_eval_batches += 1
        feed_dict = m.create_feed_dict(features, False)
        _dev_loss, _dev_gs = sess.run([m.loss_ssl, m.global_ssl_step], feed_dict=feed_dict)
        total_loss += _dev_loss
    dev_loss = total_loss / num_eval_batches
    return dev_loss

print("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
rng = random.Random(hp.rand_seed)
tf.nn.nce_loss
if not os.path.exists(hp.train_prepro):
    print(" Prepare train file")
    train_features = process_file_snli(hp.train, hp.vocab, hp.maxlen, rng)
    print(" Prepare dev file")
    eval_features = process_file_snli(hp.eval, hp.vocab, hp.maxlen, rng)

    print("save training data~~~~")
    pickle.dump(train_features,open(hp.train_prepro, 'wb'))
    pickle.dump(eval_features,open(hp.dev_prepro, 'wb'))

else:
    print("extract training data~~~~")
    train_features = pickle.load(open(hp.train_prepro, 'rb'))
    eval_features = pickle.load(open(hp.dev_prepro, 'rb'))

print("# Load model")
m = SSLNLI(hp)

print("# Session")
saver = tf.train.Saver(max_to_keep=hp.num_epochs)
with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(hp.modeldir)
    if ckpt is None:
        print("Initializing from scratch")
        sess.run(tf.global_variables_initializer())
        save_variable_specs(os.path.join(hp.modeldir, "specs"))
    else:
        saver.restore(sess, ckpt)


    _gs = sess.run(m.global_ssl_step)

    tolerant = 0
    for epoch in range(hp.num_epochs):

        total_loss = 0.0
        print("<<<<<<<<<<<<<<<< epoch {} >>>>>>>>>>>>>>>>".format(epoch))

        batch_train = get_batch(train_features, hp.batch_size, shuffle=True)
        batch_count = 0
        for features in tqdm(batch_train):
            batch_count += 1
            feed_dict = m.create_feed_dict(features, True)
            _, _loss, _gs= sess.run([m.train, m.loss_ssl,m.global_ssl_step], feed_dict=feed_dict)

            total_loss += _loss

            if batch_count and batch_count % 500 == 0:
                print("batch {:d}: total_loss {:.4f} \n".format(
                    batch_count, _loss))


        print("\n")
        print("<<<<<<<<<< epoch {} is done >>>>>>>>>>".format(epoch))
        print("# train results")
        train_loss = total_loss/batch_count

        print("训练集: total_loss {:.4f}\n".format(train_loss))
        #验证集
        dev_loss = evaluate(sess, eval_features)
        print("\n")
        print("# evaluation results")
        print("验证集: total_loss {:.4f} \n".format(dev_loss))


        # save model each epoch
        print("#########SAVE MODEL###########")
        model_output = hp.model_path % (epoch, dev_loss)
        ckpt_name = os.path.join(hp.modeldir, model_output)
        saver.save(sess, ckpt_name, global_step=_gs)
        print("training of {} epochs, {} has been saved.".format(epoch, ckpt_name))


print("Done")
