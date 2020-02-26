import tensorflow as tf

from model import SSLNLI
from tqdm import tqdm
from data_load import get_batch, process_file
from utils import save_variable_specs
import os
from hparams import Hparams
import math
import random
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def evaluate(sess, eval_init_op, num_eval_batches):

    sess.run(eval_init_op)
    total_steps = 1 * num_eval_batches
    total_acc = 0.0
    total_loss = 0.0
    for i in range(total_steps + 1):
        x, y, x_len, y_len, char_x, char_y, char_x_len, char_y_len, labels = sess.run(data_element)
        feed_dict = m.create_feed_dict(x, y, x_len, y_len, labels, False)
        if hp.char_embedding:
            feed_dict = m.create_char_feed_dict(feed_dict, char_x, char_x_len, char_y, char_y_len)

        #dev_acc, dev_loss = sess.run([dev_accuracy_op, dev_loss_op])
        dev_acc, dev_loss = sess.run([m.acc, m.loss], feed_dict=feed_dict)
        #print("xxx", dev_loss)
        total_acc += dev_acc
        total_loss += dev_loss
    return total_loss/num_eval_batches, total_acc/num_eval_batches

print("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
rng = random.Random(hp.rand_seed)

print("# Prepare train/eval batches")
train_features = process_file(hp.train, hp.vocab, hp.maxlen, hp.masked_lm_prob, hp.max_predictions_per_seq, rng)
eval_features = process_file(hp.eval, hp.vocab, hp.maxlen, hp.masked_lm_prob, hp.max_predictions_per_seq, rng)


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


    _gs = sess.run(m.global_step)
    best_acc = 0.0
    total_loss = 0.0
    total_acc = 0.0
    total_batch = 0
    tolerant = 0
    for epoch in range(hp.num_epochs):
        task1_loss = 0.0
        total_loss = 0.0
        task2_acc = 0.0
        print("<<<<<<<<<<<<<<<< epoch {} >>>>>>>>>>>>>>>>".format(epoch))

        #TODO:给get_batch加上shuffle，因为每个epoch要确保数据顺序不一样。
        batch_train = get_batch(train_features, hp.batch_size)

        for * in batch_train
        if _gs and _gs % 500 == 0:
            print("batch {:d}: loss {:.4f}, acc {:.3f} \n".format(_gs, _loss, _accuracy))

        if _gs and _gs % num_train_batches == 0:

            print("\n")
            print("<<<<<<<<<< epoch {} is done >>>>>>>>>>".format(epoch))
            print("# train results")
            train_loss = total_loss/total_batch
            train_acc = total_acc/total_batch
            print("训练集: loss {:.4f}, acc {:.3f} \n".format(train_loss, train_acc))
            dev_loss, dev_acc = evaluate(sess, eval_init_op, num_eval_batches)
            print("\n")
            print("# evaluation results")
            print("验证集: loss {:.4f}, acc {:.3f} \n".format(dev_loss, dev_acc))
            if dev_acc > best_acc:
                best_acc = dev_acc
                # save model each epoch
                print("#########New Best Result###########")
                model_output = hp.model_path % (epoch, dev_loss, dev_acc)
                ckpt_name = os.path.join(hp.modeldir, model_output)
                saver.save(sess, ckpt_name, global_step=_gs)
                print("training of {} epochs, {} has been saved.".format(epoch, ckpt_name))
            else:
                tolerant += 1

            if tolerant == hp.early_stop:
                print("early stop at {} epochs, acc three epochs has not been improved.".format(epoch))
                break

            """
            #save model when get best acc at dev set
            if dev_acc > best_acc:
                best_acc = dev_acc
                print("# save models")
                model_output = hp.model_path % (epoch, dev_loss, dev_acc)
                ckpt_name = os.path.join(hp.modeldir, model_output)
                saver.save(sess, ckpt_name, global_step=_gs)
                print("training of {} epochs, {} has been saved.".format(epoch, ckpt_name))
            """
            print("**************************************")
            total_loss = 0.0
            total_acc = 0.0
            total_batch = 0

            print("# fall back to train mode")
            sess.run(train_init_op)


print("Done")
