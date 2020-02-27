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

def evaluate(sess, eval_features):

    total_t1loss = 0.0
    total_t2loss = 0.0
    total_acc = 0.0
    total_loss = 0.0
    num_eval_batches = 0
    dev_batch = get_batch(eval_features, hp.batch_size, shuffle=False)
    for features in dev_batch:
        num_eval_batches += 1
        feed_dict = m.create_feed_dict(features, False)
        _t1loss, _t2loss, _loss, _t2acc, _gs = sess.run([m.loss_task1, m.loss_task2, m.loss_task_all,
                                                            m.acc, m.global_step], feed_dict=feed_dict)
        total_t1loss += _t1loss
        total_t2loss += _t2loss
        total_loss += _loss
        total_acc += _t2acc

    dev_loss = total_loss / batch_count
    dev_task2_acc = total_acc / batch_count
    dev_task1_loss = total_t1loss / batch_count
    dev_task2_loss = total_t2loss / batch_count


    return dev_loss, dev_task1_loss, dev_task2_loss, dev_task2_acc

print("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
rng = random.Random(hp.rand_seed)

print("# Prepare train file")
train_features = process_file(hp.train, hp.vocab, hp.maxlen, hp.masked_lm_prob, hp.max_predictions_per_seq, rng)
print("# Prepare dev file")
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

    tolerant = 0
    for epoch in range(hp.num_epochs):
        task1_loss = 0.0
        task2_loss = 0.0
        total_loss = 0.0
        task2_acc = 0.0
        print("<<<<<<<<<<<<<<<< epoch {} >>>>>>>>>>>>>>>>".format(epoch))

        batch_train = get_batch(train_features, hp.batch_size, shuffle=True)
        batch_count = 0
        for features in batch_train:
            _, _, _, _,_, _, _,_, _, _,related_labels = features
            batch_count += 1
            feed_dict = m.create_feed_dict(features, True)
            _, _t1loss, _t2loss, _loss, _logit, _t2acc, _gs= sess.run([m.train, m.loss_task1, m.loss_task2, m.loss_task_all,
                                                              m.sentence_logits, m.acc, m.global_step], feed_dict=feed_dict)


            label_pred = tf.argmax(_logit, 1, name='label_pred')
            print(label_pred)
            print(_logit)
            print(related_labels)
            print(_t2acc)
            task1_loss += _t1loss
            task2_loss += _t2loss
            total_loss += _loss
            task2_acc += _t2acc

            if batch_count and batch_count % 500 == 0:
                print("batch {:d}: task1_loss {:.4f}, task2_loss {:.4f}, total_loss {:.4f}, acc {:.3f} \n".format(
                    batch_count, _t1loss, _t2loss, _loss, _t2acc))


        print("\n")
        print("<<<<<<<<<< epoch {} is done >>>>>>>>>>".format(epoch))
        print("# train results")
        train_loss = total_loss/batch_count
        task2_acc = task2_acc/batch_count
        task1_loss = task1_loss/batch_count
        task2_loss = task2_loss/batch_count

        print("训练集: task1_loss {:.4f}, task2_loss {:.4f}, total_loss {:.4f}, acc {:.3f} \n".format(
            task1_loss, task2_loss, train_loss, task2_acc))
        #验证集
        dev_loss, dev_task1_loss, dev_task2_loss, dev_task2_acc = evaluate(sess, eval_features)
        print("\n")
        print("# evaluation results")
        print("验证集: total_loss {:.4f}, task1_loss {:.4f}, task2_loss {:.4f}, acc {:.3f} \n".format(dev_loss, dev_task1_loss, dev_task2_loss, dev_task2_acc))


        # save model each epoch
        print("#########SAVE MODEL###########")
        model_output = hp.model_path % (epoch, dev_loss, dev_task2_acc)
        ckpt_name = os.path.join(hp.modeldir, model_output)
        saver.save(sess, ckpt_name, global_step=_gs)
        print("training of {} epochs, {} has been saved.".format(epoch, ckpt_name))


print("Done")
