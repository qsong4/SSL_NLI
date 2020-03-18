import tensorflow as tf
import os,sys
from ORI_LSTM.model_lstm import SSLNLI_lstm
from tqdm import tqdm
from ORI_LSTM.data_load import get_batch, process_file_snli
import os
from ORI_LSTM.hparams import Hparams
import random
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import pickle
print(sys.path)

def evaluate(sess, eval_features):

    total_dev_acc = 0.0
    total_dev_loss_cls = 0.0
    num_eval_batches = 0
    dev_batch = get_batch(eval_features, hp.batch_size, shuffle=False)
    for features in dev_batch:
        num_eval_batches += 1

        feed_dict = m.create_feed_dict(features, False)
        _devloss_cls, _devacc, _devgs = sess.run([m.loss_cls, m.acc, m.global_step], feed_dict=feed_dict)

        total_dev_loss_cls += _devloss_cls
        total_dev_acc += _devacc

    dev_loss_cls = total_dev_loss_cls / num_eval_batches
    dev_acc = total_dev_acc / num_eval_batches

    return dev_loss_cls, dev_acc

print("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
rng = random.Random(hp.rand_seed)

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
m = SSLNLI_lstm(hp)

print("# Session")
saver = tf.train.Saver(max_to_keep=hp.num_epochs)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    tolerant = 0
    best_acc = 0.0
    for epoch in range(hp.num_epochs):

        loss_ssl = 0.0
        loss_cls = 0.0
        acc = 0.0
        print("<<<<<<<<<<<<<<<< epoch {} >>>>>>>>>>>>>>>>".format(epoch))

        batch_train = get_batch(train_features, hp.batch_size, shuffle=True)
        batch_count = 0
        for features in tqdm(batch_train):
            batch_count += 1
            feed_dict = m.create_feed_dict(features, True)
            _, _loss_cls, _acc, _gs = sess.run([m.train, m.loss_cls, m.acc, m.global_step], feed_dict=feed_dict)

            loss_cls += _loss_cls
            acc += _acc

            if batch_count and batch_count % 500 == 0:
                print("batch {:d}: loss_ssl {:.4f}, loss_cls {:.4f}, acc {:.3f} \n".format(
                    batch_count, 0, _loss_cls, _acc))


        print("\n")
        print("<<<<<<<<<< epoch {} is done >>>>>>>>>>".format(epoch))
        print("# train results")
        train_loss_cls = loss_cls / batch_count
        task2_acc = acc/batch_count

        print("训练集: ssl_loss {:.4f}, cls_loss {:.4f}, acc {:.3f} \n".format(0, train_loss_cls ,task2_acc))
        #验证集
        dev_loss2, dev_task2_acc = evaluate(sess, eval_features)
        print("\n")
        print("# evaluation results")
        print("验证集: ssl_loss {:.4f}, cls_loss {:.4f},acc {:.3f} \n".format(0, dev_loss2,dev_task2_acc))

        if dev_task2_acc > best_acc:
            best_acc = dev_task2_acc
            # save model each epoch
            print("#########SAVE MODEL###########")
            model_output = hp.model_path % (epoch, dev_task2_acc)
            ckpt_name = os.path.join(hp.modeldir, model_output)
            saver.save(sess, ckpt_name, global_step=_gs)
            print("training of {} epochs, {} has been saved.".format(epoch, ckpt_name))


print("Done")
