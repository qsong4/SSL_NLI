# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = -1

import tensorflow as tf
from SSL_pre.model import SSLNLI
from SSL_pre.hparams import Hparams
from tqdm import tqdm
import numpy as np

class Inference(object):
    def __init__(self, is_test=False):
        '''
        Here only load params from ckpt, and change read input method from dataset without placehold
        to dataset with placeholder. Because withuot placeholder you cannt init model when class build which
        means you spend more time on inference stage.
        '''
        hparams = Hparams()
        parser = hparams.parser

        self.hp = parser.parse_args()
        self.m = SSLNLI(self.hp)
        self.global_x = self.m.get_globalx()
        self.global_y = self.m.get_globaly()

        self.sess = tf.Session()
        print(self.hp.modeldir)
        ckpt = tf.train.latest_checkpoint(self.hp.modeldir)
        saver = tf.train.Saver()
        saver.restore(self.sess, ckpt)

    def infer(self, features):
        feed_dict = self.m.create_feed_dict(features, False)
        encx, ency = self.sess.run([self.global_x, self.global_y], feed_dict=feed_dict)
        return encx, ency

    def test(self):
        sent_len = 5
        maxlen = 50
        inputs_a = np.zeros((sent_len, maxlen))
        inputs_b = np.zeros((sent_len, maxlen))
        a_lens = np.zeros(sent_len)
        b_lens = np.zeros(sent_len)
        related_labels = np.zeros((sent_len, 3))

        return inputs_a,inputs_b,a_lens,b_lens,related_labels

if __name__ == '__main__':
    inf = Inference()

    features = inf.test()
    a, b = inf.infer(features)
    print(a)


