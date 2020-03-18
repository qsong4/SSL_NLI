import tensorflow as tf
from ORI_LSTM.data_load import loadGloVe, load_vocab
from ORI_LSTM.modules import get_token_embeddings
import math


class SSLNLI_lstm:
    def __init__(self, hp):
        self.hp = hp
        _, _, _vocab = load_vocab(hp.vocab)
        self.hp.vocab_size = len(_vocab)
        self.x = tf.placeholder(tf.float32, [None, self.hp.maxlen], name="text_x")
        self.y = tf.placeholder(tf.float32, [None, self.hp.maxlen], name="text_y")
        self.x_len = tf.placeholder(tf.int32, [None])  # 句子长度边界，用在attention score计算
        self.y_len = tf.placeholder(tf.int32, [None])
        self.labels = tf.placeholder(tf.int32, [None, self.hp.num_class], name="relations")  # 判断两个句子是否有关系
        self.is_training = tf.placeholder(tf.bool, shape=None, name="is_training")

        self.embedding_table = None
        if self.hp.preembedding:
            self.embedding_table = loadGloVe(self.hp.vec_path)
        with tf.variable_scope("embd"):
            self.embeddings = get_token_embeddings(self.embedding_table, self.hp.vocab_size, self.hp.d_model, zero_pad=True)

        self.cls_logits = self._cls_logits()
        self.loss_cls = self._loss_cls_op()
        self.acc = self._acc_op()
        self.global_step = self._globalStep_cls_op()
        self.train = self._training_cls_op()

    def create_feed_dict(self, features, is_training):

        inputs_a, inputs_b, a_lens, b_lens, labels = features
        feed_dict = {
            self.x: inputs_a,
            self.y: inputs_b,
            self.x_len: a_lens,
            self.y_len: b_lens,
            self.is_training: is_training,
            self.labels: labels,
        }
        return feed_dict

    def _cls_logits(self):
        x_repre = tf.nn.embedding_lookup(self.embeddings, self.x)
        y_repre = tf.nn.embedding_lookup(self.embeddings, self.y)

        print("x_repre shape: ", x_repre.shape)
        _, _, agg_repre1 = self.lstm_layer(x_repre, self.hp.lstm_dim,
                                           scope_name="lstm_agg")
        _, _, agg_repre2 = self.lstm_layer(y_repre, self.hp.lstm_dim,
                                           scope_name="lstm_agg")

        print("agg_repre shape: ", agg_repre1.shape)
        max_x = tf.reduce_max(agg_repre1, axis=1)
        avg_x = tf.reduce_mean(agg_repre1, axis=1)
        max_y = tf.reduce_max(agg_repre2, axis=1)
        avg_y = tf.reduce_mean(agg_repre2, axis=1)

        agg_res = tf.concat([max_x, avg_x, max_y, avg_y], axis=-1)
        logits = self.fc(agg_res, match_dim=agg_res.shape.as_list()[-1])

        return logits

    def fc(self, inpt, match_dim, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("fc", reuse=reuse):
            w = tf.get_variable("w", [match_dim, self.hp.num_class], dtype=tf.float32)
            b = tf.get_variable("b", [self.hp.num_class], dtype=tf.float32)
            logits = tf.matmul(inpt, w) + b

        return logits

    def _loss_cls_op(self, l2_lambda=0.0001):
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.cls_logits, labels=self.labels))
        weights = [v for v in tf.trainable_variables() if ('w' in v.name) or ('kernel') in v.name]
        l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * l2_lambda
        loss += l2_loss

        return loss

    def _globalStep_cls_op(self):
        global_step = tf.train.get_or_create_global_step()
        return global_step

    def _training_cls_op(self):
        # optimizer = tf.train.AdadeltaOptimizer(self.hp.lr)
        optimizer = tf.train.AdamOptimizer(self.hp.lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(self.loss_cls, global_step=self.global_step)
        return train_op

    def _acc_op(self):
        with tf.variable_scope('acc', reuse=tf.AUTO_REUSE):
            label_pred = tf.argmax(tf.nn.softmax(self.cls_logits), 1, name='label_pred')
            label_true = tf.argmax(self.labels, 1, name='label_true')
            correct_pred = tf.equal(tf.cast(label_pred, tf.int32), tf.cast(label_true, tf.int32))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='Accracy')
        return accuracy

    def lstm_layer(self, input_reps, lstm_dim, input_lengths=None, scope_name=None, reuse=False):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            context_lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(lstm_dim)
            context_lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(lstm_dim)

            (f_rep, b_rep), _ = tf.nn.bidirectional_dynamic_rnn(
                context_lstm_cell_fw, context_lstm_cell_bw, input_reps, dtype=tf.float32,
                sequence_length=input_lengths)  # [batch_size, question_len, context_lstm_dim]
            outputs = tf.concat(axis=2, values=[f_rep, b_rep])
        return (f_rep, b_rep, outputs)
