import tensorflow as tf

from data_load import loadGloVe, load_vocab
from modules import get_token_embeddings, ff, positional_encoding, multihead_attention, ln, gather_indexes \
    ,gelu, layer_norm
import math

class SSLNLI:
    def __init__(self, hp):
        self.hp = hp
        _, _, _vocab = load_vocab(hp.vocab)
        self.hp.vocab_size = len(_vocab)
        self.x = tf.placeholder(tf.int32, [None, self.hp.maxlen], name="text_x")
        self.y = tf.placeholder(tf.int32, [None, self.hp.maxlen], name="text_y")
        self.x_len = tf.placeholder(tf.int32, [None])#句子长度边界，用在attention score计算
        self.y_len = tf.placeholder(tf.int32, [None])
        self.labels = tf.placeholder(tf.int32, [None, self.hp.num_class], name="relations")#判断两个句子是否有关系
        self.is_training = tf.placeholder(tf.bool, shape=None, name="is_training")

        self.embedding_table = None
        if self.hp.preembedding:
            self.embedding_table = loadGloVe(self.hp.vec_path)
        self.embeddings = get_token_embeddings(self.embedding_table, self.hp.vocab_size, self.hp.d_model, zero_pad=True)

        self.represation()
        self.loss_ssl = self._loss_ssl_op()
        self.global_ssl_step = self._globalStep_ssl_op()
        self.train = self._training_ssl_op()

    def get_global_feature(self):
        return self.x_global, self.y_global

    def get_local_feature(self):
        return self.x_local, self.x_local

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

    def pre_encoder(self, x):
        with tf.variable_scope("pre_encoder", reuse=tf.AUTO_REUSE):
            #x, seqlens, sents1 = xs

            # src_masks
            src_masks = tf.math.equal(x, 0) # (N, T1)

            # embedding
            enc = tf.nn.embedding_lookup(self.embeddings, x) # (N, T1, d_model)
            enc *= self.hp.d_model**0.5 # scale

            enc += positional_encoding(enc, self.hp.maxlen)
            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=self.is_training)

            return enc, src_masks

    def inter_encode(self, encx, ency, x_masks, y_masks):

        all_layer_x = []
        all_layer_y = []
        with tf.variable_scope("inter_encoder", reuse=tf.AUTO_REUSE):
            ## self Blocks
            for i in range(self.hp.num_blocks_inter):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    _encx = multihead_attention(queries=encx,
                                              keys=encx,
                                              values=encx,
                                              key_masks=x_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=self.is_training,
                                              causality=False)


                    _ency = multihead_attention(queries=ency,
                                               keys=ency,
                                               values=ency,
                                               key_masks=y_masks,
                                               num_heads=self.hp.num_heads,
                                               dropout_rate=self.hp.dropout_rate,
                                               training=self.is_training,
                                               causality=False)


                    encx = multihead_attention(queries=encx,
                                               keys=_ency,
                                               values=encx,
                                               key_masks=x_masks,
                                               num_heads=self.hp.num_heads,
                                               dropout_rate=self.hp.dropout_rate,
                                               training=self.is_training,
                                               causality=False)

                    ency = multihead_attention(queries=ency,
                                               keys=_encx,
                                               values=ency,
                                               key_masks=y_masks,
                                               num_heads=self.hp.num_heads,
                                               dropout_rate=self.hp.dropout_rate,
                                               training=self.is_training,
                                               causality=False)
                    # feed forward

                    encx = ff(encx, num_units=[self.hp.d_ff, self.hp.d_model])
                    ency = ff(ency, num_units=[self.hp.d_ff, self.hp.d_model])

                    all_layer_x.append(encx)
                    all_layer_y.append(ency)

        return all_layer_x, all_layer_y

    def encode(self, encx, src_masks):

        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            all_layer = []
            ## Blocks
            for i in range(self.hp.num_blocks_encoder):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    encx = multihead_attention(queries=encx,
                                              keys=encx,
                                              values=encx,
                                              key_masks=src_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=self.is_training,
                                              causality=False)
                    # feed forward
                    encx = ff(encx, num_units=[self.hp.d_ff, self.hp.d_model])

                    all_layer.append(encx)

        return all_layer

    def represation(self):
        pre_x, x_mask = self.pre_encoder(self.x)
        pre_y, y_mask = self.pre_encoder(self.y)

        all_layers_x = self.encode(pre_x, x_mask)
        all_layers_y = self.encode(pre_y, y_mask)

        if self.hp.inter_attention:
            all_layers_x, all_layers_y = self.inter_encode(all_layers_x[-1], all_layers_y[-1], x_mask, y_mask)

        self.x_global = all_layers_x[-1]
        self.y_global = all_layers_y[-1]

        self.x_local = all_layers_x[self.hp.local_layer]
        self.y_local = all_layers_y[self.hp.local_layer]

        x_cls_in1 = tf.reduce_max(tf.concat([self.x_global, self.x_local], axis=-1), axis=1)
        x_cls_in2 = tf.reduce_max(tf.concat([self.x_global, self.y_local], axis=-1), axis=1)

        y_cls_in1 = tf.reduce_max(tf.concat([self.y_global, self.y_local], axis=-1), axis=1)
        y_cls_in2 = tf.reduce_max(tf.concat([self.y_global, self.x_local], axis=-1), axis=1)

        x_cls1_scores = self.cls(x_cls_in1, [self.hp.d_model, 1], scope="x_cls")
        x_cls2_scores = self.cls(x_cls_in2, [self.hp.d_model, 1], scope="x_cls")

        y_cls1_scores = self.cls(y_cls_in1, [self.hp.d_model, 1], scope="y_cls")
        y_cls2_scores = self.cls(y_cls_in2, [self.hp.d_model, 1], scope="y_cls")

        return x_cls1_scores, x_cls2_scores, y_cls1_scores, y_cls2_scores

    def fc(self, inpt, match_dim, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("fc", reuse=reuse):
            w = tf.get_variable("w", [match_dim, self.hp.num_class], dtype=tf.float32)
            b = tf.get_variable("b", [self.hp.num_class], dtype=tf.float32)
            logits = tf.matmul(inpt, w) + b

        return logits

    def _loss_ssl_op(self):
        x_cls1_scores, x_cls2_scores, y_cls1_scores, y_cls2_scores = self.represation()

        x_info_loss = -tf.reduce_mean(tf.log(x_cls1_scores + 1e-6) + tf.log(1 - x_cls2_scores + 1e-6))
        y_info_loss = -tf.reduce_mean(tf.log(y_cls1_scores + 1e-6) + tf.log(1 - y_cls2_scores + 1e-6))
        loss = x_info_loss + y_info_loss

        return loss

    def cls(self, inputs, num_units, scope="cls_base"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)
            outputs = tf.layers.dense(outputs, num_units[0], activation=tf.nn.relu)
            outputs = tf.layers.dense(outputs, num_units[1], activation=tf.nn.sigmoid)
        return outputs

    def _globalStep_ssl_op(self):
        global_step = tf.train.get_or_create_global_step()
        return global_step

    def _training_ssl_op(self):
        optimizer = tf.train.AdadeltaOptimizer(self.hp.lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(self.loss_ssl, global_step=self.global_ssl_step)

        return train_op