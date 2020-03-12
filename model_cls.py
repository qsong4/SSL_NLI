import tensorflow as tf

from data_load import loadGloVe, load_vocab
from modules import get_token_embeddings, ff, positional_encoding, multihead_attention, ln, gather_indexes \
    ,gelu, layer_norm
import math

class SSLNLI_cls:
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
        self.cls_logits = self._cls_logits()
        self.loss_ssl = self._loss_ssl_op()
        self.loss_cls = self._loss_cls_op()
        self.acc = self._acc_op()
        self.global_step = self._globalStep_ssl_op()
        # self.global_cls_step = self._globalStep_cls_op()
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

    def _cls_repres(self):

        #self.represation()

        x_masks = tf.math.equal(self.x, 0)  # (N, T1)
        y_masks = tf.math.equal(self.y, 0)  # (N, T1)

        encx, ency = self.get_global_feature()
        print("encx shape: ", encx.shape)
        print("ency shape: ", ency.shape)

        x_layer = []
        y_layer = []

        x_layer.append(encx)
        y_layer.append(ency)

        for i in range(self.hp.num_dense_blocks):
            encx, ency = self.dense_blocks(encx, ency, x_layer, y_layer, x_masks, y_masks, scope="num_dense_blocks_{}".format(i))
            x_layer.append(encx)
            y_layer.append(ency)

        return encx, ency

    def dense_blocks(self, a_repre, b_repre, x_layer, y_layer, x_masks, y_masks, scope, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse):

            _encx = multihead_attention(queries=a_repre,
                                       keys=a_repre,
                                       values=a_repre,
                                       key_masks=x_masks,
                                       num_heads=self.hp.num_heads,
                                       dropout_rate=self.hp.dropout_rate,
                                       training=self.is_training,
                                       causality=False)
            # self-attention
            _ency = multihead_attention(queries=b_repre,
                                       keys=b_repre,
                                       values=b_repre,
                                       key_masks=y_masks,
                                       num_heads=self.hp.num_heads,
                                       dropout_rate=self.hp.dropout_rate,
                                       training=self.is_training,
                                       causality=False)

            # self-attention

            encx = multihead_attention(queries=_encx,
                                       keys=_ency,
                                       values=_encx,
                                       key_masks=x_masks,
                                       num_heads=self.hp.num_heads,
                                       dropout_rate=self.hp.dropout_rate,
                                       training=self.is_training,
                                       causality=False)

            # self-attention
            ency = multihead_attention(queries=b_repre,
                                       keys=b_repre,
                                       values=b_repre,
                                       key_masks=y_masks,
                                       num_heads=self.hp.num_heads,
                                       dropout_rate=self.hp.dropout_rate,
                                       training=self.is_training,
                                       causality=False)

            encx, ency = self._infer(encx, ency)

            encx, ency = self._dense_infer(encx, ency, x_layer, y_layer)
            dim = encx.shape.as_list()[-1]
            # feed forward
            encx = ff(encx, num_units=[self.hp.d_ff, dim])
            ency = ff(ency, num_units=[self.hp.d_ff, dim])


            return encx, ency

    def _dense_infer(self, encx, ency, x_layer, y_layer, scope="dese_local_inference"):
        with tf.variable_scope(scope):

            #可以有两种方式
            #1. concat前面所有层的信息
            #2. 只concat前面一层的信息
            a_res = tf.concat([x_layer[-1]] + [encx], axis=2)
            b_res = tf.concat([y_layer[-1]] + [ency], axis=2)
            a_res = tf.layers.dropout(a_res, self.hp.dropout_rate, training=self.is_training)
            b_res = tf.layers.dropout(b_res, self.hp.dropout_rate, training=self.is_training)
            #if layer_num in self.AE_layer:
            a_res = self._project_op(a_res)  # (?,?,d_model)
            b_res = self._project_op(b_res)  # (?,?,d_model)
            # if layer_num in self.AE_layer:
            #     a_res, ae_loss_a = self._AutoEncoder(a_res)
            #     b_res, ae_loss_b = self._AutoEncoder(b_res)

        return a_res, b_res

    def _infer(self, encx, ency, scope="local_inference"):
        with tf.variable_scope(scope):

            attentionWeights = tf.matmul(encx, tf.transpose(ency, [0, 2, 1]))
            attentionSoft_a = tf.nn.softmax(attentionWeights)
            attentionSoft_b = tf.nn.softmax(tf.transpose(attentionWeights))
            attentionSoft_b = tf.transpose(attentionSoft_b)

            a_hat = tf.matmul(attentionSoft_a, ency)
            b_hat = tf.matmul(attentionSoft_b, encx)
            a_diff = tf.subtract(encx, a_hat)
            a_mul = tf.multiply(encx, a_hat)
            b_diff = tf.subtract(ency, b_hat)
            b_mul = tf.multiply(ency, b_hat)

            a_res = tf.concat([a_hat, a_diff, a_mul], axis=2)
            b_res = tf.concat([b_hat, b_diff, b_mul], axis=2)

            # BN
            # a_res = tf.layers.batch_normalization(a_res, training=self.is_training, name='bn1', reuse=tf.AUTO_REUSE)
            # b_res = tf.layers.batch_normalization(b_res, training=self.is_training, name='bn2', reuse=tf.AUTO_REUSE)
            # project
            a_res = self._project_op(a_res)  # (?,?,d_model)
            b_res = self._project_op(b_res)  # (?,?,d_model)

            # a_res += encx
            # b_res += ency
            a_res = tf.concat([encx, a_res], axis=-1)
            b_res = tf.concat([ency, b_res], axis=-1)

            # a_res = ln(a_res)
            # b_res = ln(b_res)

        return a_res, b_res

    def _cls_logits(self):
        x_repre, y_repre = self._cls_repres()
        print("x_repre", x_repre.shape)

        concat_xy = tf.concat([x_repre, y_repre], axis=-1)
        #agg_repre = self.aggregation(concat_xy, concat_xy)
        print("concat_xy shape: ", concat_xy.shape)
        _,_,agg_repre = self.lstm_layer(concat_xy, self.hp.lstm_dim, scope_name="lstm_agg")
        print("agg_repre shape: ", agg_repre.shape)
        agg_res = tf.reduce_mean(agg_repre, axis=1)
        logits = self.fc(agg_res, match_dim = agg_res.shape.as_list()[-1])

        return logits

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

    def _loss_cls_op(self, l2_lambda=0.0001):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.cls_logits, labels=self.labels))
        weights = [v for v in tf.trainable_variables() if ('w' in v.name) or ('kernel') in v.name]
        l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * l2_lambda
        loss += l2_loss

        return loss

    def fusion(self, x, align):
        with tf.variable_scope('align', reuse=tf.AUTO_REUSE):
            x = tf.concat([
                tf.layers.dense(tf.concat([x, align], axis=-1), self.hp.d_model, activation=tf.nn.relu, name='orig'),
                tf.layers.dense(tf.concat([x, x - align], axis=-1), self.hp.d_model, activation=tf.nn.relu, name='sub'),
                tf.layers.dense(tf.concat([x, x * align], axis=-1), self.hp.d_model, activation=tf.nn.relu, name='mul'),
            ], axis=-1)
            x = tf.layers.dropout(x, self.hp.dropout_rate, training=self.is_training)
            x = tf.layers.dense(x, self.hp.d_model, activation=tf.nn.relu, name="proj")
            return x

    def cls(self, inputs, num_units, scope="cls_base"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)
            outputs = tf.layers.dense(outputs, num_units[0], activation=tf.nn.relu)
            outputs = tf.layers.dense(outputs, num_units[1], activation=tf.nn.sigmoid)
        return outputs

    def _project_op(self, inputx):
        with tf.variable_scope("projection", reuse=tf.AUTO_REUSE):
            inputx = tf.layers.dense(inputx, self.hp.d_model,
                                     activation=tf.nn.relu,
                                     name='fnn',
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

            return inputx

    def _globalStep_ssl_op(self):
        global_step = tf.train.get_or_create_global_step()
        return global_step

    def _globalStep_cls_op(self):
        global_step = tf.train.get_or_create_global_step()
        return global_step

    def _training_ssl_op(self):
        optimizer = tf.train.AdadeltaOptimizer(self.hp.lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(self.loss_ssl+self.loss_cls, global_step=self.global_step)

        return train_op

    def _training_cls_op(self):
        optimizer = tf.train.AdadeltaOptimizer(self.hp.lr)
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

    def calculate_att(self, a, b, scope='alignnment'):
        with tf.variable_scope(scope):
            mask_a = tf.sequence_mask(self.x_len, self.hp.maxlen, dtype=tf.float32)
            mask_b = tf.sequence_mask(self.y_len, self.hp.maxlen, dtype=tf.float32)
            mask_a = tf.expand_dims(mask_a, axis=-1)
            mask_b = tf.expand_dims(mask_b, axis=-1)
            temperature = tf.get_variable('temperature', shape=(), dtype=tf.float32, trainable=True,
                                          initializer=tf.constant_initializer(math.sqrt(1 / self.hp.hidden_size)))

            attention = self._attention(a, b, temperature, self.hp.dropout_rate)
            attention_mask = tf.matmul(mask_a, mask_b, transpose_b=True)
            attention = attention_mask * attention + (1 - attention_mask) * tf.float32.min
            attention_a = tf.nn.softmax(attention, dim=1)
            attention_b = tf.nn.softmax(attention, dim=2)
            attention_a = tf.identity(attention_a, name='attention_a')
            attention_b = tf.identity(attention_b, name='attention_b')

            feature_b = tf.matmul(attention_a, a, transpose_a=True)
            feature_a = tf.matmul(attention_b, b)

            return feature_a, feature_b

    def _attention(self, a, b, t, dropout_keep_prob):
        with tf.variable_scope('proj'):
            a = tf.layers.dense(tf.layers.dropout(a, dropout_keep_prob, training=self.is_training),
                      self.hp.d_model, activation=tf.nn.relu)
            b = tf.layers.dense(tf.layers.dropout(b, dropout_keep_prob, training=self.is_training),
                                self.hp.d_model, activation=tf.nn.relu)
            return tf.matmul(a, b, transpose_b=True) * t


    def pooler(self, sequence_output):
        with tf.variable_scope("pooler", reuse=tf.AUTO_REUSE):
            # We "pool" the model by simply taking the hidden state corresponding
            # to the first token. We assume that this has been pre-trained
            first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
            pooled_output = tf.layers.dense(
                first_token_tensor,
                self.hp.hidden_size,
                activation=tf.tanh,
                name='pooler_dense',
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

            return pooled_output

    def lstm_layer(self, input_reps, lstm_dim, input_lengths=None, scope_name=None, reuse=False):
        with tf.variable_scope(scope_name, reuse=reuse):
            context_lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(lstm_dim)
            context_lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(lstm_dim)

            (f_rep, b_rep), _ = tf.nn.bidirectional_dynamic_rnn(
                context_lstm_cell_fw, context_lstm_cell_bw, input_reps, dtype=tf.float32,
                sequence_length=input_lengths)  # [batch_size, question_len, context_lstm_dim]
            outputs = tf.concat(axis=2, values=[f_rep, b_rep])
        return (f_rep, b_rep, outputs)
