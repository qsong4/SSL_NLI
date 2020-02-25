import tensorflow as tf

from data_load import loadGloVe
from modules import get_token_embeddings, ff, positional_encoding, multihead_attention, ln, gather_indexes \
    ,gelu, layer_norm
import math

class SSLNLI:
    def __init__(self, hp):
        self.hp = hp
        self.x = tf.placeholder(tf.int32, [None, self.hp.maxlen], name="text_x")
        self.y = tf.placeholder(tf.int32, [None, self.hp.maxlen], name="text_y")
        self.x_len = tf.placeholder(tf.int32, [None])#句子长度边界，用在attention score计算
        self.y_len = tf.placeholder(tf.int32, [None])
        self.x_mask_position = tf.placeholder(tf.int32, [self.hp.max_predictions_per_seq])#句子mask单词的位置
        self.y_mask_position = tf.placeholder(tf.int32, [self.hp.max_predictions_per_seq])
        self.x_mask_ids = tf.placeholder(tf.int32, [self.hp.max_predictions_per_seq])#句子mask位置的正确单词
        self.y_mask_ids = tf.placeholder(tf.int32, [self.hp.max_predictions_per_seq])
        self.x_mask_weight = tf.placeholder(tf.int32, [self.hp.max_predictions_per_seq])
        self.y_mask_weight = tf.placeholder(tf.int32, [self.hp.max_predictions_per_seq])
        self.is_related = tf.placeholder(tf.int32, [None, self.hp.num_relats], name="relations")#判断两个句子是否有关系
        self.is_training = tf.placeholder(tf.bool, shape=None, name="is_training")

        self.embedding_table = None
        if self.hp.preembeddinng:
            self.embedding_table = loadGloVe(self.hp.vec_path)
        self.embeddings = get_token_embeddings(self.embedding_table, self.hp.vocab_size, self.hp.d_model, zero_pad=False)

        self.represation()
        self.loss_task1, self.loss_task2, self.loss_task_all = self._loss_op()
        self.acc = self._acc_op()
        self.global_step = self._globalStep_op()
        self.train = self._training_op()

    def get_pooled_output(self):
        return self.pooled_output_x, self.pooled_output_y

    def get_sequence_output(self):
        return self.sequence_output_x, self.sequence_output_y

    def create_feed_dict(self):
        pass

    def pre_encoder(self, x):
        with tf.variable_scope("pre_encoder", reuse=tf.AUTO_REUSE):
            #x, seqlens, sents1 = xs

            # src_masks
            src_masks = tf.math.equal(x, 0) # (N, T1)

            # embedding
            enc = tf.nn.embedding_lookup(self.embeddings, x) # (N, T1, d_model)
            enc *= self.hp.d_model**0.5 # scale

            enc += positional_encoding(enc, self.hp.maxlen1)
            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=self.is_training)

            return enc, src_masks


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

        self.all_layers_x = all_layers_x
        self.all_layers_y = all_layers_y

        #目前使用的最后一层，后面有需要可以变成倒数第二层
        self.sequence_output_x = self.all_layers_x[-1] #encoder最后一层的输出(batchsize,seq_len,hidden_size)
        self.sequence_output_y = self.all_layers_y[-1]  # encoder最后一层的输出

        #简单使用第一个标签作为最后输出。
        #后面可以结合常见的attention交互
        self.pooled_output_x = self.pooler(self.sequence_output_x) #得到cls标签的向量(batchsize,hidden_size)
        self.pooled_output_y = self.pooler(self.sequence_output_y)

    def _loss_op(self):
        task2_loss = self.sentence_relate_loss(self.pooled_output_x, self.pooled_output_y)
        task1_loss_x = self.mask_lm_loss(self.sequence_output_x, self.x_mask_position,self.x_mask_ids, self.x_mask_weight)
        task1_loss_y = self.mask_lm_loss(self.sequence_output_x, self.x_mask_position,self.y_mask_ids,self.y_mask_weight)
        task1_loss = task1_loss_x + task1_loss_y
        return task1_loss, task2_loss, task1_loss + task2_loss

    def gather_indexes(self, sequence_tensor, positions):
        """Gathers the vectors at the specific positions over a minibatch."""
        sequence_shape = sequence_tensor.shape.to_list()
        batch_size = sequence_shape[0]
        seq_length = sequence_shape[1]
        width = sequence_shape[2]

        flat_offsets = tf.reshape(
            tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
        flat_positions = tf.reshape(positions + flat_offsets, [-1])
        flat_sequence_tensor = tf.reshape(sequence_tensor,
                                          [batch_size * seq_length, width])
        output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
        return output_tensor

    def mask_lm_loss(self, sequence_output, mask_position, mask_ids, label_weights):
        input_tensor = gather_indexes(sequence_output, mask_position)

        with tf.variable_scope("cls/predictions", reuse=tf.AUTO_REUSE):
            # We apply one more non-linear transformation before the output layer.
            # This matrix is not used after pre-training.
            with tf.variable_scope("transform"):
                input_tensor = tf.layers.dense(
                    input_tensor,
                    units=self.hp.hidden_size,
                    activation=gelu,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
                input_tensor = layer_norm(input_tensor)

            # The output weights are the same as the input embeddings, but there is
            # an output-only bias for each token.
            output_bias = tf.get_variable(
                "output_bias",
                shape=[self.hp.vocab_size],
                initializer=tf.zeros_initializer())
            logits = tf.matmul(input_tensor, self.embeddings, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            label_ids = tf.reshape(mask_ids, [-1])
            label_weights = tf.reshape(label_weights, [-1])

            one_hot_labels = tf.one_hot(
                label_ids, depth=self.hp.vocab_size, dtype=tf.float32)

            # The `positions` tensor might be zero-padded (if the sequence is too
            # short to have the maximum number of predictions). The `label_weights`
            # tensor has a value of 1.0 for every real prediction and 0.0 for the
            # padding predictions.
            per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
            numerator = tf.reduce_sum(label_weights * per_example_loss)
            denominator = tf.reduce_sum(label_weights) + 1e-5
            loss = numerator / denominator

            return loss

    def sentence_relate_loss(self, x, y):

        x = tf.concat([x , y, x-y, x*y], axis=-1)
        self.sentence_logits = self.fc_2l(x, num_units = [self.hp.d_model, self.hp.num_class], scope="fc_2l")
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.sentence_logits, labels=self.is_related))
        return loss

    def fc_2l(self, inputs, num_units, scope="fc_2l"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)
            outputs = tf.layers.dense(outputs, num_units[1])

        return outputs

    def _globalStep_op(self):
        global_step = tf.train.get_or_create_global_step()
        return global_step

    def _training_op(self):
        optimizer = tf.train.AdadeltaOptimizer(self.hp.lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(self.loss_task_all, global_step=self.global_step)

        return train_op

    def _acc_op(self):
        with tf.variable_scope('acc', reuse=tf.AUTO_REUSE):
            label_pred = tf.argmax(self.sentence_logits, name='label_pred')
            label_true = tf.argmax(self.is_related, name='label_true')
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
                                          initializer=tf.constant_initializer(math.sqrt(1 / self.args.hidden_size)))

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

                    enx = ff(enx, num_units=[self.hp.d_ff, self.hp.d_model])
                    eny = ff(eny, num_units=[self.hp.d_ff, self.hp.d_model])

                    all_layer_x.append(encx)
                    all_layer_y.append(ency)

        return all_layer_x, all_layer_y