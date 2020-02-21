import tensorflow as tf

from modules import get_token_embeddings, ff, positional_encoding, multihead_attention, ln

class SSLNLI:
    def __init__(self, hp):
        self.hp = hp
        self.x = tf.placeholder(tf.int32, [None, self.hp.maxlen], name="text_x")
        self.y = tf.placeholder(tf.int32, [None, self.hp.maxlen], name="text_y")
        self.x_len = tf.placeholder(tf.int32, [None])#句子长度边界，用在attention score计算
        self.y_len = tf.placeholder(tf.int32, [None])
        self.x_mask_position = tf.placeholder(tf.int32, [None])#句子mask单词的位置
        self.y_mask_position = tf.placeholder(tf.int32, [None])
        self.x_mask_ids = tf.placeholder(tf.int32, [None])#句子mask位置的正确单词
        self.y_mask_ids = tf.placeholder(tf.int32, [None])
        self.is_related = tf.placeholder(tf.int32, [None, self.hp.num_relats], name="relations")#判断两个句子是否有关系
        self.is_training = tf.placeholder(tf.bool, shape=None, name="is_training")

        # self.logits = self._logits_op()
        # self.loss = self._loss_op()
        # self.acc = self._acc_op()
        # self.global_step = self._globalStep_op()
        # self.train = self._training_op()

    def create_feed_dict(self):
        pass

    def encode(self, xs, training=True):

        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            x, seqlens, sents1 = xs

            # src_masks
            src_masks = tf.math.equal(x, 0) # (N, T1)

            # embedding
            enc = tf.nn.embedding_lookup(self.embeddings, x) # (N, T1, d_model)
            enc *= self.hp.d_model**0.5 # scale

            enc += positional_encoding(enc, self.hp.maxlen1)
            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)

            ## Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              key_masks=src_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model])
        memory = enc
        return memory, sents1, src_masks