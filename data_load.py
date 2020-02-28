import tensorflow as tf
from utils import calc_num_batches
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import numpy as np
import re
from keras.preprocessing.sequence import pad_sequences
import tensorflow.keras as kr
from gensim.models import Word2Vec
import random

#rng = random.Random(5)

def loadGloVe(filename):
    embd = np.load(filename)
    return embd

def loadGloVe_2(filename, emb_size):
    mu, sigma = 0, 0.1  # 均值与标准差
    rarray = np.random.normal(mu, sigma, emb_size)
    rarray_cls = np.random.normal(mu, sigma, emb_size)
    embd = {}
    embd['<pad>'] = [0]*emb_size
    #embd['<pad>'] = list(rarray)
    embd['<unk>'] = list(rarray)
    embd['<cls>'] = list(rarray_cls)
    file = open(filename,'r')
    for line in tqdm(file.readlines()):
        row = line.rstrip().split(' ')
        if row[0] in embd.keys():
            continue
        else:
            embd[row[0]] = [float(v) for v in row[1:]]
    file.close()
    return embd

def loadw2v(filename, emb_size):
    mu, sigma = 0, 0.1  # 均值与标准差
    rarray = np.random.normal(mu, sigma, emb_size)
    embd = {}
    embd['<pad>'] = [0]*emb_size
    #embd['<pad>'] = list(rarray)
    embd['<unk>'] = list(rarray)
    w2v = Word2Vec.load(filename)
    for word in w2v.wv.vocab:
        if word in embd.keys():
            continue
        else:
            embd[word] = w2v[word]
    return embd

def preprocessVec(gloveFile, vocab_file, outfile):
    emdb = loadGloVe_2(gloveFile, 300)
    #emdb = loadw2v(gloveFile, 300) #中文w2v用的
    trimmd_embd = []
    with open(vocab_file, 'r') as fr:
        for line in fr:
            word = line.rstrip()
            if word in emdb:
                trimmd_embd.append(emdb[word])
            else:
                trimmd_embd.append(emdb['<unk>'])
    np.save(outfile, trimmd_embd)

def load_vocab(vocab_fpath):
    '''Loads vocabulary file and returns idx<->token maps
    vocab_fpath: string. vocabulary file path.
    Note that these are reserved
    0: <pad>, 1: <unk>, 2: <mask>, 3: <cls>

    Returns
    two dictionaries.
    '''
    with open(vocab_fpath, 'r') as fr:
        vocab = [line.strip() for line in fr]

    # vocab = [line.split()[0] for line in open(vocab_fpath, 'r').read().splitlines()]
    token2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2token = {idx: token for idx, token in enumerate(vocab)}
    return token2idx, idx2token, vocab

def load_data(fpath, maxlen):
    '''Loads source and target data and filters out too lengthy samples.
    fpath1: source file path. string.
    fpath2: target file path. string.
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.

    Returns
    sents1: list of source sents
    sents2: list of target sents
    '''
    sents1, sents2 = [], []
    labels = []
    with open(fpath, 'r') as fr:
        for line in fr:
            content = line.strip().split("\t")
            sent1 = content[0].lower()
            sent2 = content[1].lower()
            #label = int(content[2]) #cn data
            label = content[2] #snli data
            if len(sent1.split()) > maxlen:
                continue
                #sent1 = sent1[len(sent1) - maxlen:]#for cn data
            if len(sent2.split()) > maxlen:
                continue
                #sent2 = sent2[len(sent2) - maxlen:]#for cn data
            sents1.append(sent1)
            sents2.append(sent2)
            labels.append([label])
    return sents1, sents2, labels

def removePunc(inputStr):
    string = re.sub(r"\W+", "", inputStr)
    return string.strip()

def encode(inp, dict, maxlen):
    '''Converts string to number. Used for `generator_fn`.
    inp: 1d byte array.
    type: "x" (source side) or "y" (target side)
    dict: token2idx dictionary

    Returns
    list of numbers
    '''
    #for cn dataset
    #x = [dict.get(t, dict["<unk>"]) for t in inp]
    #for snli dateset
    x = []
    x.append(dict["<cls>"])#第一个是cls标签
    for i in re.split(r"\W+", inp):
        i = i.strip()
        i = removePunc(i)
        i = i.lower()
        if i == "":
            continue
        x.append(dict.get(i, dict["<unk>"]))
    x_len = len(x)
    x = pad_sequences([x], maxlen=maxlen, dtype='int32',padding='post')
    #x = [dict.get(t, dict["<unk>"]) for t in re.split(r"\W+'", inp)]
    return x[0], x_len

def process_file(fpath, vocab_fpath, maxlen, masked_lm_prob, max_predictions_per_seq, rng):

    token2idx, _, vocab = load_vocab(vocab_fpath)
    vocab_len = len(vocab)
    with open(fpath, 'r') as fr:

        sentences = list(fr.readlines())
        rng.shuffle(sentences)
        sent_len = len(sentences)

        inputs_a = np.zeros((sent_len, maxlen))
        inputs_b = np.zeros((sent_len, maxlen))
        a_lens = np.zeros(sent_len)
        b_lens = np.zeros(sent_len)
        masked_positions_a = np.zeros((sent_len, max_predictions_per_seq))
        masked_labels_a = np.zeros((sent_len, max_predictions_per_seq))
        masked_weights_a = np.zeros((sent_len, max_predictions_per_seq))
        masked_positions_b = np.zeros((sent_len, max_predictions_per_seq))
        masked_labels_b = np.zeros((sent_len, max_predictions_per_seq))
        masked_weights_b = np.zeros((sent_len, max_predictions_per_seq))
        related_labels = []


        for index, sent in tqdm(enumerate(sentences)):
            content = sent.strip().split('\t')
            senta = content[0]
            sentb = content[1]
            real_label = content[2]
            label = 'related'

            enc_a, len_a = encode(senta, token2idx, maxlen)
            enc_b, len_b = encode(sentb, token2idx, maxlen)
            #contradiction和neutral的
            if real_label == 'contradiction' or real_label == "neutral":
                label = 'related'

            else:
                label = 'not_related'

            (enc_a, masked_lm_positions_a, masked_lm_labels_a, masked_lm_weights_a) \
                = create_masked_lm(enc_a, len_a, masked_lm_prob, max_predictions_per_seq, vocab_len,
                                   rng)

            (enc_b, masked_lm_positions_b, masked_lm_labels_b, masked_lm_weights_b) \
                = create_masked_lm(enc_b, len_b, masked_lm_prob, max_predictions_per_seq, vocab_len,
                                   rng)

            inputs_a[index, :] = np.array(enc_a)
            inputs_b[index, :] = np.array(enc_b)
            a_lens[index] = len_a
            b_lens[index] = len_b
            masked_positions_a[index, :] = np.array(masked_lm_positions_a)
            masked_positions_b[index, :] = np.array(masked_lm_positions_b)
            masked_labels_a[index, :] = np.array(masked_lm_labels_a)
            masked_labels_b[index, :] = np.array(masked_lm_labels_b)
            masked_weights_a[index, :] = np.array(masked_lm_weights_a)
            masked_weights_b[index, :] = np.array(masked_lm_weights_b)
            related_labels.append([label])

    #这里是二分类任务，所以numcls设置为了2
    label_enc = OneHotEncoder(sparse=False, categories='auto')
    related_labels = label_enc.fit_transform(related_labels)
    #related_labels = kr.utils.to_categorical(related_labels, num_classes = 2)

    print("***********data example***********")
    print("enc_a: ", inputs_a[0,:])
    print("masked_lm_positions_a: ", masked_positions_a[0,:])
    print("masked_lm_labels_a: ", masked_labels_a[0,:])
    print("masked_lm_weights_a: ", masked_weights_a[0,:])
    print("related_labels: ", related_labels[0, :])

    return (inputs_a, inputs_b, a_lens, b_lens,
            masked_positions_a, masked_labels_a, masked_weights_a,
            masked_positions_b, masked_labels_b, masked_weights_b,
            related_labels)

def _process_file(fpath, vocab_fpath, maxlen, masked_lm_prob, max_predictions_per_seq, rng):

    token2idx, _, vocab = load_vocab(vocab_fpath)
    vocab_len = len(vocab)
    with open(fpath, 'r') as fr:

        sentences = list(fr.readlines())
        rng.shuffle(sentences)
        sent_len = len(sentences)

        inputs_a = np.zeros((sent_len, maxlen))
        inputs_b = np.zeros((sent_len, maxlen))
        a_lens = np.zeros(sent_len)
        b_lens = np.zeros(sent_len)
        masked_positions_a = np.zeros((sent_len, max_predictions_per_seq))
        masked_labels_a = np.zeros((sent_len, max_predictions_per_seq))
        masked_weights_a = np.zeros((sent_len, max_predictions_per_seq))
        masked_positions_b = np.zeros((sent_len, max_predictions_per_seq))
        masked_labels_b = np.zeros((sent_len, max_predictions_per_seq))
        masked_weights_b = np.zeros((sent_len, max_predictions_per_seq))
        related_labels = []


        for index, sent in tqdm(enumerate(sentences)):
            content = sent.strip().split('\t')
            senta = content[0]
            sentb = content[1]
            real_label = content[2]
            label = 'related'

            #50%的概率是有关系的
            if rng.random() > 0.5:
                enc_a, len_a = encode(senta, token2idx, maxlen)
                enc_b, len_b = encode(sentb, token2idx, maxlen)

            else:
                label = 'not_related'
                enc_a, len_a = encode(senta, token2idx, maxlen)
                for _ in range(10):
                    sentb_index = rng.randint(0, sent_len - 1)
                    if sentences[sentb_index] != sent:
                        break
                sentb = sentences[sentb_index].strip().split('\t')[1]
                enc_b, len_b = encode(sentb, token2idx, maxlen)
            (enc_a, masked_lm_positions_a, masked_lm_labels_a, masked_lm_weights_a) \
                = create_masked_lm(enc_a, len_a, masked_lm_prob, max_predictions_per_seq, vocab_len,
                                   rng)

            (enc_b, masked_lm_positions_b, masked_lm_labels_b, masked_lm_weights_b) \
                = create_masked_lm(enc_b, len_b, masked_lm_prob, max_predictions_per_seq, vocab_len,
                                   rng)

            inputs_a[index, :] = np.array(enc_a)
            inputs_b[index, :] = np.array(enc_b)
            a_lens[index] = len_a
            b_lens[index] = len_b
            masked_positions_a[index, :] = np.array(masked_lm_positions_a)
            masked_positions_b[index, :] = np.array(masked_lm_positions_b)
            masked_labels_a[index, :] = np.array(masked_lm_labels_a)
            masked_labels_b[index, :] = np.array(masked_lm_labels_b)
            masked_weights_a[index, :] = np.array(masked_lm_weights_a)
            masked_weights_b[index, :] = np.array(masked_lm_weights_b)
            related_labels.append([label])

    #这里是二分类任务，所以numcls设置为了2
    label_enc = OneHotEncoder(sparse=False, categories='auto')
    related_labels = label_enc.fit_transform(related_labels)
    #related_labels = kr.utils.to_categorical(related_labels, num_classes = 2)

    print("***********data example***********")
    print("enc_a: ", inputs_a[0,:])
    print("masked_lm_positions_a: ", masked_positions_a[0,:])
    print("masked_lm_labels_a: ", masked_labels_a[0,:])
    print("masked_lm_weights_a: ", masked_weights_a[0,:])
    print("related_labels: ", related_labels[0, :])

    return (inputs_a, inputs_b, a_lens, b_lens,
            masked_positions_a, masked_labels_a, masked_weights_a,
            masked_positions_b, masked_labels_b, masked_weights_b,
            related_labels)



def create_masked_lm(enc, lenx, masked_lm_prob, max_predictions_per_seq, vocab_len, rng):
    masked_lm_labels = []
    masked_lm_positions = []

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(lenx * masked_lm_prob))))
    tem_indexes = []
    for (i, token) in enumerate(enc):
        if token in [0,1,2,3]:#排除pad,unk,mask,cls
            continue
        tem_indexes.append(i)

    rng.shuffle(tem_indexes)
    cand_indexes = tem_indexes[:num_to_predict]
    for i,index in enumerate(cand_indexes):
        if i > num_to_predict:
            break
        #记录原始标签
        masked_lm_labels.append(enc[index])
        if rng.random()<0.8:
            enc[index] = 2
        else:
            if rng.random() < 0.5:
                enc[index] = rng.randint(4, vocab_len - 1)
        #记录mask的位置
        masked_lm_positions.append(index)

    masked_lm_weights = [1.0] * len(masked_lm_labels)
    masked_lm_weights = pad_sequences([masked_lm_weights], maxlen=max_predictions_per_seq, dtype='int32',
                                      padding='post')
    masked_lm_labels = pad_sequences([masked_lm_labels], maxlen=max_predictions_per_seq, dtype='int32',
                                      padding='post')
    masked_lm_positions = pad_sequences([masked_lm_positions], maxlen=max_predictions_per_seq, dtype='int32',
                                     padding='post')
    return enc,masked_lm_positions[0],masked_lm_labels[0], masked_lm_weights[0]

def shuffle_helper(features):
    pass


def get_batch(features, batch_size, shuffle=True):
    inputs_a, inputs_b, a_lens, b_lens, masked_positions_a, masked_labels_a, masked_weights_a,\
    masked_positions_b, masked_labels_b, masked_weights_b, related_labels = features
    instance_len = len(inputs_a)
    num_batches = calc_num_batches(instance_len, batch_size)

    if shuffle:
        indices = np.random.permutation(np.arange(instance_len))
        inputs_a = inputs_a[indices]
        inputs_b = inputs_b[indices]
        a_lens = a_lens[indices]
        b_lens = b_lens[indices]
        masked_positions_a = masked_positions_a[indices]
        masked_labels_a = masked_labels_a[indices]
        masked_weights_a = masked_weights_a[indices]
        masked_positions_b = masked_positions_b[indices]
        masked_labels_b = masked_labels_b[indices]
        masked_weights_b = masked_weights_b[indices]
        related_labels = related_labels[indices]


    for i in range(num_batches):
        start_id = i*batch_size
        end_id = min((i+1)*batch_size, instance_len)
        yield (inputs_a[start_id:end_id], inputs_b[start_id:end_id], a_lens[start_id:end_id],
              b_lens[start_id:end_id], masked_positions_a[start_id:end_id],
              masked_labels_a[start_id:end_id], masked_weights_a[start_id:end_id],
              masked_positions_b[start_id:end_id], masked_labels_b[start_id:end_id],
              masked_weights_b[start_id:end_id], related_labels[start_id:end_id])


if __name__ == '__main__':
    preprocessVec("./data/vec/glove.840B.300d.txt", "./data/snli.vocab", "./data/vec/snil_trimmed_vec.npy")
