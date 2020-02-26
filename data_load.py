import tensorflow as tf
from utils import calc_num_batches
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import numpy as np
import re
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
import random

rng = random.Random(5)

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
    x = pad_sequences([x], maxlen=maxlen, dtype='int32',padding='post')
    #print(x)
    #x = [dict.get(t, dict["<unk>"]) for t in re.split(r"\W+'", inp)]
    return x[0]


def process_file(fpath, vocab_fpath, masklen, masked_lm_prob, max_predictions_per_seq, rng):
    inputs_a = []
    inputs_b = []
    masked_positions_a = []
    masked_labels_a = []
    masked_weights_a = []
    masked_positions_b = []
    masked_labels_b = []
    masked_weights_b = []
    related_labels = []

    token2idx, _, vocab = load_vocab(vocab_fpath)
    vocab_len = len(vocab)
    with open(fpath, 'r') as fr:

        sentences = list(fr.readlines())
        rng.shuffle(sentences)
        sent_len = len(sentences)
        for sent in sentences:
            content = sent.strip().split('\t')
            senta = content[0]
            sentb = content[1]
            label = 'related'

            #50%的概率是有关系的
            if rng.random() > 0.5:
                enc_a = encode(senta, token2idx, masklen)
                enc_b = encode(sentb, token2idx, masklen)

            else:
                label = 'not_related'
                enc_a = encode(senta, token2idx)
                for _ in range(10):
                    sentb_index = rng.randint(0, sent_len - 1)
                    if sentences[sentb_index] != sent:
                        break
                sentb = sentences[sentb_index].strip().split('\t')[1]
                enc_b = encode(sentb, token2idx)

            (enc_a, masked_lm_positions_a, masked_lm_labels_a, masked_lm_weights_a) \
                = create_masked_lm(enc_a, masked_lm_prob, max_predictions_per_seq, vocab_len,
                                   rng)

            (enc_b, masked_lm_positions_b, masked_lm_labels_b, masked_lm_weights_b) \
                = create_masked_lm(enc_b, masked_lm_prob, max_predictions_per_seq, vocab_len,
                                   rng)

            inputs_a.append(enc_a)
            inputs_b.append(enc_b)
            masked_positions_a.append(masked_lm_positions_a)
            masked_labels_a.append(masked_lm_labels_a)
            masked_weights_a.append(masked_lm_weights_a)
            masked_positions_b.append(masked_lm_positions_b)
            masked_labels_b.append(masked_labels_b)
            masked_weights_b.append(masked_weights_b)
            related_labels.append(label)
    return (inputs_a, inputs_b, masked_positions_a, masked_labels_a, masked_weights_a,
            masked_positions_b, masked_labels_b, masked_weights_b, related_labels)



def create_masked_lm(enc, masked_lm_prob, max_predictions_per_seq, vocab_len, rng):
    masked_lm_labels = []
    masked_lm_positions = []

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(enc) * masked_lm_prob))))
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
            enc[index] = 3
        else:
            if rng.random() < 0.5:
                enc[index] = rng.randint(4, vocab_len - 1)
        #记录mask的位置
        masked_lm_positions.append(index)

    masked_lm_weights = [1.0] * len(masked_lm_labels)

    return enc,masked_lm_positions,masked_lm_labels, masked_lm_weights

def shuffle_helper(features):
    pass


def get_batch(features, batch_size, rng, shuffle=False):
    inputs_a, inputs_b, masked_positions_a, masked_labels_a, masked_weights_a,\
    masked_positions_b, masked_labels_b, masked_weights_b, related_labels = features
    instance_len = len(inputs_a)
    num_batches = calc_num_batches(instance_len, batch_size)

    for i in range(num_batches):
        start_id = i*batch_size
        end_id = min((i+1)*batch_size, instance_len)
        yield inputs_a[start_id:end_id], inputs_b[start_id:end_id], masked_positions_a[start_id:end_id],\
              masked_labels_a[start_id:end_id], masked_weights_a[start_id:end_id],\
              masked_positions_b[start_id:end_id], masked_labels_b[start_id:end_id], \
              masked_weights_b[start_id:end_id], related_labels[start_id:end_id]


if __name__ == '__main__':
    preprocessVec("./data/vec/glove.840B.300d.txt", "./data/snli.vocab", "./data/vec/snil_trimmed_vec.npy")
