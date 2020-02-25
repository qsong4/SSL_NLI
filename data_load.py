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


def process_file(fpath, vocab_fpath, masked_lm_prob, max_predictions_per_seq):
    token2idx, _, vocab = load_vocab(vocab_fpath)
    vocab_len = len(vocab)
    with open(fpath, 'r') as fr:

        sentences = list(fr.readlines())
        sentences = rng.shuffle(sentences)
        sent_len = len(sentences)
        for sent in sentences:
            content = sent.strip().split('\t')
            senta = content[0]
            sentb = content[1]
            label = 'related'

            #50%的概率是有关系的
            if rng.random() > 0.5:
                enc_a = encode(senta, token2idx)
                enc_b = encode(sentb, token2idx)

            else:
                label = 'not_related'
                enc_a = encode(senta, token2idx)
                for _ in range(10):
                    sentb_index = rng.randint(0, sent_len - 1)
                    if sentences[sentb_index] != sent:
                        break
                sentb = sentences[sentb_index].strip().split('\t')[1]
                enc_b = encode(sentb, token2idx)

            (enc_a, masked_lm_positions_a, masked_lm_labels_a) \
                = create_masked_lm(enc_a, masked_lm_prob, max_predictions_per_seq, vocab_len,
                                   rng)

            (enc_b, masked_lm_positions_b, masked_lm_labels_b) \
                = create_masked_lm(enc_b, masked_lm_prob, max_predictions_per_seq, vocab_len,
                                   rng)

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

    return enc,masked_lm_positions,masked_lm_labels



def get_batch(fpath, maxlen, vocab_fpath, batch_size, shuffle=False, char_maxlen=-1, char_vocab_fpath=None, with_char=False):

    sents1, sents2, labels = load_data(fpath, maxlen)
    batches = input_fn(sents1, sents2, labels, maxlen, vocab_fpath, batch_size, char_maxlen=char_maxlen, char_vocab_fpath=char_vocab_fpath, with_char=with_char, shuffle=shuffle)
    num_batches = calc_num_batches(len(sents1), batch_size)

    return batches, num_batches, len(sents1)


def get_batch_infer(sents1, sents2, maxlen, vocab_fpath, batch_size):
    batches = input_fn_infer(sents1, sents2, maxlen, vocab_fpath, batch_size)
    return batches

if __name__ == '__main__':
    #preprocessVec("./data/vec/glove.840B.300d.txt", "./data/snli.vocab", "./data/vec/snil_trimmed_vec.npy")
    #a = encode_char()
