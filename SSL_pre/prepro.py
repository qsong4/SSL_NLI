import re
def prepro(train_file, dev_file, vocab_file):
    vocab_set = set()
    label_set = set()
    for f in (train_file, dev_file):
        with open(f, "r") as fr:
            for line in fr:
                content = line.strip().split()
                label = content[2]
                sent1 = content[0]
                sent2 = content[1]
                for i in sent1+sent2:
                    vocab_set.add(i)
                label_set.add(label)
    #0: <pad>, 1: <unk>, 2: <s>, 3: </s>
    with open(vocab_file, "w") as fw:
        fw.write("<pad>"+"\n")
        fw.write("<unk>"+"\n")
        for i in vocab_set:
            fw.write(i+'\n')
    print("labels: ", label_set)

def removePunc(inputStr):
    string = re.sub(r"\W+", "", inputStr)
    return string.strip()

def prepro_snli(train_file, dev_file, test_file, vocab_file):
    vocab_set = set()
    label_set = set()
    for f in (train_file, dev_file, test_file):
        with open(f, "r") as fr:
            for line in fr:
                content = line.strip().split("\t")
                label = content[2]
                sent1 = content[0]
                sent2 = content[1]
                sent = sent1 + ' ' + sent2
                for i in re.split(r"\W+", sent):
                    i = i.strip()
                    i = removePunc(i)
                    i = i.lower()
                    if i == "":
                        continue
                    vocab_set.add(i)
                label_set.add(label)
    #0: <pad>, 1: <unk>, 2: <mask>, 3:*
    with open(vocab_file, "w") as fw:
        fw.write("<pad>"+"\n")
        fw.write("<unk>"+"\n")
        fw.write("<mask>" + "\n")
        # fw.write("<cls>" + "\n")
        for i in vocab_set:
            fw.write(i+'\n')

    print("labels: ", label_set)



if __name__ == '__main__':
    train_file = "./data/snli_train.tsv"
    dev_file = "./data/snli_dev.tsv"
    test_file = "./data/snli_test.tsv"
    vocab_file = "./data/snli.vocab"
    #prepro(train_file, dev_file, vocab_file)
    prepro_snli(train_file, dev_file, test_file, vocab_file)
    #genvocabGlove("./data/vec/glove.840B.300d.txt", vocab_file)
