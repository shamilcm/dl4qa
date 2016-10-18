import numpy as np
from collections import namedtuple
from keras.models import Sequential, Model
from keras.layers import Dense,Activation,Merge,Input, Lambda,merge
from keras.optimizers import SGD
import keras.backend as K
import sys

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size, k))
    i = 0
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    word_vecs["<unk>"] = np.random.uniform(-0.25, 0.25, 300)
    return word_vecs

def load_vocab(vocab_file):
    file_reader = open(vocab_file)
    lines = file_reader.readlines()
    file_reader.close()
    vocab = {}
    for line in lines:
        parts = line.split('\t')
        qs = parts[0]
        ans = parts[1]
        qwords = qs.split()
        for word in qwords:
            if vocab.has_key(word):
                vocab[word] += 1
            else:
                vocab[word] = 1
        answords = ans.split()
        for word in answords:
            if vocab.has_key(word):
                vocab[word] += 1
            else:
                vocab[word] = 1
    return vocab

def load_samples(file):
    file_reader = open(file)
    lines = file_reader.readlines()
    file_reader.close()
    samples = []
    for line in lines:
        parts = line.split('\t')
        qs = parts[0]
        ans = parts[1]
        qwords = qs.split()
        answords = ans.split()
        label = int(parts[2].replace('\n', ''))
        sample = QASample(QsWords=qwords, AnsWords=answords, Label=label)
        samples.append(sample)
    return samples

def load_stop_words(stop_file):
    stop_words=[]
    file_reader=open(stop_file)
    lines=file_reader.readlines()
    for line in lines:
        line=line.replace('\n','')
        stop_words.append(line)
    return stop_words

def load_neural_net_data(samples):
    qsdata = []
    ansdata = []
    labels = []
    for sample in samples:
        qsvec = get_bag_of_words_vec(sample.QsWords)
        ansvec=get_bag_of_words_vec(sample.AnsWords)
        qsdata.append(qsvec)
        ansdata.append(ansvec)
        labels.append(sample.Label)

    qsdata_nn = np.array(qsdata)
    ansdata_nn = np.array(ansdata)
    label_nn = np.array(labels)
    return qsdata_nn,ansdata_nn,label_nn

def get_bag_of_words_vec(words):
    vec = np.zeros(300, dtype='float32')
    word_count = 0
    for word in words:
        if stop_words.count(word) > 0:
            continue
        if word_vecs.has_key(word):
            vec += word_vecs[word]
        else:
            vec += word_vecs["<unk>"]
        word_count += 1
    # vec *= 100
    # vec /= word_count
    return vec

if __name__=="__main__":

    train_file = sys.argv[1]
    word_vec_file=sys.argv[2]
    stop_words_file=sys.argv[3]
    test_file=sys.argv[4]

    QASample=namedtuple("QASample","QsWords AnsWords Label")

    vocab=load_vocab(train_file)
    word_vecs=load_bin_vec(word_vec_file,vocab)
    W,word_idx_map=get_W(word_vecs=word_vecs)
    stop_words=load_stop_words(stop_file=stop_words_file)

    train_samples = load_samples(train_file)
    train_qsdata,train_ansdata,train_label=load_neural_net_data(train_samples)

    test_samples=load_samples(test_file)
    test_qsdata,test_ansdata,test_label=load_neural_net_data(test_samples)

    batch_size=100
    epoch=20

    qs_input=Input(shape=(300,),dtype='float32',name='qs_input')
    ans_input = Input(shape=(300,),dtype='float32',name='ans_input')
    qtm = Dense(output_dim=300,input_dim=300,activation='linear')(qs_input)
    merged = merge([qtm, ans_input], mode='dot', dot_axes=(1, 1))
    labels=Activation('sigmoid',name='labels')(merged)
    model=Model(input=[qs_input,ans_input],output=[labels])

    model.compile(loss={'labels':'binary_crossentropy'}, optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True), metrics=['accuracy'])

    model.fit({'qs_input':train_qsdata,'ans_input':train_ansdata},{'labels':train_label},nb_epoch=epoch,batch_size=batch_size)
    probs = model.predict([test_qsdata, test_ansdata], batch_size=batch_size)

    correct_1 = 0
    total_1=0
    correct_0=0
    total_0=0
    cut_off=0.5
    for i in range(0, len(probs)):
        if test_label[i]==1:
            total_1 +=1
        if probs[i] >= cut_off and test_label[i] == 1:
            correct_1 += 1
        if test_label[i]==0:
            total_0 +=1
        if probs[i] < cut_off and test_label[i] == 0:
            correct_0 += 1

    print "Accuracy 1: ", float(correct_1) / total_1
    print "Accuracy 0: ", float(correct_0) / total_0
    print "Avg. Accuracy: ", (float(correct_1) / total_1 + float(correct_0) / total_0)/2

    print "Finished......"



















































