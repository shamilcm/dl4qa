import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

import numpy as np
from collections import namedtuple
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Merge, Input, merge,Flatten,Lambda
from keras.layers.convolutional import Convolution1D
from keras.optimizers import SGD,Adam, Adagrad
from keras.layers.pooling import AveragePooling1D
#from Pooling import SumPooling1D,MeanOverTime
from keras import backend as K
import sys
from anssel import evaluator as Score
from collections import defaultdict

vec_dim=300
sent_vec_dim=300
ans_len_cut_off=40

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
    word_vecs["<unk>"] = np.random.uniform(-0.25, 0.25, vec_dim)
    return word_vecs

def load_word2vec(fname):
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
            word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
    word_vecs["<unk>"] = np.random.uniform(-0.25, 0.25, vec_dim)
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

def load_bag_of_words_based_neural_net_data(samples):
    qsdata = []
    ansdata = []
    labels = []
    for sample in samples:
        qsvec = get_bag_of_words_based_sentence_vec(sample.QsWords)
        ansvec=get_bag_of_words_based_sentence_vec(sample.AnsWords)
        qsdata.append(qsvec)
        ansdata.append(ansvec)
        labels.append(sample.Label)

    qsdata_nn = np.array(qsdata)
    ansdata_nn = np.array(ansdata)
    label_nn = np.array(labels)
    return qsdata_nn,ansdata_nn,label_nn

def get_bag_of_words_based_sentence_vec(words):
    vec = np.zeros(vec_dim, dtype='float32')
    word_count = 0
    for word in words:
        if stop_words.count(word) > 0:
            continue
        if word_vecs.has_key(word):
            vec += word_vecs[word]
        else:
            vec += word_vecs["<unk>"]
        word_count += 1
    #vec *= 100
    #vec /= word_count
    return vec

def run_neural_model(train_qsdata,train_ansdata,train_label,test_qsdata,test_ansdata,ref_lines):
    batch_size = 100
    epoch = 20

    qs_input = Input(shape=(vec_dim,), dtype='float32', name='qs_input')
    ans_input = Input(shape=(vec_dim,), dtype='float32', name='ans_input')
    qtm = Dense(output_dim=vec_dim, input_dim=vec_dim, activation='linear')(qs_input)
    merged = merge([qtm, ans_input], mode='dot', dot_axes=(1, 1))
    labels = Activation('sigmoid', name='labels')(merged)
    model = Model(input=[qs_input, ans_input], output=[labels])

    model.compile(loss={'labels': 'binary_crossentropy'}, optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                  metrics=['accuracy'])
    #SGD(lr=0.001, momentum=0.9, nesterov=True)

    for c in range(0,epoch):
        model.fit({'qs_input': train_qsdata, 'ans_input': train_ansdata}, {'labels': train_label}, nb_epoch=1,
                  batch_size=batch_size)
        probs = model.predict([test_qsdata, test_ansdata], batch_size=batch_size)
        line_count = 0
        pred_lines = defaultdict(list)
        for ref_line in ref_lines:
            ref_line = ref_line.replace('\n', '')
            parts = ref_line.strip().split()
            qid, aid, lbl = int(parts[0]), int(parts[2]), int(parts[3])
            pred_lines[qid].append((aid, lbl, probs[line_count]))
            line_count += 1
        print "Mean Avg. Precision: ", Score.calc_mean_avg_prec(pred_lines)
        print "Mean Reciprocal Rank: ", Score.calc_mean_reciprocal_rank(pred_lines)

def run_bag_of_words_model(train_samples,test_samples,ref_lines):

    train_qsdata, train_ansdata, train_label = load_bag_of_words_based_neural_net_data(train_samples)
    test_qsdata, test_ansdata, test_label = load_bag_of_words_based_neural_net_data(test_samples)
    run_neural_model(train_qsdata,train_ansdata,train_label,test_qsdata,test_ansdata,ref_lines)


def get_bigram_data(samples,max_qs_l,max_ans_l):
    qsdata = np.zeros(shape=(len(samples), max_qs_l, vec_dim), dtype="float32")
    ansdata = np.zeros(shape=(len(samples), max_ans_l, vec_dim), dtype="float32")
    labeldata = np.zeros(len(samples), dtype="int32")
    sent_count = 0
    for sample in samples:
        word_count = 0
        for word in sample.QsWords:
            if (word_vecs.has_key(word)):
                qsdata[sent_count][word_count] = word_vecs[word]
            else:
                qsdata[sent_count][word_count] = word_vecs["<unk>"]
            word_count += 1
        word_count = 0
        for word in sample.AnsWords:
            if (word_vecs.has_key(word)):
                ansdata[sent_count][word_count] = word_vecs[word]
            else:
                ansdata[sent_count][word_count] = word_vecs["<unk>"]
            word_count += 1
            if word_count==40:
                break
        labeldata[sent_count] = sample.Label
        sent_count += 1
    return qsdata,ansdata,labeldata

def run_bigram_model(train_samples, test_samples, ref_lines):
    max_qs_l = len(train_samples[0].QsWords)
    for i in range(1, len(train_samples)):
        if len(train_samples[i].QsWords) > max_qs_l:
            max_qs_l = len(train_samples[i].QsWords)

    for i in range(0, len(test_samples)):
        if len(test_samples[i].QsWords) > max_qs_l:
            max_qs_l = len(test_samples[i].QsWords)

    max_ans_l = len(train_samples[0].AnsWords)
    for i in range(1, len(train_samples)):
        if len(train_samples[i].AnsWords) > max_ans_l:
            max_ans_l = len(train_samples[i].AnsWords)

    for i in range(0, len(test_samples)):
        if len(test_samples[i].AnsWords) > max_ans_l:
            max_ans_l = len(test_samples[i].AnsWords)

    if max_ans_l > ans_len_cut_off:
        max_ans_l = ans_len_cut_off

    Reduce = Lambda(lambda x: x[:, 0, :], output_shape=lambda shape: (shape[0], shape[-1]))
    train_qsdata,train_ansdata,train_labeldata=get_bigram_data(train_samples,max_qs_l,max_ans_l)

    qs_input = Input(shape=(max_qs_l,vec_dim,), dtype='float32', name='qs_input')
    qsconvmodel=Convolution1D(nb_filter=sent_vec_dim,filter_length=2,activation="tanh", border_mode='valid')(qs_input)
    qsconvmodel=AveragePooling1D(pool_length=max_qs_l-1)(qsconvmodel)
    #qsconvmodel=MeanOverTime()(qsconvmodel)
    qsconvmodel=Reduce(qsconvmodel)

    qtm = Dense(output_dim=sent_vec_dim, activation='linear')(qsconvmodel)

    ans_input = Input(shape=(max_ans_l, vec_dim,), dtype='float32', name='ans_input')
    ansconvmodel=Convolution1D(nb_filter=sent_vec_dim, filter_length=2, activation="tanh", border_mode='valid')(ans_input)
    ansconvmodel=AveragePooling1D(pool_length=max_ans_l-1)(ansconvmodel)
    #ansconvmodel=MeanOverTime()(ansconvmodel)
    ansconvmodel=Reduce(ansconvmodel)

    merged = merge([qtm, ansconvmodel], mode='dot', dot_axes=(1, 1))
    labels = Activation('sigmoid', name='labels')(merged)
    model = Model(input=[qs_input, ans_input], output=[labels])

    model.compile(loss={'labels': 'binary_crossentropy'}, optimizer=Adagrad(lr=0.01, epsilon=1e-08, decay=0.0), metrics=['accuracy'])
    #Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
    #SGD(lr=0.01, momentum=0.9, nesterov=True)
    test_qsdata, test_ansdata, test_labeldata = get_bigram_data(test_samples,max_qs_l,max_ans_l)

    batch_size = 100
    epoch = 5

    for epoch_itr in range(0,epoch):
        model.fit({'qs_input': train_qsdata, 'ans_input': train_ansdata}, {'labels': train_labeldata}, nb_epoch=1,
                  batch_size=batch_size)

        probs = model.predict([test_qsdata, test_ansdata], batch_size=batch_size)

        line_count = 0
        pred_lines = defaultdict(list)
        for ref_line in ref_lines:
            ref_line = ref_line.replace('\n', '')
            parts = ref_line.strip().split()
            qid, aid, lbl = int(parts[0]), int(parts[2]), int(parts[3])
            pred_lines[qid].append((aid, lbl, probs[line_count]))
            line_count += 1
        print "Mean Avg. Precision: ", Score.calc_mean_avg_prec(pred_lines)
        print "Mean Reciprocal Rank: ", Score.calc_mean_reciprocal_rank(pred_lines)



if __name__=="__main__":

    #train_file = sys.argv[1]
    #word_vec_file=sys.argv[2]
    #stop_words_file=sys.argv[3]
    #test_file=sys.argv[4]
    #ref_file=sys.argv[5]

    train_file="data/WikiQASent-train.txt"
    word_vec_file="data/GoogleNews-vectors-negative300.bin"
    stop_words_file="data/short-stopwords.txt"
    test_file="data/WikiQASent-test-filtered.txt"
    ref_file="data/WikiQASent-test-filtered.ref"

    QASample=namedtuple("QASample","QsWords AnsWords Label")

    #vocab=load_vocab(train_file)
    #word_vecs=load_bin_vec(word_vec_file,vocab)
    word_vecs=load_word2vec(word_vec_file)
    #W,word_idx_map=get_W(word_vecs=word_vecs)
    stop_words=load_stop_words(stop_file=stop_words_file)

    train_samples = load_samples(train_file)
    test_samples = load_samples(test_file)

    file_reader=open(ref_file)
    ref_lines=file_reader.readlines()
    file_reader.close()

    #Bar of words model
    #run_bag_of_words_model(train_samples,test_samples,ref_lines)

    #Bigram model
    run_bigram_model(train_samples, test_samples, ref_lines)

    print "Finished......"



















































