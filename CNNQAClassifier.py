import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

import re
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
from anssel import Score
from collections import defaultdict
from sklearn import linear_model
from keras.models import load_model

vec_dim=300
sent_vec_dim=300
ans_len_cut_off=40
reg_feature_dim=7

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
    batch_size = 20
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
    print train_qsdata.shape, dev_qsdata.shape
    for c in range(0,epoch):
        model.fit({'qs_input': train_qsdata, 'ans_input': train_ansdata}, {'labels': train_label}, nb_epoch=1,
                  batch_size=batch_size)
        probs = model.predict([test_qsdata, test_ansdata], batch_size=batch_size)
        cal_score(ref_lines, probs)

def run_bag_of_words_model(train_samples,test_samples,ref_lines):

    train_qsdata, train_ansdata, train_label = load_bag_of_words_based_neural_net_data(train_samples)
    test_qsdata, test_ansdata, test_label = load_bag_of_words_based_neural_net_data(test_samples)
    run_neural_model(train_qsdata,train_ansdata,train_label,test_qsdata,test_ansdata,ref_lines)


def get_cnn_data(samples,max_qs_l,max_ans_l):
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

def get_max_len(train_samples,dev_samples,test_samples):
    max_qs_l = len(train_samples[0].QsWords)
    for i in range(1, len(train_samples)):
        if len(train_samples[i].QsWords) > max_qs_l:
            max_qs_l = len(train_samples[i].QsWords)

    for i in range(0, len(dev_samples)):
        if len(dev_samples[i].QsWords) > max_qs_l:
            max_qs_l = len(dev_samples[i].QsWords)

    for i in range(0, len(test_samples)):
        if len(test_samples[i].QsWords) > max_qs_l:
            max_qs_l = len(test_samples[i].QsWords)

    max_ans_l = len(train_samples[0].AnsWords)
    for i in range(1, len(train_samples)):
        if len(train_samples[i].AnsWords) > max_ans_l:
            max_ans_l = len(train_samples[i].AnsWords)

    for i in range(0, len(dev_samples)):
        if len(dev_samples[i].AnsWords) > max_ans_l:
            max_ans_l = len(dev_samples[i].AnsWords)

    for i in range(0, len(test_samples)):
        if len(test_samples[i].AnsWords) > max_ans_l:
            max_ans_l = len(test_samples[i].AnsWords)

    if max_ans_l > ans_len_cut_off:
        max_ans_l = ans_len_cut_off
    return max_qs_l, max_ans_l

def train_cnn(data_folder, max_qs_l, max_ans_l,
              train_qsdata, train_ansdata, train_labeldata,
              dev_qsdata, dev_ansdata,
              test_qsdata, test_ansdata,
              dev_ref_lines, test_ref_lines):
    Reduce = Lambda(lambda x: x[:, 0, :], output_shape=lambda shape: (shape[0], shape[-1]))

    qs_input = Input(shape=(max_qs_l, vec_dim,), dtype='float32', name='qs_input')
    qsconvmodel = Convolution1D(nb_filter=sent_vec_dim, filter_length=2, activation="tanh", border_mode='valid')(
        qs_input)
    qsconvmodel = AveragePooling1D(pool_length=max_qs_l - 1)(qsconvmodel)
    # qsconvmodel=MeanOverTime()(qsconvmodel)
    qsconvmodel = Reduce(qsconvmodel)

    qtm = Dense(output_dim=sent_vec_dim, activation='linear')(qsconvmodel)

    ans_input = Input(shape=(max_ans_l, vec_dim,), dtype='float32', name='ans_input')
    ansconvmodel = Convolution1D(nb_filter=sent_vec_dim, filter_length=2, activation="tanh", border_mode='valid')(
        ans_input)
    ansconvmodel = AveragePooling1D(pool_length=max_ans_l - 1)(ansconvmodel)
    # ansconvmodel=MeanOverTime()(ansconvmodel)
    ansconvmodel = Reduce(ansconvmodel)

    merged = merge([qtm, ansconvmodel], mode='dot', dot_axes=(1, 1))
    labels = Activation('sigmoid', name='labels')(merged)
    model = Model(input=[qs_input, ans_input], output=[labels])

    model.compile(loss={'labels': 'binary_crossentropy'},
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), metrics=['accuracy'])
    # Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
    # SGD(lr=0.01, momentum=0.9, nesterov=True)


    batch_size = 20
    epoch = 20
    best_MAP=-10.0
    best_model_file=os.path.join(data_folder,"best_cnn_model.h5")
    train_probs_epochs=[]
    dev_probs_epochs=[]
    test_probs_epochs=[]
    print train_qsdata.shape
    for epoch_count in range(0,epoch):

        model.fit({'qs_input': train_qsdata, 'ans_input': train_ansdata}, {'labels': train_labeldata}, nb_epoch=1,
              batch_size=batch_size)
        train_probs=model.predict([train_qsdata, train_ansdata], batch_size=batch_size)
        train_probs_epochs.append(train_probs)
        dev_probs=model.predict([dev_qsdata,dev_ansdata],batch_size=batch_size)
        dev_probs_epochs.append(dev_probs)
        test_probs = model.predict([test_qsdata, test_ansdata], batch_size=batch_size)
        test_probs_epochs.append(test_probs)
        MAP, MRR=cal_score(dev_ref_lines,dev_probs)
        if MAP > best_MAP :
            best_MAP=MAP
            model.save(best_model_file)

    best_model=load_model(best_model_file)

    train_probs = best_model.predict([train_qsdata, train_ansdata], batch_size=batch_size)
    dev_probs=best_model.predict([dev_qsdata, dev_ansdata], batch_size=batch_size)
    test_probs = best_model.predict([test_qsdata, test_ansdata], batch_size=batch_size)

    MAP, MRR = cal_score(test_ref_lines, test_probs)

    return MAP, MRR, train_probs_epochs, dev_probs_epochs, test_probs_epochs, train_probs, dev_probs, test_probs

def train_lr_using_dense_layer(reg_train_data_np, reg_dev_data_np, reg_test_data_np, train_labeldata, dev_ref_lines, test_ref_lines):
    reg_input = Input(shape=(reg_feature_dim,), dtype='float32', name='reg_input')
    reg_layer = Dense(output_dim=1)(reg_input)
    reg_output = Activation('sigmoid', name='reg_output')(reg_layer)
    reg_model = Model(input=[reg_input], output=[reg_output])
    reg_model.compile(loss={'reg_output': 'binary_crossentropy'},
                      optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                      metrics=['accuracy'])
    epoch=20
    batch_size=1
    best_MAP = -10.0
    best_model_file = os.path.join(data_folder, "best_lr_dense_model.h5")
    for epoch_count in range(0, epoch):
        reg_model.fit({'reg_input': reg_train_data_np}, {'reg_output': train_labeldata}, nb_epoch=1,
                      batch_size=batch_size)

        dev_probs = reg_model.predict([reg_dev_data_np], batch_size=batch_size)
        MAP, MRR=cal_score(dev_ref_lines,dev_probs)
        if MAP > best_MAP:
            best_MAP=MAP
            reg_model.save(best_model_file)
    best_model=load_model(best_model_file)
    test_probs=best_model.predict([reg_test_data_np], batch_size=batch_size)

    MAP, MRR = cal_score(test_ref_lines, test_probs)
    return MAP, MRR

def train_lr_using_sklearn(train_probs_epochs, dev_probs_epochs, test_probs_epochs, train_samples, dev_samples, test_samples, train_labeldata, stop_words, idf, dev_ref_lines, test_ref_lines):
    best_dev_MAP=-10.0
    best_test_MAP=0.0
    best_test_MRR=0.0
    for i in range(0,len(train_probs_epochs)):
        train_probs=train_probs_epochs[i]
        dev_probs=dev_probs_epochs[i]
        test_probs=test_probs_epochs[i]
        reg_train_data_np = get_lr_data(train_samples, train_probs, stop_words, idf)
        reg_dev_data_np = get_lr_data(dev_samples, dev_probs, stop_words, idf)
        reg_test_data_np = get_lr_data(test_samples, test_probs, stop_words, idf)
        clf = linear_model.LogisticRegression(C=0.01, solver='lbfgs')
        clf = clf.fit(reg_train_data_np, train_labeldata)
        lr_dev_preds = clf.predict_proba(reg_dev_data_np)
        dev_probs = []
        for lr_dev_pred in lr_dev_preds:
            dev_probs.append(lr_dev_pred[1])
        dev_MAP, dev_MRR = cal_score(dev_ref_lines, dev_probs)

        lr_test_preds = clf.predict_proba(reg_test_data_np)
        test_probs = []
        for lr_test_pred in lr_test_preds:
            test_probs.append(lr_test_pred[1])
        test_MAP, test_MRR = cal_score(test_ref_lines, test_probs)
        if dev_MAP > best_dev_MAP :
            best_dev_MAP=dev_MAP
            best_test_MAP=test_MAP
            best_test_MRR=test_MRR

    return best_test_MAP, best_test_MRR

def get_lr_data(samples, probs, stop_words, idf):
    reg_data = []
    data_index = 0
    for sample in samples:
        feat = cali_feature_extractor(sample.QsWords, sample.AnsWords, probs[data_index], stop_words, idf)
        reg_data.append(feat)
        data_index += 1

    reg_data_np = np.array(reg_data)
    return reg_data_np

def run_bigram_model(data_folder, train_samples, dev_samples, dev_ref_lines, test_samples, test_ref_lines, stop_words, idf):

    max_qs_l, max_ans_l=get_max_len(train_samples,dev_samples,test_samples)

    train_qsdata, train_ansdata, train_labeldata = get_cnn_data(train_samples, max_qs_l, max_ans_l)
    dev_qsdata, dev_ansdata, dev_labeldata=get_cnn_data(dev_samples, max_qs_l, max_ans_l)
    test_qsdata, test_ansdata, test_labeldata = get_cnn_data(test_samples, max_qs_l, max_ans_l)

    CNN_MAP, CNN_MRR, train_probs_epochs, dev_probs_epochs, test_probs_epochs, train_probs, dev_probs, test_probs=train_cnn(data_folder, max_qs_l, max_ans_l,
                                      train_qsdata, train_ansdata, train_labeldata,
                                      dev_qsdata, dev_ansdata,
                                      test_qsdata, test_ansdata,
                                      dev_ref_lines, test_ref_lines)


    reg_train_data_np=get_lr_data(train_samples,train_probs,stop_words,idf)
    reg_dev_data_np=get_lr_data(dev_samples,dev_probs,stop_words,idf)
    reg_test_data_np = get_lr_data(test_samples, test_probs, stop_words, idf)

    LR_Dense_MAP, LR_Dense_MRR = train_lr_using_dense_layer(reg_train_data_np, reg_dev_data_np, reg_test_data_np, train_labeldata, dev_ref_lines, test_ref_lines)

    LR_Sklearn_MAP, LR_Sklearn_MRR= train_lr_using_sklearn(train_probs_epochs, dev_probs_epochs, test_probs_epochs, train_samples, dev_samples, test_samples, train_labeldata,
                                                           stop_words, idf, dev_ref_lines, test_ref_lines)

    print "CNN"
    print "MAP:",CNN_MAP
    print "MRR:",CNN_MRR
    print "CNN-Cnt-Dense"
    print "MAP:", LR_Dense_MAP
    print "MRR:", LR_Dense_MRR
    print "CNN-Cnt-Sklearn"
    print "MAP:", LR_Sklearn_MAP
    print "MRR:", LR_Sklearn_MRR

def cal_score(ref_lines, probs):
    line_count = 0
    pred_lines = defaultdict(list)
    for ref_line in ref_lines:
        ref_line = ref_line.replace('\n', '')
        parts = ref_line.strip().split()
        qid, aid, lbl = int(parts[0]), int(parts[2]), int(parts[3])
        pred_lines[qid].append((aid, lbl, probs[line_count]))
        line_count += 1
    MAP=Score.calc_mean_avg_prec(pred_lines)
    MRR=Score.calc_mean_reciprocal_rank(pred_lines)
    return MAP, MRR

def clean_str(string):
    """
    Tokenization/string cleaning
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def build_idf(files, stopwords):
    idf = defaultdict(float)
    n = 0
    for fname in files:
        with open(fname, "rb") as f:
            for line in f:
                n += 1
                parts = line.strip().split("\t")
                question = clean_str(parts[0])
                words = set(question.split())
                for word in words:
                    if word in stopwords: continue
                    idf[word] += 1
    for word in idf:
        idf[word] = np.log(n / idf[word])
    return idf


def count_feature_extractor(qtoks, atoks, stop_words, idf):
    qset, aset = set(qtoks), set(atoks)
    count, weighted_count = 0.0, 0.0
    for word in qset:
        if word not in stop_words and word in aset:
            count += 1.0
            weighted_count += idf[word]
    return [count, weighted_count]


def cali_feature_extractor(qtoks, atoks, sim_probs, stop_words, idf):
    feat = count_feature_extractor(qtoks, atoks, stop_words, idf)
    feat.append(len(qtoks))
    feat.append(len(atoks))
    count, idf_sum = 1.0, 0.0
    for word in qtoks:
        if word not in stop_words:
            count += 1.0
            idf_sum += idf[word]
    feat.append(idf_sum / count)
    count, idf_sum = 1.0, 0.0
    for word in atoks:
        if word not in stop_words:
            count += 1.0
            idf_sum += idf[word]
    feat.append(idf_sum / count)
    feat.append(sim_probs)
    return feat

if __name__=="__main__":

    data_folder=sys.argv[1]
    word_vec_file =os.path.join(data_folder, sys.argv[2])
    stop_words_file = os.path.join(data_folder, sys.argv[3])
    train_file = os.path.join(data_folder, sys.argv[4])
    dev_file=os.path.join(data_folder, sys.argv[5])
    dev_ref_file=os.path.join(data_folder, sys.argv[6])
    test_file=os.path.join(data_folder, sys.argv[7])
    test_ref_file=os.path.join(data_folder, sys.argv[8])

    QASample=namedtuple("QASample","QsWords AnsWords Label")

    #vocab=load_vocab(train_file)
    #word_vecs=load_bin_vec(word_vec_file,vocab)
    word_vecs=load_word2vec(word_vec_file)
    #W,word_idx_map=get_W(word_vecs=word_vecs)
    stop_words=load_stop_words(stop_file=stop_words_file)

    files=[]
    files.append(train_file)
    files.append(test_file)
    idf=build_idf(files,stop_words)

    train_samples = load_samples(train_file)
    dev_samples=load_samples(dev_file)
    test_samples = load_samples(test_file)

    file_reader = open(dev_ref_file)
    dev_ref_lines = file_reader.readlines()
    file_reader.close()

    file_reader=open(test_ref_file)
    test_ref_lines=file_reader.readlines()
    file_reader.close()



    #Bar of words model
    #run_bag_of_words_model(train_samples,test_samples,ref_lines)

    #Bigram model
    run_bigram_model(data_folder, train_samples, dev_samples, dev_ref_lines, test_samples, test_ref_lines, stop_words, idf)

    print "Finished......"



















































