import numpy as np
import logging
import sys

class Dataset:

    def __init__(self, dataset):
        self.path = dataset
        self.vocab = self.load_vocab();
        self.samples = None
        self.labels = None

    def load_vocab(self):
        """
        Loads vocabulary from the training file
        """
        with open(self.path) as f:
            vocab = {}
            for line in f:
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

    def process(self, embeddings, stop_words):
        """
        Returns the training samples and labels as numpy array
        """
        s1samples_list = []
        s2samples_list = []
        labels_list = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                parts = line.split('\t')
                s1 = parts[0]           # Question part
                s2 = parts[1]           # Answer part
                label = int(parts[2])   # Label
                # each sample is a list of word vectors

                s1samples_list.append(self.get_sample(s1.split(), embeddings, stop_words))
                s2samples_list.append(self.get_sample(s2.split(), embeddings, stop_words))
                labels_list.append(label)

        self.samples = [s1samples_list,s2samples_list]
        self.labels = labels_list
   

    def get_sample(self, words, embeddings, stop_words):
        """
        Given a sentence, gets the input in the required format.
        """
        vecs = []
        vec = np.zeros(embeddings.emb_dim, dtype='float32')
        for word in words:
            if word in stop_words:
                continue
            vec = embeddings.get_word_vec(word)
            vecs.append(np.array(vec))
        if vecs == []:  # if all words are removed as stop words, send a 0 
            vecs.append(np.zeros(embeddings.emb_dim))
        return np.array(vecs)

class Embeddings:
    def __init__(self, w2v_fname, vocab):
        self.word_vecs, self.emb_dim = self.load_word2vec(w2v_fname, vocab)

    def load_word2vec(self, fname, vocab):
        """
        Loads word vecs from Google (Mikolov) word2vec. Assumes format is correct!
        """
        word_vecs = {}
        with open(fname, "rb") as f:
            header = f.readline()
            vocab_size, emb_dim = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * emb_dim
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
        word_vecs["<unk>"] = np.random.uniform(-0.25, 0.25, emb_dim)
        return word_vecs, emb_dim

    def get_word_vec(self, word):
        """
        Returns the word vector for a word
        """
        if word in self.word_vecs:
            return self.word_vecs[word]
        else:
            return self.word_vecs["<unk>"]

