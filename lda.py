from dataset import CNN, TwentyNewsDataset
import numpy as np
import utils
import _lda
from dataset import CNN, TwentyNewsDataset
import time

import sys

class LDA:

    def __init__(self, n_topics, alpha=0.1, beta=0.01, random_state=0):
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.random_state = random_state
        rng = utils.check_random_state(random_state)
        self._rands = rng.rand(1024**2 // 8)

    def __str__(self):
        return "Class type: standard LDA"

    def initialize(self, X):
        self.D, self.W = D, W = X.shape
        self.beta_sum = self.W * self.beta

        # for labels
        self.doc_seed_labels = []
        self.word_seed_labels = []
        self.doc_pair_labels = []
        self.word_pair_labels = []
        self.potential = np.ones((self.n_topics, self.D, self.W))

        N = int(X.sum())
        self.nzw = np.zeros((self.n_topics, W), dtype=np.intc)
        self.ndz = np.zeros((D, self.n_topics), dtype=np.intc)
        self.nz = np.zeros(self.n_topics, dtype=np.intc)

        self.WS, self.DS = utils.matrix_to_lists(X)
        self.ZS = np.empty_like(self.WS, dtype=np.intc)
        #self.PS = np.zeros(N, dytpe=np.inc) #indicate if a word token has a different potential value
        np.testing.assert_equal(N, len(self.WS))
        for i in range(N):
            w, d = self.WS[i], self.DS[i]
            z_new = i % self.n_topics
            self.ZS[i] = z_new
            self.ndz[d, z_new] += 1
            self.nzw[z_new, w] += 1
            self.nz[z_new] += 1
        self.loglikelihoods_ = []

        print self.ndz.shape
        print self.nzw.shape

    def update_potential(self):
        for label in self.doc_seed_labels:
            doc = label[0]
            topic = label[1]
            self.potential[:, doc, :] = 0
            self.potential[topic, doc, :] = 1
        for label in self.word_seed_labels:
            word = label[0]
            topic = label[1]
            self.potential[:, :, word] = 0
            self.potential[topic, :, word] = 1
        for label in self.doc_pair_labels:
            doc1 = label[0]
            doc2 = label[1]
            for k in range(self.n_topics):
                self.potential[k, doc1, :] = self.ndz[doc2, k]
                self.potential[k, doc2, :] = self.ndz[doc1, k]
        for label in self.word_pair_labels:
            word1 = label[0]
            word2 = label[1]
            #self.potential[:, word1, :] = self.nzw[:, word1]
            for k in range(self.n_topics):
                self.potential[k, :, word1] = self.nzw[k, word2]
                self.potential[k, :, word2] = self.nzw[k, word1]

    def fit(self):
        """
        fit the model with X for one iteration
        """
        random_state = utils.check_random_state(self.random_state)
        rands = self._rands.copy()

        self.update_potential()
        _lda._sample_topics(self.WS, self.DS, self.ZS, self.nzw, self.ndz,
                            self.nz, self.alpha, self.beta, self.beta_sum, rands,
                            self.potential)

    def add_label(self, data_label, labelType):
        '''data_label is tuple'''
        if labelType == 'docseed':
            self.doc_seed_labels.append(data_label)
        if labelType == 'wordseed':
            self.word_seed_labels.append(data_label)
        if labelType == 'docpair':
            self.doc_pair_labels.append(data_label)
        if labelType == 'wordpair':
            self.word_pair_labels.append(data_label)

    def fininsh(self):
        self.get_topic_word()
        self.get_doc_topic()

        # delete attributes no longer needed after fitting to save memory and reduce clutter
        del self.WS
        del self.DS
        del self.ZS
        return self

    def loglikelihood(self):
        """Calculate complete log likelihood, log p(w,z)
        Formula used is log p(w,z) = log p(w|z) + log p(z)
        """
        nd = np.sum(self.ndz, axis=1).astype(np.intc)
        return _lda._loglikelihood(self.nzw, self.ndz, self.nz, nd, self.alpha, self.beta)

    def get_topic_word(self):
        self.components_ = (self.nzw + self.beta).astype(float)
        self.components_ /= np.sum(self.components_, axis=1)[:, np.newaxis]
        self.topic_word_ = self.components_

    def get_doc_topic(self):
        self.doc_topic_ = (self.ndz + self.alpha).astype(float)
        self.doc_topic_ /= np.sum(self.doc_topic_, axis=1)[:, np.newaxis]

    def print_top_words(self, feature_names, n_top_words):
        for topic_idx, topic in enumerate(self.components_):
            print("Topic #%d: " % topic_idx +
                  " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

    def print_labels(self):
        print 'doc seed: ' + str(self.doc_seed_labels)
        print 'doc pair: ' + str(self.doc_pair_labels)
        print 'word seed: ' + str(self.word_seed_labels)
        print 'word pair: ' + str(self.word_pair_labels)

    def load_label(self, file, feature_names):
        """load labels from file"""
        with open(file, 'r') as f:
            for line in f.readlines():
                if line[0] != '#':
                    splits = line.split(' ')
                    if splits[0] == 'wordseed':
                        self.add_label((feature_names.index(splits[1]), int(splits[2])), splits[0])
                    elif splits[0] == 'wordpair':
                        self.add_label((feature_names.index(splits[1]), feature_names.index(splits[2])), splits[0])
                    else:
                        self.add_label((int(splits[1]), int(splits[2])), splits[0])


if __name__ == '__main__':
    name = 'cnn'
    if name == 'cnn':
        dataset = CNN()
        dataset.load_data('/home/yi/Dropbox/workspace/data/cnn/')
        n_topics = 5
        n_iter = 20
    elif name == '20news':
        dataset = TwentyNewsDataset()
        dataset.load_data()
        n_topics = 20
        n_iter = 100

    lda = LDA(n_topics)
    lda.initialize(dataset.data_matrix)
    lda.load_label('labels.txt', dataset.dictionary)
    print lda.print_labels()

    start_time = time.time()
    for iter in range(n_iter):
        lda.fit()
    elapsed_time = time.time() - start_time
    print 'training time: ' + str(elapsed_time)
    lda.fininsh()
    lda.print_top_words(dataset.dictionary, 10)

