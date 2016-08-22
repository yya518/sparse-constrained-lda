from lda import LDA
from dataset import CNN, TwentyNewsDataset
import utils
import _lda
import sys
import numpy as np

class SparseLDA(LDA):
    def __init__(self, n_topics, alpha=0.1, beta=0.01, random_state=0):
        LDA.__init__(self, n_topics, alpha=0.1, beta=0.01, random_state=0)

    def __str__(self):
        return "Class type: Sparse LDA"

    def fit(self):
        """
        fit the model with X for one iteration
        """
        random_state = utils.check_random_state(self.random_state)
        rands = self._rands.copy()

        self.update_potential()
        #self.sample_topics_sparse(rands)
        _lda._sample_topics_sparse(self.WS, self.DS, self.ZS, self.nzw, self.ndz,
                            self.nz, self.alpha, self.beta, self.beta_sum, rands,
                            self.potential)


    def sample_topics_sparse(self, rands):
        value = 1.0
        smoothing_only_bucket = np.zeros(self.n_topics, dtype=float)
        topic_word_coef = np.zeros(self.n_topics, dtype=float)
        doc_topic_bucket = np.zeros(self.n_topics, dtype=float)
        topic_word_bucket = np.zeros(self.n_topics, dtype=float)
        smoothing_only_sum = 0.0
        d_idx = -1
        z_new = -1
        self.N = self.WS.shape[0]
        self.D = self.ndz.shape[0]
        n_rand = rands.shape[0]
        for k in range(n_topics):
            smoothing_only_bucket[k] = self.alpha * self.beta / (self.beta_sum + self.nz[k])
            smoothing_only_sum += smoothing_only_bucket[k]
        for k in range(n_topics):
            topic_word_coef[k] = self.alpha / (self.beta_sum + self.nz[k])

        for i in range(self.N):
            d = self.DS[i]
            if d != d_idx:  # a new document
                d_idx += 1
                doc_topic_sum = 0.0
                """ compute doc topic bucket & update topic word coefficient
                   r(z, d) = N(z|d) * beta / (beta * |V| + N(z))
                   q_coefficient(z, d) = (alpha(z) + N(z|d)) / (beta * |V| + N(z))
                """
                for k in range(n_topics):
                    if self.ndz[d, k] != 0:
                        doc_topic_bucket[k] = self.beta * self.ndz[d, k] / (self.beta_sum + self.nz[k])
                        doc_topic_sum += doc_topic_bucket[k]
                        topic_word_coef[k] = (self.alpha + self.ndz[d, k]) / (self.beta_sum + self.nz[k])
            w = self.WS[i]
            z = self.ZS[i]

            # remove word topic and update bucket values related to z
            self.nzw[z, w] -= 1
            self.ndz[d, z] -= 1
            self.nz[z] -= 1
            smoothing_only_sum -= smoothing_only_bucket[z]
            smoothing_only_bucket[z] = self.alpha * self.beta / (self.beta_sum + self.nz[z])
            smoothing_only_sum += smoothing_only_bucket[z]
            doc_topic_sum -= doc_topic_bucket[z]
            doc_topic_bucket[z] = self.beta * self.ndz[d, z] / (self.beta_sum + self.nz[z])
            doc_topic_sum += doc_topic_bucket[z]
            topic_word_coef[z] = (self.alpha + self.ndz[d, z]) / (self.beta_sum + self.nz[z])
            topic_word_sum = 0.0
            for k in range(n_topics):
                if self.nzw[k, w] != 0:
                    topic_word_bucket[k] = self.nzw[k, w] * topic_word_coef[k]
                    topic_word_sum += topic_word_bucket[k]

            # incorporate potential
            tmp_smoothing_only_bucket = smoothing_only_bucket
            tmp_smoothing_only_sum = smoothing_only_sum
            tmp_doc_topic_bucket = doc_topic_bucket
            tmp_doc_topic_sum = doc_topic_sum
            tmp_topic_word_bucket = topic_word_bucket
            tmp_topic_word_sum = topic_word_sum
            for k in range(n_topics):
                if self.potential[k, d, w] != 1:
                    tmp_smoothing_only_sum -= smoothing_only_bucket[k]
                    tmp_smoothing_only_bucket[k] = smoothing_only_bucket[k] * self.potential[k, d, w]
                    tmp_smoothing_only_sum += smoothing_only_bucket[k]

                    tmp_doc_topic_sum -= tmp_doc_topic_bucket[k]
                    tmp_doc_topic_bucket[k] = doc_topic_bucket[k] * self.potential[k, d, w]
                    tmp_doc_topic_sum += tmp_doc_topic_bucket[k]

                    tmp_topic_word_sum -= tmp_topic_word_bucket[k]
                    tmp_topic_word_bucket[k] = topic_word_bucket[k] * self.potential[k, d, w]
                    tmp_topic_word_sum += tmp_topic_word_bucket[k]

            #sample new topic
            total_mass = tmp_smoothing_only_sum + tmp_doc_topic_sum + tmp_topic_word_sum
            #print smoothing_only_sum, doc_topic_sum , topic_word_sum, total_mass,
            sample = rands[i % n_rand] * total_mass

            #print topic_word_bucket, topic_word_sum,
            #print total_mass, sample,
            if sample < tmp_topic_word_sum:
                for k in range(n_topics):
                    if self.nzw[k, w] != 0:
                        sample -= tmp_topic_word_bucket[k]
                        #print sample,
                        if sample <= 0:
                            z_new = k
                            break
            else:
                sample -= tmp_topic_word_sum
                if sample < tmp_doc_topic_sum:
                    for k in range(n_topics):
                        if self.ndz[d, k] != 0:
                            sample -= tmp_doc_topic_bucket[k]
                            if sample <= 0:
                                z_new = k
                                break
                else:
                    sample -= tmp_doc_topic_sum
                    for k in range(n_topics):
                        sample -= tmp_smoothing_only_bucket[k]
                        if sample <= 0:
                            z_new = k
                            break

            #print z_new
            self.ZS[i] = z_new
            # add word topic
            self.nzw[z_new, w] += 1
            self.ndz[d, z_new] += 1
            self.nz[z_new] += 1
            smoothing_only_sum -= smoothing_only_bucket[z_new]
            smoothing_only_bucket[z_new] = self.alpha * self.beta / (self.beta_sum + self.nz[z_new])
            smoothing_only_sum += smoothing_only_bucket[z_new]
            doc_topic_sum -= doc_topic_bucket[z_new]
            doc_topic_bucket[z_new] = self.beta * self.ndz[d, z_new] / (self.beta_sum + self.nz[z_new])
            doc_topic_sum += doc_topic_bucket[z_new]
            topic_word_coef[z_new] = (self.alpha + self.ndz[d, z_new]) / (self.beta_sum + self.nz[z_new])

            """if next token is in a new document, we need to update topic_word coefficient
            q_coefficient(z) = alpha(z) / (beta * |V| + N(z))
            """
            if i != self.N - 1 and self.DS[i + 1] > d:
                for k in range(n_topics):
                    if self.ndz[d, k] != 0:
                        topic_word_coef[k] = self.alpha / (self.beta_sum + self.nz[k])

if __name__ == '__main__':
    name = 'cnn'
    if name == 'cnn':
        dataset = CNN()
        dataset.load_data('/home/yi/Dropbox/workspace/data/cnn/')
        n_topics = 5
        n_iter = 200

    lda = SparseLDA(n_topics)
    lda.initialize(dataset.data_matrix)
    lda.load_label('labels.txt', dataset.dictionary)
    print lda.print_labels()

    for iter in range(n_iter):
        lda.fit()

    lda.fininsh()
    print lda.ndz[1, :]
    print lda.ndz[200, :]
    print lda.nzw[:, 386]#sandy
    print lda.nzw[:, 429]
    lda.print_top_words(dataset.dictionary, 10)
    print lda
