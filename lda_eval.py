from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn import cross_validation
from sklearn import svm
from dataset import CNN
from lda import LDA

class Evaluator:
    def __init__(self, dataset, lda):
        self.dataset = dataset
        self.lda = lda
        self.doc_features = lda.doc_topic_
        self.doc_class = dataset.doc_class
        assert len(self.doc_class) == self.doc_features.shape[0]

    def clustering_measure(self, n_cluster):
        km = KMeans(n_cluster)
        km.fit(self.doc_features)
        print("Adjusted Rand-Index: %.3f"
              % metrics.adjusted_rand_score(self.doc_class, km.labels_))

    def cross_validation(self):
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            self.doc_features, self.doc_class, test_size=0.4, random_state=0)
        clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
        print ("Cross-Validation Score: %.3f" % clf.score(X_test, y_test))


if __name__ == '__main__':
    # load dataset
    dataset = CNN()
    dataset.load_data('/home/yi/Dropbox/workspace/data/cnn/')

    # train lda
    lda = LDA(5)
    lda.initialize(dataset.data_matrix)
    #lda.load_label('labels.txt', dataset.dictionary)
    for iter in range(20):
        lda.fit(dataset.data_matrix)
    lda.fininsh()
    lda.print_top_words(dataset.dictionary, 10)

    # evaluate lda
    eval = Evaluator(dataset, lda)
    eval.clustering_measure(n_cluster=5)
    eval.cross_validation()