from sklearn.datasets import fetch_20newsgroups, fetch_20newsgroups_vectorized
import logging
logging.basicConfig(level=logging.ERROR)
from sklearn.feature_extraction.text import CountVectorizer
import os
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import sys
from pandas import DataFrame
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics

class Dataset:
    def __init__(self):
        self.data_matrix = None
        self.doc_labels = None
        self.word_labels = None

        self.dictionary = None
        self.doc_size = None
        self.word_size = None

        self.doc_class = None

    def load_data(self, path):
        raise NotImplementedError

    def load_word_labels(self, path):
        raise NotImplementedError

    def load_doc_labels(self, path):
        raise NotImplementedError

class CNN(Dataset):
    def __init__(self):
        Dataset.__init__(self)

    def get_class_name(self, file_dir_path):
        return file_dir_path[file_dir_path.rfind('/') + 1:]

    def extract_file_content(self, file_path):
        paragraphs = []
        with open(file_path, 'rt') as f:
            content = f.read().replace('\n', '').replace('&','')  # python xml parse cannot escape & symbol so we have to remove it
            tree = ET.fromstring(content)
        for node in tree.iter('paragraph'):
            paragraphs.append(node.text)
        return ' '.join(paragraphs)

    def read_files(self, path):
        '''return file content and it's classname
        '''
        for root, dir_names, file_names in os.walk(path):
            for path in dir_names:
                self.read_files(os.path.join(root, path))
            for file_name in file_names:
                file_path = os.path.join(root, file_name)
                yield self.extract_file_content(file_path), self.get_class_name(os.path.dirname(file_path))

    def load_data(self, path):
        rows = []
        for content, classname in self.read_files(path):
            rows.append({'text': content, 'class': classname})
        data_frame = DataFrame(rows)
        vectorizer = CountVectorizer(max_df=0.5, max_features=500, min_df=2, stop_words='english',
                                     analyzer='word', token_pattern='[^\W\d]+')
        X = vectorizer.fit_transform(data_frame['text'])

        self.data_matrix = X
        self.doc_labels = []
        self.word_labels = []

        self.doc_size = self.data_matrix.shape[0]
        self.word_size = self.data_matrix.shape[1]
        self.dictionary = vectorizer.get_feature_names()
        self.doc_class = data_frame['class']

    def load_doc_labels(self, path):
        self.label_map = {}
        with open(path, 'r') as f:
            for row in f.readlines():
                splits = row.split(',')
                self.label_map[splits[0]] = int(splits[1])
        self.doc_labels = [self.label_map[idx_target] for idx_target in self.doc_class]

    def load_word_labels(self, path):
        pass

class TwentyNewsDataset(Dataset):
    def __init__(self):
        Dataset.__init__(self)

    def load_doc_labels(self, path):
        pass

    def load_word_labels(self, path):
        pass

    def load_data(self, path=None):
        dataset = fetch_20newsgroups(subset='all')
        vectorizer = CountVectorizer(max_df=0.5, max_features=1000, min_df=2, stop_words='english',
                                     analyzer='word', token_pattern='[a-z]+')

        self.data_matrix = vectorizer.fit_transform(dataset.data)
        self.doc_labels = []
        self.word_labels = []
        self.doc_size = self.data_matrix.shape[0]
        self.word_size = self.data_matrix.shape[1]
        self.dictionary = vectorizer.get_feature_names()


if __name__ == '__main__':
    name = 'cnn'
    dataset = None
    if name == 'cnn':
        dataset = CNN()
        dataset.load_data('/home/yi/Dropbox/workspace/data/cnn/')
        dataset.load_doc_labels('/home/yi/Dropbox/workspace/data/cnn.doclabel.txt')
        dataset.load_word_labels('/home/yi/Dropbox/workspace/data/cnn.wordlabel.txt')
    elif name == '20news':
        dataset = TwentyNewsDataset()
        dataset.load_data()
    print dataset.word_size
    print dataset.dictionary