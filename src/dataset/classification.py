from enum import Enum

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


class ClassificationMethod(Enum):
    NaiveBayes = 'NaiveBayes'
    SVM = 'SVM'
    BERT = 'BERT'


class Classification():

    def __init__(self):
        self.model = None
        self.method = ''

    def NaiveBayes_train(self, dataset):
        self.method = ClassificationMethod.NaiveBayes
        clf = MultinomialNB()
        clf.fit(dataset.data, dataset.targets)
        self.model = clf

    def SVM_train(self, dataset):
        self.method = ClassificationMethod.SVM
        clf = SVC(kernel='rbf')
        clf.fit(dataset.data, dataset.targets)
        self.model = clf

    def test(self, data):
        return self.model.predict(data)
