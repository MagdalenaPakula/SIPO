from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

class Classification():

    def __init__(self):
        self.model = None

    def NaiveBayes_train(self, dataset):
        clf = MultinomialNB()
        clf.fit(dataset.data, dataset.targets)
        self.model = clf

    def SVM_train(self, dataset):
        clf = SVC(kernel='rbf')
        clf.fit(dataset.data, dataset.targets)
        self.model = clf

    def BERT_train(self, dataset):
        pass

    def test(self, data):
        return self.model.predict(data)