import torch
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW

from src.dataset.CustomDataset import CustomDataset
from src.dataset.extraction import TF_IDF

data_path = "data/processed/Translated_IMDB_Dataset_MERGED.csv"


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

    def test(self, data):
        return self.model.predict(data)
