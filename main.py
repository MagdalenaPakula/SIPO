import os

import joblib
from lime.lime_text import LimeTextExplainer
from sklearn.metrics import accuracy_score

from src.dataset.CustomDataset import CustomDataset
from src.dataset.extraction import TF_IDF
from src.dataset.classification import Classification

from src.NLP.NLP_processing import preprocess_reviews

model_path = 'src/models/model.joblib'
vectorizer_path = 'src/models/vectorizer.joblib'


def train_and_save_model():
    dataset = CustomDataset(data_path="data/processed/Translated_IMDB_Dataset_MERGED.csv")

    dataset.data = preprocess_reviews(dataset.data)

    X_train, vectorizer = TF_IDF(dataset.data)

    train_dataset = CustomDataset(data=X_train, targets=dataset.targets)

    clf = Classification()
    clf.SVM_train(train_dataset)

    joblib.dump(clf.model, 'model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')



if __name__ == '__main__':
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        print("Loading existing model and vectorizer...")
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
    else:
        train_and_save_model()


