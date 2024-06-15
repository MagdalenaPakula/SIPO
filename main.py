import os

import joblib

from src.dataset.CustomDataset import CustomDataset
from src.dataset.extraction import TF_IDF
from src.dataset.classification import Classification

from src.NLP.NLP_processing import preprocess_reviews

# Check if model and vectorizer already exist
model_path = 'models/model.joblib'
vectorizer_path = '../models/vectorizer.joblib'



if __name__ == '__main__':
    # dataset = CustomDataset(data_path="data/processed/Translated_IMDB_Dataset_MERGED.csv")
    #
    # dataset.data = preprocess_reviews(dataset.data)
    #
    # X_train, vectorizer = TF_IDF(dataset.data)
    #
    # train_dataset = CustomDataset(data=X_train, targets=dataset.targets)
    #
    # clf = Classification()
    # clf.SVM_train(train_dataset)
    #
    # joblib.dump(clf.model, 'model.joblib')
    # joblib.dump(vectorizer, 'vectorizer.joblib')

    os.path.exists(model_path)