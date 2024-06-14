import joblib
import numpy as np

from src.dataset.CustomDataset import CustomDataset
from src.dataset.extraction import TF_IDF, BoW
from src.dataset.classification import Classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.NLP.NLP_processing import preprocess_reviews


if __name__ == '__main__':
    dataset = CustomDataset(data_path="data/processed/Translated_IMDB_Dataset_MERGED.csv")
    # dataset = CustomDataset(data_path="data/raw/IMDB_Dataset.csv")

    dataset.data = preprocess_reviews(dataset.data)

    # X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.targets[:1000], test_size=0.2, random_state=42)

    X_train, vectorizer = TF_IDF(dataset.data)
    # X_test = vectorizer.transform(X_test)

    train_dataset = CustomDataset(data=X_train, targets=dataset.targets)
    # test_dataset = CustomDataset(data=X_test, targets=y_test)

    clf = Classification()
    clf.SVM_train(train_dataset)

    # test_X = vectorizer.transform(test_dataset.data)
    # result = clf.test(test_X)
    # print(classification_report(test_dataset.targets, result))

    # # Save the trained model and the vectorizer
    joblib.dump(clf.model, 'model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')