import joblib
from tqdm import tqdm

from src.dataset.CustomDataset import CustomDataset
from src.dataset.extraction import TF_IDF, BoW, WordEmbeddings
from src.dataset.classification import Classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


if __name__ == '__main__':
    print("Loading the data...")
    dataset = CustomDataset(data_path="data/processed/Translated_IMDB_Dataset_PROCESSED.csv")

    X, vectorizer = TF_IDF(dataset.data)

    X_train, X_test, y_train, y_test = train_test_split(X, dataset.targets, test_size=0.5, random_state=42)

    train_dataset = CustomDataset(data=X_train, targets=y_train)
    test_dataset = CustomDataset(data=X_test, targets=y_test)

    clf = Classification()
    print("Training the model...")
    # for _ in tqdm(range(100)):
    clf.SVM_train(train_dataset)

    print("Testing the model...")
    result = clf.test(test_dataset.data)
    print(classification_report(test_dataset.targets, result))

    # Save the trained model and the vectorizer
    print("Saving the model and vectorizer...")
    joblib.dump(clf, 'model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')
    print("Model saved as: model.joblib")
    print("Vectorizer saved as: vectorizer.joblib")