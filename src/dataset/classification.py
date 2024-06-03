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

    def BERT_train(self, dataset, num_epochs=3, batch_size=16, learning_rate=2e-5):
        # pre-trained BERT model
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # train
        self.model.train()
        for epoch in range(num_epochs):
            for batch in train_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch[1]

                optimizer.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = output.loss
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    def test(self, data):
        # evaluate
        self.model.eval()

        # predict
        with torch.no_grad():
            output = self.model(**data)
            logits = output.logits
            predictions = torch.argmax(logits, dim=1)

        return predictions.numpy()


if __name__ == '__main__':
    dataset = CustomDataset(data_path=data_path)

    X = dataset.data
    X_train, X_test, y_train, y_test = train_test_split(X, dataset.targets, test_size=0.2, random_state=42)

    # TF-IDF feature extraction
    X_train_tfidf = TF_IDF(X_train)
    X_test_tfidf = TF_IDF(X_test)

    # BERT Classification
    clf = Classification()
    clf.BERT_train(dataset)

    # Test the model
    train_dataset = CustomDataset(data=X_train_tfidf, targets=y_train)
    test_dataset = CustomDataset(data=X_test_tfidf, targets=y_test)

    train_result = clf.test(train_dataset)
    test_result = clf.test(test_dataset)

    print("Train accuracy:", accuracy_score(y_train, train_result))
    print("Test accuracy:", accuracy_score(y_test, test_result))
