from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import BertTokenizer
from nltk.tokenize import word_tokenize

from src.dataset.CustomDataset import CustomDataset
from src.dataset.classification import Classification
from src.dataset.extraction import TF_IDF

data_path = "../../data/processed/Translated_IMDB_Dataset_MERGED.csv"


class BERTTokenizer:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def tokenize(self, text):
        return self.tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')['input_ids'][0]


class NLTKTokenizer:
    def __init__(self):
        pass

    def tokenize(self, text):
        return word_tokenize(text)


if __name__ == '__main__':
    # Testing tokenizera
    bert_tokenizer = BERTTokenizer()

    dataset_bert = CustomDataset(data_path=data_path, tokenizer=bert_tokenizer)

    # Extraction for BERT tokenizer
    print("Extracting features with BERT Tokenizer...")
    X_bert = []
    for sample, _ in tqdm(dataset_bert, desc="Extracting Features", unit="sample"):
        X_bert.append(' '.join(map(str, sample)))
    X_bert = TF_IDF(X_bert)

    X_train_bert, X_test_bert, y_train_bert, y_test_bert = train_test_split(X_bert, dataset_bert.targets, test_size=0.2,
                                                                            random_state=42)

    train_dataset_bert = CustomDataset(data=X_train_bert, targets=y_train_bert)
    test_dataset_bert = CustomDataset(data=X_test_bert, targets=y_test_bert)

    # Classification for BERT tokenizer
    print("Training SVM for BERT Tokenizer...")
    clf_bert = Classification()
    clf_bert.SVM_train(train_dataset_bert)

    print("Testing SVM for BERT Tokenizer...")
    result_bert = clf_bert.test(test_dataset_bert.data)
    print("BERT Tokenizer Results")
    print(classification_report(test_dataset_bert.targets, result_bert))


    # # Load dataset with NLTK tokenizer
    # nltk_tokenizer = NLTKTokenizer()
    # dataset_nltk = CustomDataset(data_path=data_path, tokenizer=nltk_tokenizer)
    #
    # # Extraction for NLTK tokenizer
    # print("Extracting features with NLTK Tokenizer...")
    # X_nltk = []
    # for sample, _ in tqdm(dataset_nltk, desc="Extracting Features (NLTK)", unit="sample"):
    #     X_nltk.append(' '.join(map(str, sample)))
    # X_nltk = TF_IDF(X_nltk)
    #
    # X_train_nltk, X_test_nltk, y_train_nltk, y_test_nltk = train_test_split(X_nltk, dataset_nltk.targets, test_size=0.2,
    #                                                                         random_state=42)
    #
    # train_dataset_nltk = CustomDataset(data=X_train_nltk, targets=y_train_nltk)
    # test_dataset_nltk = CustomDataset(data=X_test_nltk, targets=y_test_nltk)
    #
    # # Classification for NLTK tokenizer
    # print("Training SVM for NLTK Tokenizer...")
    # clf_nltk = Classification()
    # clf_nltk.SVM_train(train_dataset_nltk)
    #
    # print("Testing SVM for NLTK Tokenizer...")
    # result_nltk = clf_nltk.test(test_dataset_nltk.data)
    # print("NLTK Tokenizer Results")
    # print(classification_report(test_dataset_nltk.targets, result_nltk))
