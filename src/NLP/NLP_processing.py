import pandas as pd
from nltk import RegexpTokenizer, PorterStemmer

from src.NLP.polish_stopwords import POLISH_STOPWORDS

output_file = 'data/processed/Translated_IMDB_Dataset_PROCESSED.csv'
tokenizer = RegexpTokenizer(r'\w+')
stemmer = PorterStemmer()


def preprocess_reviews(data):
    df = pd.DataFrame(data, columns=['review'])
    df['review'] = df['review'].apply(preprocess_text)
    df.to_csv(output_file, index=False)
    return [item for sublist in df.values.tolist() for item in sublist]


def preprocess_text(text):
    tokens = tokenizer.tokenize(text.lower())
    tokens = [stemmer.stem(word) for word in tokens]
    filtered_text = [word for word in tokens if word not in POLISH_STOPWORDS]
    preprocessed_text = ' '.join(filtered_text)
    return preprocessed_text
