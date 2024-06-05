import pandas as pd
from nltk import RegexpTokenizer, PorterStemmer
from tqdm import tqdm

from src.NLP.polish_stopwords import POLISH_STOPWORDS

# Define PATHS
input_file = '../../data/processed/Translated_IMDB_Dataset_MERGED.csv'
output_file = '../../data/processed/Translated_IMDB_Dataset_PROCESSED.csv'


def preprocess_reviews(df):
    tokenizer = RegexpTokenizer(r'\w+')  # Regular expression tokenizer to extract alphanumeric tokens
    stemmer = PorterStemmer()

    def preprocess_text(text):
        tokens = tokenizer.tokenize(text.lower())

        tokens = [stemmer.stem(word) for word in tokens]

        filtered_text = [word for word in tokens if word not in POLISH_STOPWORDS]

        preprocessed_text = ' '.join(filtered_text)

        return preprocessed_text

    tqdm.pandas()
    df['review'] = df['review'].apply(preprocess_text)
    return df


if __name__ == "__main__":
    df = pd.read_csv(input_file)

    df_processed = preprocess_reviews(df)

    df_processed.to_csv(output_file, index=False)
