from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def TF_IDF(data):
    tfidf = TfidfVectorizer(max_features=5000)
    transformed_data = tfidf.fit_transform(data)
    return transformed_data, tfidf


def BoW(data):
    vectorizer = CountVectorizer()
    transformed_data = vectorizer.fit_transform(data)
    return transformed_data, vectorizer
