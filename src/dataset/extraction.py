from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def TF_IDF(data):
    tfidf = TfidfVectorizer()
    return tfidf.fit_transform(data)

def BoW(data):
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(data)

def WordEmbeddings(data):
    pass