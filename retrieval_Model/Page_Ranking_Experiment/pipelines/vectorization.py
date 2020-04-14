def vector_fit(vectorizer, split_corpus):
    vec = vectorizer.fit(split_corpus.Data.values.tolist())
    return vec
