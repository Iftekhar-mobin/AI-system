import pandas as pd
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from sklearn.base import BaseEstimator
from cd_r.vectorizers import BM25Vectorizer


class BaseRetriever(BaseEstimator, ABC):
    """
    Abstract base class for all Retriever classes.
    All retrievers should inherit from this class.
    Each retriever class should implement a _fit_vectorizer method and a
    _compute_scores method
    """

    def __init__(self, vectorizer, top_n=25, verbose=False):
        self.vectorizer = vectorizer
        self.top_n = top_n
        self.verbose = verbose

    @abstractmethod
    def _fit_vectorizer(self, df):
        pass

    @abstractmethod
    def _compute_scores(self, query):
        pass

    @abstractmethod
    def _get_features(self):
        pass

    def features(self):
        return self._get_features()

    def fit(self, df: pd.DataFrame, y=None):
        self.metadata = df
        return self._fit_vectorizer(df)

    def predict(self, query: str):
        t0 = time.time()
        # corpus_scores, question_vector = self._compute_scores(query)
        # return corpus_scores, question_vector

        scores = self._compute_scores(query)

        idx_scores = [(idx, score) for idx, score in enumerate(scores)]
        best_idx_scores = OrderedDict(
            sorted(idx_scores, key=(lambda tup: tup[1]), reverse=True)[: self.top_n]
        )

        return best_idx_scores


class BM25Retriever(BaseRetriever):

    def __init__(self, lowercase=True, preprocessor=None, tokenizer=None, stop_words="english",
                 token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1, 1), max_df=1, min_df=1, vocabulary=None,
                 top_n=25, verbose=False, k1=3, b=0.9, floor=None,):
        self.lowercase = lowercase
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.token_pattern = token_pattern
        self.ngram_range = ngram_range
        self.max_df = max_df
        self.min_df = min_df
        self.vocabulary = vocabulary
        self.k1 = k1
        self.b = b
        self.floor = floor

        vectorizer = BM25Vectorizer(lowercase=self.lowercase, preprocessor=self.preprocessor, tokenizer=self.tokenizer,
                                    stop_words=self.stop_words, token_pattern=self.token_pattern,
                                    ngram_range=self.ngram_range, max_df=self.max_df, min_df=self.min_df,
                                    vocabulary=self.vocabulary, k1=self.k1, b=self.b, floor=self.floor,)
        super().__init__(vectorizer, top_n, verbose)

    def _fit_vectorizer(self, df, y=None):
        self.bm25_matrix = self.vectorizer.fit_transform(df["content"])
        return self

    def _compute_scores(self, query):
        question_vector = self.vectorizer.transform([query], is_query=True)
        scores = self.bm25_matrix.dot(question_vector.T).toarray()
        return scores

    def _get_features(self):
        return self.vectorizer.get_feature_names()

    # def _compute_scores(self, query):
    #     question_vector = self.vectorizer.transform([query], is_query=True)
    #     corpus_scores = self.bm25_matrix.dot(question_vector.T).toarray()
    #     return corpus_scores, question_vector
