import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
# from cd_r.text_transformers import BM25Transformer
from text_transformers import BM25Transformer


class BM25Vectorizer(CountVectorizer):

    def __init__(self, input="content", encoding="utf-8", decode_error="strict", strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer="word", stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False,
                 dtype=np.float64, norm=None, use_idf=True, k1=1.5, b=0.75, floor=None,):

        super().__init__(input=input, encoding=encoding, decode_error=decode_error, strip_accents=strip_accents,
                         lowercase=lowercase, preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
                         stop_words=stop_words, token_pattern=token_pattern, ngram_range=ngram_range, max_df=max_df,
                         min_df=min_df, max_features=max_features, vocabulary=vocabulary, binary=binary, dtype=dtype,)

        self._bm25 = BM25Transformer(norm, use_idf, k1, b)

    @property
    def norm(self):
        return self._bm25.norm

    @norm.setter
    def norm(self, value):
        self._bm25.norm = value

    @property
    def use_idf(self):
        return self._bm25.use_idf

    @use_idf.setter
    def use_idf(self, value):
        self._bm25.use_idf = value

    @property
    def k1(self):
        return self._bm25.k1

    @k1.setter
    def k1(self, value):
        self._bm25.k1 = value

    @property
    def b(self):
        return self._bm25.b

    @b.setter
    def b(self, value):
        self._bm25.b = value

    @property
    def idf_(self):
        return self._bm25.idf_

    @idf_.setter
    def idf_(self, value):
        self._validate_vocabulary()
        if hasattr(self, "vocabulary_"):
            if len(self.vocabulary_) != len(value):
                raise ValueError(
                    "idf length = %d must be equal "
                    "to vocabulary size = %d" % (len(value), len(self.vocabulary))
                )
        self._bm25.idf_ = value

    def get_feature_names(self):
        """Array mapping from feature integer indices to feature name"""
        if not hasattr(self, 'vocabulary_'):
            self._validate_vocabulary()
        self._check_vocabulary()
        return [t for t, i in sorted(self.vocabulary_.items(), key=Items(1))]

    def fit(self, raw_documents, y=None):
        X = super().fit_transform(raw_documents)
        self._bm25.fit(X)
        return self

    def transform(self, raw_corpus, is_query=False):

        X = super().transform(raw_corpus) if is_query else None
        return self._bm25.transform(X, copy=False, is_query=is_query)

    def fit_transform(self, raw_documents, y=None):
        X = super().fit_transform(raw_documents)
        self._bm25.fit(X)
        return self._bm25.transform(X, copy=False)


class Items:
    """
    Return a callable object that fetches the given item(s) from its operand.
    After f = itemgetter(2), the call f(r) returns r[2].
    After g = itemgetter(2, 5, 3), the call g(r) returns (r[2], r[5], r[3])
    """
    __slots__ = ('_items', '_call')

    def __init__(self, item, *items):
        if not items:
            self._items = (item,)

            def func(obj):
                return obj[item]

            self._call = func
        else:
            self._items = items = (item,) + items

            def func(obj):
                return tuple(obj[i] for i in items)

            self._call = func

    def __call__(self, obj):
        return self._call(obj)

    def __repr__(self):
        return '%s.%s(%s)' % (self.__class__.__module__,
                              self.__class__.__name__,
                              ', '.join(map(repr, self._items)))

    def __reduce__(self):
        return self.__class__, self._items
