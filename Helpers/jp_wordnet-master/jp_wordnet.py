# coding: utf-8
from nltk.corpus.reader.wordnet import WordNetCorpusReader
import nltk
import six


class JapaneseWordNetCorpusReader(WordNetCorpusReader):
    def __init__(self):
        self.cache = {}
        "データのロード"
        root = nltk.data.find('/home/iftekhar/nltk_data/corpora/wordnet')
        #        cd = os.path.dirname(__file__)
        #        if cd == "":
        #            cd = "."
        #        filename = cd+'/wnjpn-ok.tab'
        filename = '/home/iftekhar/nltk_data/corpora/wordnet/wnjpn-all.tab'
        WordNetCorpusReader.__init__(self, root, None)
        import codecs
        with codecs.open(filename, encoding="utf-8") as f:
            self._jword2offset = {}
            counter = 0
            for line in f:
                try:
                    _cells = line.strip().split('\t')
                    _offset_pos = _cells[0]
                    _word = _cells[1]
                    if len(_cells) > 2: _tag = _cells[2]
                    _offset, _pos = _offset_pos.split('-')
                    self._jword2offset[_word] = {'offset': int(_offset), 'pos': _pos}
                    counter += 1
                except:
                    print("failed to lead line %d" % counter)

    def synset(self, word):
        "synsetの取得"
        if word in self._jword2offset:
            return WordNetCorpusReader._synset_from_pos_and_offset(
                self, self._jword2offset[word]['pos'], self._jword2offset[word]['offset']
            )
        else:
            return None

    def printSimilarity(self, a, b):
        "類似度の表示"
        sim = self.similarity(a, b)
        if sim != None:
            print("「" + a + "」と「" + b + "」の類似度:", sim)
        else:
            print("「" + a + "」と「" + b + "」:辞書に無い単語を含みます")

    def similarity(self, a, b):
        "類似度の計算"
        # if not isinstance(a, six.string_types):
        #     a = unicode(a)
        # if not isinstance(b, six.string_types):
        #     b = unicode(b)
        # キャッシュに保存するために順番を統一
        if a > b:
            a, b = b, a
        # キャッシュに結果がのこっていないか調べる
        # if self.cache.has_key((a, b)):
        #     return self.cache[(a, b)]
        # 類似度の計算
        jsyn_a = self.synset(a)
        jsyn_b = self.synset(b)
        if jsyn_a and jsyn_b:
            sim = jsyn_a.path_similarity(jsyn_b)
        else:
            sim = None
        self.cache[(a, b)] = sim # キャッシュに結果の保存
        return sim

obj = JapaneseWordNetCorpusReader()
jsyn_apple = obj.synset(u'りんご')
jsyn_orange = obj.synset(u'ミカン')
print(jsyn_apple.path_similarity(jsyn_orange))
print(obj.printSimilarity('りんご', 'ミカン'))

# if __name__ == '__main__':
#    pass
