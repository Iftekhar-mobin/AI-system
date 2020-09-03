from nltk import word_tokenize
from preprocessor.processor import Preprocessor
from constants.fixed_names import PROCESSED_TEXTS_FILE, NLU_FILE_EN, NLU_DATA_DIR, CORPUS_DIR
from preprocessor import text_preprocessing
from loader import reader_writer
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.util import ngrams
import re
import random

TOKEN_SIZE = 3
REPLY_LEN = 30


def tokenize(query):
    return word_tokenize(query)[-3:]


# def load_data(corpus, directory):
#     if path.exists(path.join(directory, "corpus_dict.json")) and \
#             path.exists(path.join(directory, "tokenized_dict.json")):
#         return loading_processed_data(directory)
#     else:
#         print("Please wait, I am processing data and preparing dataset ...")
#         Preprocessor(corpus, directory)
#         return loading_processed_data(directory)


# def create_vector(features_dict, page_id, line):
#     return TfidfVectorizer().fit_transform(features_dict[str(page_id)] + [line])


def reply_generate(sequence_dict, query):
    curr_sequence = ' '.join(query)
    result = curr_sequence
    for i in range(REPLY_LEN):
        if curr_sequence not in sequence_dict.keys():
            break
        possible_words = sequence_dict[curr_sequence]
        result += ' ' + Counter(possible_words).most_common(1)[0][0]
        seq_words = result.split()
        curr_sequence = ' '.join(seq_words[len(seq_words) - TOKEN_SIZE:len(seq_words)])
    return result


class Predictor:
    def __init__(self, data_dir):
        self.model = defaultdict(lambda: defaultdict(lambda: 0))
        self.data = reader_writer.load_corpus(data_dir)
        # self.corpus_dict, self.features_dict = load_data(path.join(data_dir, DEFAULT_CORPUS), data_dir)
        self.features_dict = reader_writer.loading_processed_data()

        # self.data = load_corpus(data_dir)

    def driver(self, query):
        words = ' '.join(query).lower().split()
        if len(words) == 2:
            return self.predict_word(word1=words[0], word2=words[1])
        elif len(words) == 3:
            return self.predict_word(word1=words[0], word2=words[1], word3=words[2])
        # else:
        #     print("Out of scope")

    def sentence_model(self):
        for records in self.features_dict.values():
            for lines in records:
                for w1, w2, w3 in list(ngrams(re.sub('\s+', ' ', lines).split(), n=3)):
                    self.model[(w1, w2)][w3] += 1

    def count_frequency_w3(self):
        for sentence in self.data:
            sent = sentence.split()
            for w1, w2, w3, w4 in list(ngrams(sent, n=4)):
                self.model[(w1, w2, w3)][w4] += 1

    def count_frequency_w2(self):
        for sentence in self.data:
            sent = sentence.split()
            for w1, w2, w3 in list(ngrams(sent, n=3)):
                self.model[(w1, w2)][w3] += 1

    def estimate_probability(self):
        for word in self.model:
            total_count = float(sum(self.model[word].values()))
            for prediction in self.model[word]:
                self.model[word][prediction] /= total_count

    def predict_word(self, **kwargs):
        if len(kwargs.keys()) == 2:
            self.count_frequency_w2()
            self.estimate_probability()
            return sorted(dict(self.model[kwargs['word1'], kwargs['word2']]), reverse=True)
        elif len(kwargs.keys()) == 3:
            self.count_frequency_w3()
            self.estimate_probability()
            return sorted(dict(self.model[kwargs['word1'], kwargs['word2'], kwargs['word3']]), reverse=True)
        else:
            print("Support three words only")

    def sequence_frequency_dict(self):
        # words_tokens = reader_writer.get_page_data(self.data, page_id).split()
        words_tokens = self.data
        sequence_dict = {}
        for i in range(len(words_tokens) - TOKEN_SIZE):
            seq = ' '.join(words_tokens[i:i + TOKEN_SIZE])
            if seq not in sequence_dict.keys():
                sequence_dict[seq] = []
            sequence_dict[seq].append(words_tokens[i + TOKEN_SIZE])
        return sequence_dict

    # def chat_reply(self, page_id, query):
    #     lines = text_preprocessing.page_text_split(reply_generate(self.sequence_frequency_dict(page_id),
    #                                                               text_preprocessing.text_processing()), 10)
    #     saver = []
    #     for line in lines:
    #         matrix = create_vector(self.features_dict, page_id, line)
    #         idx = cosine_similarity(matrix[-1], matrix).argsort()[0][-2]
    #         saver.append(idx)
    #
    #     sentences = ''
    #     for idx in sorted(list(set(saver))):
    #         sentences += self.corpus_dict[str(page_id)][idx] + ' '
    #     return sentences

    def generate_sentence(self, query):
        keywords = query[-2:]
        collector = []
        i = 20
        while i > 0:
            end = False
            r = random.uniform(0.5, 1)
            breaker = 0
            while not end:
                adder = 0.0
                for word in self.model[tuple(keywords[-2:])].keys():
                    adder += self.model[tuple(keywords[-2:])][word]
                    if adder >= r:
                        keywords.append(word)
                        break
                if keywords[-2:] == [None, None] or breaker > 300 or len(keywords) > 20:
                    end = True
                breaker += 1
            collector.append(' '.join([t for t in keywords if t]))
            i -= 1
        return list(set(collector))

    def predict_sentence(self, query):
        self.sentence_model()
        self.estimate_probability()
        return self.generate_sentence(query)


obj = Predictor('/home/iftekhar/AI-system/Experiment/data')

print(reply_generate(obj.sequence_frequency_dict(), ['could', 'please', 'provide', 'payment']))

# while True:
#     q = input('Do have any Query?\n')
#     if q:
#         collector = []
#         out1 = obj.driver(text_preprocessing.text_processing(q))
#         out2 = obj.predict_sentence(text_preprocessing.text_processing(q))
#         if out1 or out2:
#             collector += out1
#             if len(out2) > 1 or out2[0].lower() != q.lower():
#                 collector += out2
#         if collector:
#             print(collector)
#         # print(reply_generate())
