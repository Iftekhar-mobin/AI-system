import random
# from amie_core.core.tokenizer.stop_words import get_questions_delimiter_ja, get_questions_delimiter_en
from preprocessor import text_preprocessing
from loader import reader_writer
from constants.fixed_names import NLU_DATA_DIR, NLU_FILE_JP, NLU_FILE_EN, REPLY_FILE_EN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from os import path


CUT_OFF = 0.2
MAX_COUNT = 3
RANK_START = 12
RANK_END = 2
TERM_FREQUENCY = 2
MAX_GRAM = 3

# import MeCab
# tokenizer_lib = MeCab.Tagger("-Owakati")
# nlp = spacy.load('ja_ginza')
# import en_core_web_sm
# nlp = en_core_web_sm.load()

# def tokenize(query):
#     tokenizer = MecabTokenizer()
#     tokenizer._load_mecab()
#     return tokenizer.wakati_base_form(query)


class FaqBot:
    def __init__(self):
        # for lang in [NLU_FILE_JP, NLU_FILE_EN]:
        #     self.nlu_file = path.join(NLU_FILE_DIR, lang)
        #     self.generator()

        self.nlu_file = path.join(NLU_DATA_DIR, NLU_FILE_EN)
        self.reply_file = path.join(NLU_DATA_DIR, REPLY_FILE_EN)
        self.content = reader_writer.load_content(self.nlu_file)
        self.reply_content = reader_writer.load_answer(self.reply_file)
        self.texts, self.index_list = self.catalog()
        self.vector = TfidfVectorizer(ngram_range=(1, MAX_GRAM), min_df=TERM_FREQUENCY)
        print('Data processed and loaded into memory.\n')

    def catalog(self):
        collect_texts = []
        data = list(self.content.values())
        index_list = []
        count = 0
        for ids, contents in enumerate(data):
            for i in contents:
                index_list.append([count, ids])
                count += 1
        for items in data:
            collect_texts += items
        return collect_texts, index_list

    def detect_intent(self, query):
        matrix = self.vector.fit_transform(self.texts + [' '.join(text_preprocessing.text_processing(query))])
        distance = cosine_similarity(matrix[-1], matrix)
        if sum(sorted(distance[0])[-MAX_COUNT:])/MAX_COUNT < CUT_OFF:
            return 'None'
        else:
            return self.intent(list(distance.argsort()[0][-RANK_START:-RANK_END]))

    def intent(self, result):
        save = []
        for serial, lines in enumerate(self.texts):
            if [x for x in result if x in [serial]]:
                for index, category in self.index_list:
                    if index == serial:
                        save.append(category)
        return list(self.content.keys())[Counter(save).most_common(1)[0][0]]

    def generate_reply(self, query):
        out = []
        for items in list(self.reply_content.keys()):
            if items.find(self.detect_intent(query)) is not -1:
                out.append(self.reply_content[items])
        if out:
            return random.choice(out[0]).strip().replace('\n', '')
        else:
            return 'Sorry I could not understand you.'


# obj = FaqBot()
#
# while True:
#     q = input('Do have any Query?\n')
#     if q:
#         print(obj.generate_reply(q))
#     # print(obj.detect_intent(q))


#     def content_maker(self, result):
#         file_contents = intent_chunks(self.nlu_file)
#         for lines, values in result.items():
#             for match in re.finditer(lines, file_contents):
#                 collector = '\n'
#                 end_flag = len(values)
#                 for sentences in values:
#                     if end_flag > 1:
#                         collector += '- ' + sentences + '\n'
#                     else:
#                         collector += '- ' + sentences
#                     end_flag -= 1
#                 file_contents = (file_contents[:match.end()] + collector + file_contents[match.end():])
#         return file_contents

#       self.file_rewriter()

#     def file_rewriter(self):
#         file = open(self.nlu_file, "w")
#         file.truncate(0)
#         file.write(self.content)
#         file.close()

# def question_maker(query, lang):
#     collector = []
#     question = query
#     count = 20
#     if lang == NLU_FILE_JP:
#         deli = get_questions_delimiter_ja()
#     else:
#         deli = get_questions_delimiter_en()
#
#     while count > 0:
#         shuffle_slice(question, 1, len(question))
#         question_delimiter = random.choice(deli)
#         collector.append(' '.join(question + [question_delimiter]))
#         question = query
#         count -= 1
#     return list(set(collector))
#
#
# def shuffle_slice(a, start, stop):
#     i = start
#     while i < stop - 1:
#         idx = random.randrange(i, stop)
#         a[i], a[idx] = a[idx], a[i]
#         i += 1



