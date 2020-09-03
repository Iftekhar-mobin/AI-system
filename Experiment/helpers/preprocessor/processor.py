from os import path
import pandas as pd
import re
import json
from nltk import word_tokenize
from nltk.corpus import stopwords
END_MARK = '(ですね|ですよ|ですか|です|でした|ょうか|ます|ますか|ません)'


def split_end_marker(end, items):
    pos = []
    marker = 0
    for match in re.finditer(r'' + end + '', items):
        if match:
            if match.start() > 20 and match.start() - marker > 20:
                pos.append(match)
        marker = match.start()

    collect_text = []
    if not pos:
        collect_text.append(items)
    else:
        if len(pos) > 1:
            flag = 0
            for match in pos:
                if not flag:
                    collect_text.append(items[:match.end()])
                    flag = 1
                else:
                    collect_text.append(items[marker:match.end()])
                marker = match.end()
            collect_text.append(items[marker:])
        else:
            match = pos[0]
            collect_text.append(items[:match.end()])
            collect_text.append(items[match.end():])
    return collect_text


def clean(text):
    replaced = text.replace("\\", "")
    replaced = replaced.replace("+", "")
    replaced = replaced.replace("①", "").replace("②", "").replace("③", "").replace("④",
                    "").replace("⑤", "").replace("⑥", "").replace("⑦", "").replace("⑧", "").replace("⑨", "")
    replaced = replaced.replace("ない", "")
    replaced = re.sub('_', '', replaced)
    replaced = re.sub('\W+', ' ', replaced)
    replaced = re.sub(r'￥', '', replaced)  # 【】の除去
    replaced = re.sub(r'．', '', replaced)  # ・ の除去
    replaced = re.sub(r'｣', '', replaced)  # （）の除去
    replaced = re.sub(r'｢', '', replaced)  # ［］の除去
    replaced = re.sub(r'～', '', replaced)  # メンションの除去
    replaced = re.sub(r'｜', '', replaced)  # URLの除去
    replaced = re.sub(r'＠', '', replaced)  # 全角空白の除去
    replaced = re.sub(r'？', '', replaced)  # 数字の除去
    replaced = re.sub(r'％', '', replaced)
    replaced = re.sub(r'＝', '', replaced)
    replaced = re.sub(r'！', '', replaced)
    replaced = re.sub(r'｝', '', replaced)
    replaced = re.sub(r'：', '', replaced)
    replaced = re.sub(r'－', '', replaced)
    replaced = re.sub(r'･', '', replaced)
    replaced = re.sub(r'ｔ', '', replaced)
    replaced = re.sub(r'ｋ', '', replaced)
    replaced = re.sub(r'ｄ', '', replaced)
    replaced = re.sub(r'\d+', '', replaced)

    return replaced


def cleaner(text):
    collector = []
    for items in text.split():
        cleaned = clean(items)
        cleaned = re.sub(r"\s+", '', cleaned)
        if cleaned is not '' or cleaned is not ' ':
            collector.append(clean(items))

    return ' '.join(collector)


def single_character_remover(text):
    collector = []
    for items in text.split():
        items = items.strip(' ')
        if len(items) < 2:
            replaced = re.sub(r'[ぁ-んァ-ン]', '', items)
            replaced = re.sub(r'[A-Za-z]', '', replaced)
            replaced = re.sub(r'[0-9]', '', replaced)
            collector.append(replaced)
        else:
            collector.append(items)

    return ' '.join([temp.strip(' ') for temp in collector])


# def get_stop_words_ja():
#     stop_word_file = Path(
#         "/home/iftekhar/amiebot/exp_amiecore/amieCore/amie_core/core/retriever/Tag_recommender/methods"
#         "/stop_word_ja.txt")
#     with open(stop_word_file, encoding='utf-8') as f:
#         stop_word_list = f.read().splitlines()
#     return stop_word_list


def token_collection(text):
    tokens = word_tokenize(text)
    return ' '.join([word for word in tokens if word not in stopwords.words()])


def clean_html(raw_html):
    pattern = re.compile('<.*?>|{.*?}|\"\>')
    return re.sub(pattern, '', raw_html)


class Preprocessor:
    def __init__(self, f_name, directory):
        self.corpus = pd.read_csv(path.join(directory, f_name))
        self.corpus_dict = self.sentence_maker()
        self.features_dict = self.get_features()
        self.save_data(directory)

    def save_data(self, directory):
        file = open(path.join(directory, "corpus_dict.json"), "w", encoding='utf-8')
        json.dump(self.corpus_dict, file, ensure_ascii=False)
        file.close()

        file = open(path.join(directory, "tokenized_dict.json"), "w", encoding='utf-8')
        json.dump(self.features_dict, file, ensure_ascii=False)
        file.close()

    def sentence_maker(self):
        if 'file' in self.corpus.columns:
            file_names = list(self.corpus.file.unique())
            iterator = file_names
            query = self.corpus.file
        else:
            page_ids = list(self.corpus.page.unique())
            iterator = page_ids
            query = self.corpus.page

        return self.lines_collection(iterator, query)

    def lines_collection(self, iterator, query):
        corpus_dict = {}
        i = 0
        for key in iterator:
            collector = []
            raw_text = clean_html(' '.join([j for j in list(self.corpus[query == key].text.values) if str(j) != 'nan']))
            page_text = raw_text.split('。')
            for sentence in page_text:
                for items in re.split(r'([A-Z]\.|\d+\.|→|〇|□|【|】|★|※|・|○|●|:|①|②|③|④|⑤|⑥|⑦|⑧|⑨|\(図\d+\))',
                                      sentence):
                    if items:
                        if len(items) > 100:
                            [collector.append(x) for x in split_end_marker(END_MARK, items)]
                        else:
                            collector.append(items)
            corpus_dict[str(i)] = collector
            i += 1
        return corpus_dict

    def get_features(self):
        feature_dict = {}
        for keys, sentences in self.corpus_dict.items():
            all_text = []
            for sentence in sentences:
                sentence = token_collection(sentence)
                sentence = single_character_remover(cleaner(sentence))
                all_text.append(sentence)
            feature_dict[keys] = all_text
        return feature_dict

# For debug
# Preprocessor('corpus.csv', '/home/iftekhar/amiebot/Resources/876/data')

