import re
import requests
import json
from os import path
import pandas as pd
import numpy as np
import _pickle as pickle
from collections import Counter
from logger import logger
from constants import RETRIEVER_WORD_DIC, RECOMMENDER_CORPUS, CLEANED_RETRIEVER_WORD_DIC

TAG_LEN = 10
RESERVOIR_LEN = 5


def replace_with_regex(text, remove_mention=True, remove_url=True):
    # text = unicodedata.normalize('NFKC', text).lower()
    text = text.lower()
    replaced_text = re.sub(r'[【】「」／]', ' ', text)  # 【】「」／ の除去
    replaced_text = re.sub(r'[・_!！？?☛]', '', replaced_text)  # ・ の除去
    replaced_text = re.sub(r'[（）()]', ' ', replaced_text)  # （）の除去
    replaced_text = re.sub(r'[［］\[\]]', ' ', replaced_text)  # ［］の除去
    replaced_text = re.sub(r'　', ' ', replaced_text)  # 全角空白の除去
    replaced_text = re.sub(r'[⑤⑥②①③④⑦⑧⑨⑩]', '', replaced_text)
    # replaced_text = re.sub(r'\d+', '', replaced_text)  # 数字の除去
    replaced_text = re.sub(r'[/。,、.=]', ' ', replaced_text)  # others
    replaced_text = re.sub(r'[●■]', '', replaced_text)

    if remove_mention:
        replaced_text = re.sub(r'[@＠]\w+', '', replaced_text)  # メンションの除去
    if remove_url:
        replaced_text = re.sub(r'https?:\/\/.*?[\r\n ]', '', replaced_text)  # URLの除去
    return replaced_text


def clean_dict(words):
    garbage = []
    for x, y in words.items():
        if len(x) < 3 and bool(re.compile('[a-z]').match(x)):
            garbage.append(x)
        elif len(x) < 2:
            w = re.sub(r'[ぁ-んァ-ン]', '', x)
            w = re.sub(r'[a-z]', '', w)
            w = re.sub(r'[0-9]', '', w)
            w = re.sub(r'\W+', '', w)
            if not w:
                garbage.append(x)

    for x in garbage:
        words.pop(x, 'None')

    return words


def single_character_remover(text):
    collector = []
    for items in text.split():
        if len(items) < 2:
            replaced = re.sub(r'[ぁ-んァ-ン]', '', items)
            replaced = re.sub(r'[A-Za-z]', '', replaced)
            replaced = re.sub(r'[0-9]', '', replaced)
            collector.append(replaced)
        else:
            collector.append(items)

    return ' '.join([temp.strip(' ') for temp in collector])


def dump_to_file(file_path, data):
    with open(file_path, 'wb') as output_file:
        pickle.dump(data, output_file)


def tokenize(query, noun=False):
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    API_ENDPOINT = "http://127.0.0.1:5000/api/v1.0/predict/"

    r = requests.post(url=API_ENDPOINT, data=json.dumps(query), headers=headers)
    r = json.loads(str(r.text))
    if r:
        text = r['keyword']
    else:
        text = ''

    if noun:
        # return tokenizer.get_nouns(query)
        return single_character_remover(text).split()
    else:
        return single_character_remover(text).split()


class Recommender:
    def __init__(self, model_dir, data_dir):

        if path.exists(path.join(model_dir, CLEANED_RETRIEVER_WORD_DIC)):
            self.word_dic = pd.read_pickle(path.join(model_dir, CLEANED_RETRIEVER_WORD_DIC))
        else:
            self.word_dic = clean_dict(pd.read_pickle(path.join(model_dir, RETRIEVER_WORD_DIC)))
            dump_to_file(path.join(model_dir, CLEANED_RETRIEVER_WORD_DIC), self.word_dic)

        self.keys = np.array(list(self.word_dic.keys()))
        self.corpus = pd.read_pickle(path.join(data_dir, RECOMMENDER_CORPUS))
        logger.debug('Dictionary+Corpus Loaded for sequence recommend')

    def footprint(self, query):
        collector = []
        query_tokens = tokenize(query)
        for items in [i for i in query_tokens if i in self.keys]:
            try:
                collector += self.word_dic[items]
            except KeyError:
                pass
        return collector, query_tokens

    def ranker(self, query):
        collect = []
        rank = []
        mapping, query_tokens = self.footprint(query)
        if mapping:
            rank = sorted(mapping, key=lambda x: len(x[1]), reverse=True)[:RESERVOIR_LEN]
            for index, items in rank:
                collect.append(index)
        return collect, rank, query_tokens

    def tag_generate(self, query):
        collect, rank, query_tokens = self.ranker(query)
        df = self.corpus.loc[self.corpus['page'].isin(collect)]
        collector = []
        for page, index in rank:
            for row, col in df.iterrows():
                if page == col['page']:
                    for pos in index:
                        collector.append(col['text'][pos:pos + 2*TAG_LEN])
        return collector, query_tokens

    def get_recommend(self, input_query):
        all_tag = []
        collector, query_tokens = self.tag_generate(input_query)

        for tag in [replace_with_regex(x, True, False) for x in
                    [w for w, _ in Counter(collector).most_common(2*TAG_LEN)]]:
            all_tag.extend(tokenize(tag, noun=True)[:-1])

        unique_tags = sorted(list(set(all_tag)), key=len, reverse=True)[:RESERVOIR_LEN]

        try:
            if len(unique_tags) > 0:
                query_list = query_tokens
                for query in query_list:
                    if query in unique_tags:
                        unique_tags.remove(query)

        except Exception as e:
            logger.info(str(e))
            logger.exception(f'{__name__} Exception occurred')

        return unique_tags[:3]
