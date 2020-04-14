import pandas as pd
import numpy as np
import sys
import re
sys.path.insert(0, '/home/iftekhar/myworkplace/AI-system/retrieval_Model/Page_Ranking_Experiment/methods_collection/')
import make_question as question_maker

def corpus_per_page(corpus):
    total_pages = corpus.PageID.unique()
    data = []
    PageID = []
    for i in list(total_pages):
        page_data = corpus[corpus.PageID == i].Data.values
        data.append(' '.join(list(page_data)))
        PageID.append(i)
    df = pd.DataFrame(zip(data, PageID), columns=["Data", "PageID"])
    return df

def corpus_split(corpus, sentence_length):
    labels = corpus.PageID.unique()
    lines = []
    all_ids = []
    for i in list(labels):
        text_list = corpus[corpus.PageID == i].Data.values
        split_text = fixed_length_sentence(' '.join(text_list), sentence_length)
        ids = [i] * len(split_text)
        lines += split_text
        all_ids += ids
    split_corpus = pd.DataFrame(zip(lines,all_ids), columns=["Data", "PageID"])
    return split_corpus

def clean_text(text):
    replaced = text.replace("\\","")
    replaced = replaced.replace("+","")
    replaced = re.sub('\W+',' ', replaced)
    replaced = re.sub(r'￥', '', replaced)       # 【】の除去
    replaced = re.sub(r'．', '', replaced)       # ・ の除去
    replaced = re.sub(r'｣', '', replaced)     # （）の除去
    replaced = re.sub(r'｢', '', replaced)   # ［］の除去
    replaced = re.sub(r'～', '', replaced)  # メンションの除去
    replaced = re.sub(r'｜', '', replaced)  # URLの除去
    replaced = re.sub(r'＠', '', replaced)  # 全角空白の除去
    replaced = re.sub(r'？', '', replaced) # 数字の除去
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
    return replaced

def fixed_length_sentence(contents, word_limit):
    contents_list = contents.split()
    end = len(contents_list)
    count = 0
    collector = []
    line = []
    for items in contents_list:
        if count < word_limit-1 and end > 1:
            collector.append(items)
            count += 1
        else:
            collector.append(items)
            line.append(' '.join(collector))
            collector = []
            count = 0
        end -= 1
    return line

def page_text_split(page_text, word_limit):
    page_text = page_text.split()
    chunks = [' '.join(page_text[i:i + word_limit]) for i in range(0,
                                                len(page_text), word_limit)]
    return chunks

def query_parsing(query):
    query_list = query.split()
    query_list = [items for items in query_list if items not in question_maker.get_questions_delimiter_ja()]
    query_text = ' '.join(query_list)
    return query_text
