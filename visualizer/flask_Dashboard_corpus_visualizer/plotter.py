import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from pathlib import Path
import re


def frequency_rank(frequency_dict):
    return sorted(frequency_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:20]


def make_bar_chart(df):
    fig = go.Figure(data=[go.Bar(x=df.Word, y=df.Frequency, text=df.Frequency)])
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    return fig


def make_pie_chart(df):
    fig = go.Figure(data=[go.Pie(labels=df.Word, values=df.Frequency)])
    fig.update_traces(textposition='outside', textinfo='percent+label')
    return fig


def make_stack_chart(df):
    return px.bar(df, x=df.columns[:-1], y=df.columns[-1], title="Query Distribution in Corpus")


def prepare_data_to_plot(df):
    df = df.reset_index()
    df['Query'] = df['index']
    df = df.drop(['index'], axis=1)
    col_name = df.columns[:-1]
    df.columns = ['Page_' + str(i) for i in col_name] + [df.columns[-1]]
    return df.loc[:, (df != 0).any(axis=0)]


def ranking_result(corpus_df):
    sum_rows = []
    for index, col in corpus_df.iterrows():
        adder = 0
        for column in corpus_df.columns:
            adder += col[column]
        sum_rows.append(adder)
    corpus_df['rank'] = sum_rows
    corpus_df = corpus_df.sort_values(by=['rank'], ascending=False).head(20)

    return corpus_df.drop(['rank'], axis=1).T


class PlotMaker:
    def __init__(self, corpus, page_id, query):
        self.query = query
        self.corpus_name = corpus
        self.page_id = page_id
        self.data = self.load_corpus()

    def driver(self):
        rank = frequency_rank(self.count_word_frequency())
        return pd.DataFrame(rank, columns=['Word', 'Frequency'])

    def load_corpus(self):
        data_file = Path(self.corpus_name)
        with open(data_file, encoding='utf-8') as f:
            data_list = f.read().splitlines()
        return data_list

    def corpus_to_frame(self):
        corpus = []
        for ids, content in enumerate(self.data):
            corpus.append([ids, content])
        corpus = pd.DataFrame(corpus, columns=['PageID', 'Data'])
        return corpus

    def count_word_frequency(self):
        page_data = self.get_page_data(self.data)
        word_freq = {}
        for token in page_data.split():
            if token not in word_freq.keys():
                word_freq[token] = 1
            else:
                word_freq[token] += 1
        return word_freq

    def get_page_data(self, data):
        return data[int(self.page_id)]

    def making_query_collection(self):
        query = self.query.lower()
        query_parts = query.split()
        question_parts = []
        for i in range(len(query_parts)):
            if len(query_parts) - 1 > i:
                question_parts.append(query_parts[i] + " " + query_parts[i + 1])
                if len(query_parts) - 2 > i:
                    question_parts.append(query_parts[i] + " " + query_parts[i + 1] + " " + query_parts[i + 2])
        question_parts = question_parts + query_parts + [query]
        return list(set(question_parts))

    def count_query_parts_frequency(self):
        word_freq_corpus = {}
        for items in self.making_query_collection():
            word_freq_corpus[items] = {}
            for index, col in self.corpus_to_frame().iterrows():
                temp = re.findall(r'\b' + items + r'\b', col['Data'])
                word_freq_corpus[items][col['PageID']] = len(temp)

        return prepare_data_to_plot(ranking_result(pd.DataFrame(word_freq_corpus)))


