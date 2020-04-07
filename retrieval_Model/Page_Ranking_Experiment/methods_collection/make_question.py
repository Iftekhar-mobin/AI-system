import MeCab 
import pandas as pd
import re
import random
from pathlib import Path

mecab = MeCab.Tagger('-Owakati')

def question_dataframe_generator_1000(dataset, sample_size):
    dataset = dataset.sample(frac=1)
    
    question_saver = []
    PageID_saver = []
    count = 0
    for index, col in dataset.iterrows():
        question_saver.append(make_question_from_sentence(col["Data"]))
        PageID_saver.append(col["PageID"])
        if count > sample_size:
            break
        count += 1
    questions_samples = pd.DataFrame(zip(question_saver, PageID_saver), columns=['Question', 'PageID'])
    return questions_samples

def make_question_from_sentence(sentence):
    content = mecab.parse(sentence).strip("\n").rstrip()
    item_list = content.split()
    #item_list = list(set(item_list))
    #item_list = list(set(content))
    random_item_from_list = random.choice(item_list)
    item_list.remove(random_item_from_list)
    #random.shuffle(item_list)
    question_delimiter = random.choice(get_questions_delimiter_ja())
    item_list.append(question_delimiter)
    question = ' '.join(item_list)
    
    return question


def get_questions_delimiter_ja():
    questions_word = []
    questions_word_file = Path("/home/iftekhar/AI-system/JPBERT/questions_delimiter.txt")
    with open(questions_word_file, encoding='utf-8') as f:
        questions_word_list = f.read().splitlines()
    return questions_word_list

def make_question(split_lines_corpus):
    pages_list = []
    content_list = []
#     mecab_tokenizer = MecabTokenizer()
#     mecab_tokenizer._load_mecab()

    for pages in split_lines_corpus:
        for sentences in pages:
            pages_list.append(sentences[0])
        content_list.append(pages_list)
        pages_list = []

    count = 0
    all_question_list = []
    for pages in content_list:
        question_list = []

        for sent in pages:
            sent = str(sent)

            if re.search(r'\w+', sent):
                #content = mecab_tokenizer.wakati_base_form(sent)
                
                item_list = []
                content = mecab.parse(sent).strip("\n").rstrip()
                item_list = content.split()
                item_list = list(set(item_list))

                #item_list = list(set(content))
                random_item_from_list = random.choice(item_list)
                item_list.remove(random_item_from_list)
                #random.shuffle(item_list)

                question_delimiter = random.choice(get_questions_delimiter_ja())
                item_list.append(question_delimiter)

                question = ' '.join(item_list)
                question_list.append([question, count])

        count += 1
        all_question_list.append(question_list)

    labels = []
    text_list = []
    for i in all_question_list:
        for j in i:
            text_list.append(j[0])
            labels.append(j[1])
    question_data = pd.DataFrame(zip(text_list, labels), columns=['Question', 'PageID'])
    question_data = question_data.sample(frac=1).reset_index(drop=True)

    return question_data

def split_lines_corpus(per_page_corpus):
    corpus_list = []
    whole_corpus = []
    n = 100
    word_limit = 10
    for index, row in per_page_corpus.iterrows():
        var = str(row['Data']).split('。')

        for elements in var:
            if len(elements) > n:
                elements = elements.split()
                chunks = [' '.join(elements[i:i + word_limit]) for i in range(0, len(elements), word_limit)]
                for items in chunks:
                    corpus_list.append([items, row['PageID']])
            else:
                corpus_list.append([elements, row['PageID']])
        whole_corpus.append(corpus_list)
        corpus_list = []
    return whole_corpus

