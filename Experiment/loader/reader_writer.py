from pathlib import Path
from os import path
from preprocessor import text_preprocessing
from constants.fixed_names import NLU_DICT, NLU_DATA_DIR, CORPUS_DIR, PROCESSED_TEXTS_FILE
import json
import ast


def load_data_list(processed_corpus):
    data_file = Path(processed_corpus)
    with open(data_file, encoding='utf-8') as f:
        data_list = f.read().splitlines()
    return data_list


def write_file(collector, directory, file_name):
    f_name = path.join(directory, file_name)
    with open(f_name, 'w') as f:
        for lines in collector.values():
            [f.write('{}\n'.format(line)) for line in lines if line]


def load_content(f_name):
    if not path.exists(path.join(CORPUS_DIR, NLU_DICT)):
        generate_corpus_dictionary(f_name)
    return loading_processed_data()


def loading_processed_data():
    file = open(path.join(CORPUS_DIR, NLU_DICT), "r")
    contents = file.read()
    corpus_dict = ast.literal_eval(contents)
    file.close()
    return corpus_dict


def save_data(data_dict):
    file = open(path.join(CORPUS_DIR, NLU_DICT), "w", encoding='utf-8')
    json.dump(data_dict, file, ensure_ascii=False)
    file.close()


def generate_corpus_dictionary(f_name):
    collector = {}
    for lines in text_preprocessing.intent_chunks(f_name).split('##'):
        if lines:
            intents = lines.split('-')
            saver = []
            [saver.append(' '.join(text_preprocessing.text_processing(items))) for items in intents[1:]]
            collector[intents[0].strip()] = saver
    return save_data(collector)


def load_answer(f_name):
    collector = {}
    for lines in text_preprocessing.intent_chunks(f_name).split('##'):
        if lines:
            intents = lines.split('-')
            collector[intents[0].strip().replace('\n', '').replace(':', '')] = intents[1:]
    return collector


def load_corpus(data_dir):
    processed_corpus = path.join(data_dir, PROCESSED_TEXTS_FILE)
    if not path.exists(processed_corpus):
        write_file(loading_processed_data(), CORPUS_DIR, PROCESSED_TEXTS_FILE)
    return load_data_list(processed_corpus)


def get_page_data(data, page_id):
    return data[int(page_id)]
