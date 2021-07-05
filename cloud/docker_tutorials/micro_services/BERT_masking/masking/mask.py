import torch
import time
import _pickle as pickle
from logger import logger
from os import path
from constants import PROCESSED_TEXTS_FILE, SYNONYMS_FILE, UNIQUE_WORD_DICT, RETRIEVER_WORD_DIC
from transformers import BertJapaneseTokenizer, BertForMaskedLM
from nltk.corpus import wordnet as wn


def load_bert(model_path):
    tok = BertJapaneseTokenizer.from_pretrained(model_path)
    masked_model = BertForMaskedLM.from_pretrained(model_path)
    return {"tokenizer": tok, 'model': masked_model}


def dump_to_file(file_path, data):
    with open(file_path, 'wb') as output_file:
        pickle.dump(data, output_file)


def load_from_file(file_path):
    with open(file_path, "rb") as input_file:
        data = pickle.load(input_file)
    return data


# vocabulary_synonyms
def synonym_supplier(word):
    # Wordnetから単語取得
    synsets = wn.synsets(word, lang='jpn')
    # シノニム取得
    syms_terms = [u.lemma_names('jpn') for u in list(synsets)]

    synonyms = []
    for items in syms_terms:
        for syms in items:
            synonyms.append(syms)

    return list(set(synonyms))


# vocabulary_synonyms
def generate_synonyms(output_dir, vocabulary):
    synonym_dict = {}
    updated_dict = {}
    uniques = []

    for item in vocabulary:
        # シノニム取得
        syms = synonym_supplier(item)
        if syms:
            synonym_dict.update({item: syms})

            # generate_unique_dict
            temp = [x for x in syms if x not in uniques]
            updated_dict.update({item: temp})
            uniques += temp

    synonyms_file_path = path.join(output_dir, SYNONYMS_FILE)
    dump_to_file(synonyms_file_path, synonym_dict)

    unique_dict_path = path.join(output_dir, UNIQUE_WORD_DICT)
    dump_to_file(unique_dict_path, updated_dict)


# リコメンドタグ関連
# vocabulary
def generate_vocabulary(processed_texts):
    vocabulary = []
    for sublist in processed_texts:
        vocabulary.extend(sublist.split(' '))
    vocabulary = list(set(vocabulary))

    return vocabulary


def generate_unique_dict(data_dir):
    if not path.exists(path.join(data_dir, UNIQUE_WORD_DICT)):
        return make_unique_dict(data_dir)
    else:
        updated_dict = load_from_file(path.join(data_dir, UNIQUE_WORD_DICT))
        return updated_dict, [x for save in [items for items in updated_dict.values()] for x in save]


def make_unique_dict(data_dir):
    uniques = []
    if not path.exists(path.join(data_dir, SYNONYMS_FILE)):
        logger.info('Predict uri is called without Mecab_tokenizer proper training.')
        logger.info('Please make synonym dictionary.')
        try:
            reader = open(path.join(data_dir, PROCESSED_TEXTS_FILE), 'r')
        except FileNotFoundError:
            logger.exception(f'{__name__} exception occurred.')
            logger.info('Please do the training and generate processed_text file.')
            return 0
        vocabulary = generate_vocabulary(reader.readlines())
        generate_synonyms(data_dir, vocabulary)
        synonyms_word_dic = load_from_file(path.join(data_dir, SYNONYMS_FILE))
    else:
        synonyms_word_dic = load_from_file(path.join(data_dir, SYNONYMS_FILE))

    updated_dict = {}
    for key, val in synonyms_word_dic.items():
        temp = [x for x in val if x not in uniques]
        updated_dict[key] = temp
        uniques += temp
    dump_to_file(path.join(data_dir, UNIQUE_WORD_DICT), updated_dict)
    return updated_dict, uniques


class Rewriter:
    def __init__(self, data_dir, model_dir):
        self.unique_dict, _ = generate_unique_dict(data_dir)
        word_dic_file_path = path.join(model_dir, RETRIEVER_WORD_DIC)
        self.word_dic = load_from_file(word_dic_file_path)

    def question_rewriter(self, query, model, tokenizer):
        collector = ''
        question = query.split()
        start = time.time()
        for word in question:
            if word not in self.word_dic.keys():
                masked_index = question.index(word)
                question[masked_index] = '[MASK]'
                indexed_tokens = tokenizer.convert_tokens_to_ids(question)
                tokens_tensor = torch.tensor([indexed_tokens])
                outputs = model(tokens_tensor)
                predictions = outputs[0][0, masked_index].topk(10)

                collection = []
                for i, index_t in enumerate(predictions.indices):
                    index = index_t.item()
                    collection.append(tokenizer.convert_ids_to_tokens([index])[0])

                neighbors = [item for item in collection if item in self.unique_dict.keys()]

                if neighbors:
                    print('BERT model is hit')
                    collector += neighbors[0] + ' '
            else:
                collector += word + ' '

        logger.info('Given query: {} and rewritten query {}'.format(' '.join(question), collector))
        logger.info("Time taken for context model predict: {}".format(time.time() - start))
        return collector.strip()
