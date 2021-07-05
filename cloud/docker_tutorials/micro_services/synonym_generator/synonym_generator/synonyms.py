import _pickle as pickle
from logger import logger
from os import path
from constants import PROCESSED_TEXTS_FILE, SYNONYMS_FILE, UNIQUE_WORD_DICT
from nltk.corpus import wordnet as wn


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


def add_users_synonyms(data_dir, user_synonyms):
    if not path.exists(path.join(data_dir, UNIQUE_WORD_DICT)):
        sym_dic, _ = make_unique_dict(data_dir)
    else:
        sym_dic = load_from_file(path.join(data_dir, UNIQUE_WORD_DICT))

    for key, val in user_synonyms.items():
        if key in sym_dic.keys():
            new_val = [sym for sym in val if sym not in sym_dic[key]]
            if new_val:
                sym_dic[key].extend(new_val)
        else:
            sym_dic.update({key: val})

    dump_to_file(path.join(data_dir, UNIQUE_WORD_DICT), sym_dic)
    return str("Synonym dictionary is rewritten successfully")
