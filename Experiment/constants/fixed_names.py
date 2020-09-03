import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = BASE_DIR + '/data'

DEFAULT_CORPUS = "corpus.csv"
PROCESSED_TEXTS_FILE = "processed_texts.txt"
DEEP_CLEAN_DATA_FILE = "deep_clean_data.txt"
VOCAB_FILE = "vocab.txt"

NLU_DATA_DIR = BASE_DIR + '/faq/data'
NLU_FILE_EN = 'amie_nlu_en.md'
NLU_FILE_JP = 'amie_nlu_ja.md'
NLU_DICT = 'nlu_dict.json'
REPLY_FILE_EN = 'reply_db.txt'

