import nltk
from nltk.corpus import wordnet as wn
import _pickle as pickle
import pandas as pd
import os
import re
from features.Tag_recommender.methods import sequencer
from constants.fixed_names import RETRIEVER_WORD_DIC, PROCESSED_TEXTS_FILE, VOCAB_FILE, DEEP_CLEAN_DATA_FILE
from collections import OrderedDict
nltk.download('wordnet')
nltk.download('omw')

# This Part is to test with chat like input interface
# while True:
#     input_query = input("Type your query: ")
#     if input_query and input_query is not " ":
#         tags, details = sequencer.hash_tag_generator(corpus, input_query)
#         if tags and len(tags) > 10:
#             max_matched = sequencer.get_maximum_longest_matched_chunk(input_query, synonyms_dict, corpus)
#             matched_tags = sequencer.synonym_matching_from_tags(tags, max_matched, synonyms_dict, input_query)
#             if matched_tags:
#                 print("Suggested Tags (From synonym matching): ", matched_tags)
#             else:
#                 print("Suggested Tags (All tags): ", tags)
#         elif tags:
#             print("Suggested Tags (All tags lesser than 10 tags): ", tags)
#         # break
#         else:
#             # print('Word/Sequence not found ')
#             # try:
#             tags, not_matched, max_matched = sequencer.unknown_word_sequence_handler(input_query, vocabulary, synonyms_dict, corpus)
#             # print('Details: ', tags, not_matched, max_matched)
#             # exit()
#
#             if tags and len(tags) > 10:
#                 not_matched_present = sequencer.check_not_matched_present_in_tags(tags, not_matched)
#                 matched_tags = sequencer.synonym_matching_from_tags(tags, max_matched, synonyms_dict, input_query)
#                 if not_matched_present:
#                     print("Suggested Tags (Not matched Present): ", not_matched_present)
#                 elif matched_tags:
#                     print("Suggested Tags (From synonym matching): ", matched_tags)
#                 else:
#                     print("Suggested Tags (All tags): ", tags)
#             elif 'Not Matched' not in tags:
#                 print("Suggested Tags (Tag found): ", tags)
#             else:
#                 print("Tag not found: ", tags)


def get_recommend(output_dir, data_dir, input_query):
    # タグリスト取得
    tags = get_recommend_tags(output_dir, data_dir, input_query)

    all_tag = []
    for tag in tags:
        all_tag.extend(tag.split())
    unique_tags = list(OrderedDict.fromkeys(sorted(all_tag, key=all_tag.count, reverse=True)))

    try:
        if len(unique_tags) > 0:
            query_list = input_query.split()
            for query in query_list:
                if query in unique_tags:
                    unique_tags.remove(query)

    except Exception as e:
        print("Exception occurs")

    return unique_tags[:3]


# タグリスト取得
def get_recommend_tags(output_dir, data_dir, input_query):
    # print("________\nInput Query: ", input_query)

    # corpus取得
    processed_texts = []
    with open(os.path.join(data_dir, PROCESSED_TEXTS_FILE)) as f:
        processed_texts = f.readlines()
    df = pd.DataFrame(processed_texts, columns=['Data'])
    df = df.reset_index()
    corpus = df.rename(columns={'index': 'PageID'})

    # synonym取得
    synonyms_dict = {}
    synonyms_file_path = os.path.join(data_dir, "vocabulary_synonyms.pk")
    if os.path.isfile(synonyms_file_path):
        synonyms_dict = load_from_file(synonyms_file_path)

    # vocabulary取得
    word_dic_file_path = os.path.join(output_dir, RETRIEVER_WORD_DIC)
    word_dic = load_from_file(word_dic_file_path)
    vocabulary = word_dic.keys()

    # if input_query and input_query is not " ":
    tags, details = sequencer.hash_tag_generator(corpus, input_query)
    if tags and len(tags) > 10:
        max_matched = sequencer.get_maximum_longest_matched_chunk(input_query, synonyms_dict, corpus)
        print("max_matched : ", max_matched)
        matched_tags = sequencer.synonym_matching_from_tags(tags, max_matched, synonyms_dict, input_query)
        print("matched_tags: ", matched_tags)
        if matched_tags:
            print("Suggested Tags (From synonym matching): ", matched_tags)
            tags = matched_tags
        else:
            print("Suggested Tags (All tags): ", tags)
    elif tags:
        print("Suggested Tags (All tags lesser than 10 tags): ", tags)
    else:
        print('Word/Sequence not found ')
        tags, not_matched, max_matched = sequencer.unknown_word_sequence_handler(input_query, vocabulary, synonyms_dict,
                                                                                 corpus)
        print('Details: ', tags, not_matched, max_matched)

        if tags and len(tags) > 10:
            not_matched_present = sequencer.check_not_matched_present_in_tags(tags, not_matched)
            matched_tags = sequencer.synonym_matching_from_tags(tags, max_matched, synonyms_dict, input_query)
            if not_matched_present:
                print("Suggested Tags (Not matched Present): ", not_matched_present)
                tags = not_matched_present
            elif matched_tags:
                print("Suggested Tags (From synonym matching): ", matched_tags)
                tags = matched_tags
            else:
                print("Suggested Tags (All tags): ", tags)
        elif 'Not Matched' not in tags:
            print("Suggested Tags (Tag found): ", tags)
        else:
            print("Tag not found: ", tags)
            tags = []

    return tags


def dump_to_file(file_path, data):
    with open(file_path, 'wb') as output_file:
        pickle.dump(data, output_file)


def load_from_file(file_path):
    with open(file_path, "rb") as input_file:
        data = pickle.load(input_file)
    return data


# vocabulary_synonyms
def generate_synonyms(output_dir, vocabulary):
    synonym_dict = {}

    for item in vocabulary:
        # シノニム取得
        syms = synonym_supplier(item)
        if syms:
            synonym_dict.update({item: syms})

    synonyms_file_path = os.path.join(output_dir, SYNONYMS_FILE)
    dump_to_file(synonyms_file_path, synonym_dict)


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


# スペリングチェック関連
# vocabulary_spelling
def vocabulary_spelling(output_dir, processed_texts):
    data_file = os.path.join(output_dir, DEEP_CLEAN_DATA_FILE)
    data = single_character_remover(" ".join(processed_texts))
    with open(data_file, 'w') as f:
        f.write(data)

    vocab_file = os.path.join(output_dir, VOCAB_FILE)
    vocabulary = list(set(data.split()))
    with open(vocab_file, 'w') as out:
        out.writelines("%s\n" % vocab for vocab in vocabulary)


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
