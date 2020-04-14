import re
import difflib
from utility import get_unique_2Dlist
from corpus_handling_methods import query_parsing
import sys
sys.path.insert(0, sys.path.insert(0,
'/home/iftekhar/myworkplace/AI-system/retrieval_Model/Page_Ranking_Experiment/methods_collection/'))
from make_question import making_query_collection

FREQ_WORDS = ['し', 'ま', 'す', 'か', 'しま', 'カ', 'た']

def sequence_searcher(corpus_per_page, question_parts):
    collector = []
    for index, col in corpus_per_page.iterrows():
        for items in question_parts:
             if re.search(items, col['Data']):
                collector.append(col['PageID'])
    collector = list(set(collector))
    # print("Debug for Collector: ", collector)
    return collector

def sequence_matcher(sequence1, sequence2):
    matcher = difflib.SequenceMatcher(None, sequence1, sequence2)
    matches = matcher.get_matching_blocks()
    matching_result_collection = []
    for match in matches:
        if len(sequence1[match.a:match.a + match.size]) > 0:
            matching_result_collection.append(sequence1[match.a:match.a + match.size])
    return matching_result_collection


def perpage_sequence_match(collector, perpage_dataset, split_corpus, query):
    result = []
    res = []
    for pages_id in collector:
        page_data = perpage_dataset[(perpage_dataset['PageID']==pages_id)].Data.values
        result.append([sequence_matcher(query_parsing(query), str(page_data)), pages_id])

    for items, ids in result:
        contents = [terms for terms in items if terms not in FREQ_WORDS]
        contents = [temp.strip(' ') for temp in contents]
        if len(contents) > 1:
            res.append(ids)
    return get_sequence_match_ID(res, split_corpus, query)


def get_sequence_match_ID(res, split_corpus, query):
    match_collection = []
    for ids in res:
        page_data = split_corpus[(split_corpus['PageID']==ids)].Data.values
        for lines in page_data:
            for parts in making_query_collection(query):
                if lines.find(parts) !=-1:
                    match_collection.append([ids, lines])
    sorted_list = get_unique_2Dlist(match_collection)
    return sorted_list
