from sklearn.metrics.pairwise import cosine_similarity
#from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing
import difflib
import re
import pickle
import pandas as pd
#from nltk.tokenize import word_tokenize
from make_question import making_query_collection


def build_voc(dataset):
    corpus_text = ' '.join(dataset.Data)
    corpus_list = corpus_text.split()
    words_list = list(set(corpus_list))
    word_ids = list(enumerate(words_list))
    word_ids = sorted(word_ids, key=lambda l:l[1], reverse=True)

    with open('/home/iftekhar/AI-system/retrieval_Model/Page_Ranking_Experiment/customized_model/vocabulary.pkl', 'wb') as f:
        pickle.dump(word_ids,f)

def make_vector(dataset, word_ids):
    text_vectors=[]
    page_ids = []
    for index, col in dataset.iterrows():
        text_vectors.append(custom_vectorizer(str(col['Data']).split(), word_ids))
        page_ids.append(col['PageID'])

    vector = list(zip(text_vectors, page_ids))
    # Not writing on disk because already it is present there
    with open('/home/iftekhar/AI-system/retrieval_Model/Page_Ranking_Experiment/customized_model/vector.pkl', 'wb') as f:
        pickle.dump(vector ,f)

# Vectorizer Just Words -> IDs
def custom_vectorizer(input_list, word_ids):
    collector = []
    for token in input_list:
        for key, value in word_ids:
            if value == token:
                # print(value, "->", key)
                collector.append(key)
                break
    return collector



def return_match_length(list_a, list_b):
    a = set(list_a)
    b = set(list_b)
    saver = [list(b - a), list(a - b)]
    not_matched = [x for sublist in saver for x in sublist]
    matched = [items for items in list_a+list_b if items not in not_matched]
    return len(matched)

def ranking_result_collection(query, text_collections, ids_list, vec):
    X = vec.transform([str(query)])

    rank=[]
    for contents in text_collections:
        Y = vec.transform([contents])
        rank.append(cosine_similarity(X, Y))
    flat = [x for sublist in rank for x in sublist]
    result = sorted(zip(ids_list, flat), key=lambda l:l[1], reverse=True)[:3]
    return result

def filtered_vector(collectors, vectorizer, dataset):
    text_collections=[]
    ids_list=[]
    for match, ids in collectors:
        temp = dataset[dataset.PageID == str(ids)].Data.values
        temp_list = ' '.join(temp.tolist())
        text_collections += [temp_list]
        ids_list.append(ids)
    vec = vectorizer.fit(text_collections)
    return vec, text_collections, ids_list

def collecting_few_results(rank_list, limit = 10):
    collectors = []
    cut=0
    temp=None
    for items, ids in rank_list:
        if items == 0:
            continue
        elif items!=temp and cut>limit:
            break
        elif items==temp:
            collectors.append([items,ids])
        else:
            cut+=1
            temp = items
            collectors.append([items,ids])
    return collectors

def rank_list_collection(result_collection, dataset):
    matching_frequency = []
    ids_collection = []
    for matches, ids in result_collection:
        matching_frequency.append(matches)
        ids_collection.append(ids)
    rank_list = sorted(zip(matching_frequency,ids_collection), reverse=True)[:len(dataset)]
    rank_list=list(set(rank_list))
    rank_list.sort(reverse=True)
    return rank_list



def sequence_wise_ranking(match_list_collection):
    adder = 0
    score_collection = []
    for i, ids in match_list_collection:
        for items in i:
            if len(items) > 1:
                adder += len(items)
            else:
                adder += 0.3
        score_collection.append([adder,ids])
        adder = 0
    return sorted(score_collection,key=lambda l:l[0], reverse=True)[:30]


def sequence_handler(collectors, vector, query_vector):
    vector_collections=[]
    ids_list=[]
    for match, ids in collectors:
        flag = 0
        for v, i in vector:
            if int(i) == int(ids):
                vector_collections.append([v, i])
                flag = 1
            elif flag == 1:
                break
    match_list_collection = []
    for items_vector, ids in vector_collections:
        match_list_collection.append([sequence_matcher(items_vector, query_vector),ids])
    sequence_wise_rank = sequence_wise_ranking(match_list_collection)
    sequence_wise_rank = get_unique_2Dlist(sequence_wise_rank)
    return sequence_wise_rank

def gensim_trainer(collectors, dataset):
    text_collections=[]
    ids_list=[]
    for match, ids in collectors:
        text_collections.append(dataset["Data"][ids])
        ranks = get_score_details_record(df, questions_samples, fast_model)

    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in zip(ids_list, text_collections)]
    model = Doc2Vec(size=200, alpha=0.025, min_alpha=0.065, min_count=1, dm=1, window=10, workers=4)
    #Doc2Vec(dm=1, dm_mean=1, vector_size=300, window=10, negative=5, min_count=1, workers=5, alpha=0.065, min_alpha=0.065)
    model.build_vocab(tagged_data)

    max_epochs = 2
    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.iter) # decrease the learning rate
        model.alpha -= 0.0002 # fix the learning rate, no decay
        model.min_alpha = model.alpha
    #model.save("d2v.model")
    return model

def custom_countvectorizer_ranking(dataset, vector, query, query_vector,
                                vectorizer, gensim_active, sequence_ranking):
    result_collection = []
    for words_vector, page_id in vector:
        result_collection.append([return_match_length(words_vector, query_vector), page_id])

    rank_list = rank_list_collection(result_collection, dataset)
    collectors = collecting_few_results(rank_list)
    # print("collectors", collectors)

    if sequence_ranking is True:
        sequence_wise_rank = sequence_handler(collectors, vector, query_vector)
        sequence_wise_rank = collecting_few_results(sequence_wise_rank,2)
        print("sequence_wise_rank", sequence_wise_rank)

    if gensim_active is True:
        model = gensim_trainer(collectors, dataset)
        query_vector_gensim = model.infer_vector(query.split())
        ranking_result_gensim = model.docvecs.most_similar([query_vector_gensim], topn=3)
        return ranking_result_gensim
    else:
        if sequence_ranking is True:
            vec, text_collections, ids_list = filtered_vector(sequence_wise_rank, vectorizer, dataset)
        else:
            vec, text_collections, ids_list = filtered_vector(collectors, vectorizer, dataset)

        ranks = ranking_result_collection(query, text_collections, ids_list, vec)
        return ranks

def get_score_details_record(dataset, questions_samples, vectorizer, vector,
                             word_ids, sequence_ranking, gensim_active=False):
    sample_count = 0
    sum_score = 0
    container = []

    for index, col in questions_samples.iterrows():
        query = str(col['Question'])
        input_vector = list(set(str(query).split()))
        query_vector = custom_vectorizer(input_vector, word_ids)

        rank_list = custom_countvectorizer_ranking(dataset, vector, query,
                                               query_vector, vectorizer, gensim_active, sequence_ranking)
        #print(rank_list)

        page_answers = []
        prediction_scores = []
        for ids, score in rank_list:
            # print(ids, score)
            page_answers.append(ids)
            prediction_scores.append(score)

        MRR_score = mean_reciprocal_rank_score(col['PageID'], page_answers)
        sum_score += MRR_score

        container.append([MRR_score, col['PageID'], page_answers, prediction_scores, col['Question']])
        sample_count += 1

    result = pd.DataFrame(container, columns=['score', 'actual_answer', 'page_answers', 'prediction_scores',
                                              'query'])
    result.to_csv('performance.csv')
    score = sum_score / sample_count

    return score

###################  For Seq_CountVectorizer ##########################

def converting_vector(collectors, vectorizer, dataset):
    text_collections=[]
    ids_list=[]
    for ids in collectors:
        temp = dataset[dataset.PageID == str(ids)].Data.values
        temp_list = ' '.join(temp.tolist())
        text_collections += [temp_list]
        #text_collections += page_text_split(temp_list)
        ids_list.append(ids)
    vec = vectorizer.fit(text_collections)
    return vec, text_collections, ids_list

def seq_countvec_main(questions_samples, corpus_per_page, vectorizer, dataset):
    sample_count = 0
    sum_score = 0
    container = []
    for index, col in questions_samples.iterrows():
        query = str(col['Question'])
        question_parts = making_query_collection(query)
        collector = sequence_searcher(corpus_per_page, question_parts)
    #     vec, text_collections, ids_list = converting_vector(collector, vectorizer, dataset)
    #     rank_list = ranking_result_collection(query, text_collections, ids_list, vec)
    #
    #     page_answers = []
    #     prediction_scores = []
    #     for ids, score in rank_list:
    #         page_answers.append(ids)
    #         prediction_scores.append(score)
    #     MRR_score = mean_reciprocal_rank_score(col['PageID'], page_answers)
    #     sum_score += MRR_score
    #     container.append([MRR_score, col['PageID'], page_answers, prediction_scores, col['Question']])
    #     sample_count += 1
    #
    # result = pd.DataFrame(container, columns=['score', 'actual_answer',
    # 'page_answers', 'prediction_scores', 'query'])
    # result.to_csv('seq_CountTFIDFVectorizer_performance.csv')
    # score = sum_score/sample_count
    return score
