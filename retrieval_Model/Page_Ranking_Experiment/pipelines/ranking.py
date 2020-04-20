from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import MeCab
mt = MeCab.Tagger('')
mt.parse('')

def mean_reciprocal_rank_score(actual_value, predicted_values):
    pos = 0
    val = 0
    for i in predicted_values:
        if i == actual_value and pos == 0:
            val = 1
            break
        elif i == actual_value and pos == 1:
            val = 0.5
            break
        elif i == actual_value and pos == 2:
            val = 0.33
            break
        else:
            val = 0
        pos += 1

    return val

def crude_ranks(sorted_list, query, vector):
    X = vector.transform([str(query)])
    rank = []
    ids_list = []
    for ids, items in sorted_list:
        for sentences in items:
            Y = vector.transform([sentences])
            ids_list.append(ids)
            rank.append(cosine_similarity(X, Y))
    flat = [x for sublist in rank for x in sublist]
    ranks = sorted(zip(ids_list, flat), key=lambda l:l[1], reverse=True)[:3]
    return ranks

def delete_duplicate_ids(ranks):
    temp = 0
    ids_list = []
    filtered = []
    for page_id, score in ranks:
        if page_id == temp:
            continue
        else:
            ids_list.append(page_id)
            filtered.append([page_id, score[0]])
            temp = page_id
    return filtered, ids_list

def filtering_ranks(ranks, sorted_list, query, vector):
    extra_ranks = 0
    filtered_extra = None
    filtered, ids_list = delete_duplicate_ids(ranks)
    saved_list = sorted_list
    if len(ids_list) < 3:
        for ids in ids_list:
            index = 0
            for matched_id, items in sorted_list:
                if ids == matched_id:
                    saved_list.pop(index)
                index += 1
        extra_ranks = crude_ranks(saved_list, query, vector)
        filtered_extra, ids_list_extra = delete_duplicate_ids(extra_ranks)
        sum_filtered = filtered + filtered_extra
        
        return sorted(sum_filtered, key=lambda l:l[1], reverse=True)[:3]
    else:
        return filtered

def get_vector(text, gensim_model):
    sum_vec = np.zeros(200)
    word_count = 0
    node = mt.parseToNode(text)
    while node:
        fields = node.feature.split(",")
        if fields[0] == '名詞' or fields[0] == '動詞' or fields[0] == '形容詞':
            try: 
                temp = gensim_model.wv[node.surface]
            except KeyError:
                temp = 0
            sum_vec += temp
            word_count += 1
        node = node.next
    return sum_vec / word_count

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def word2vec_ranks(perpage_sequence_match, query, gensim_model):
    X = get_vector(query, gensim_model)
    rank = []
    ids_list = []
    for ids, items in perpage_sequence_match:
        for sentences in items:
            Y = get_vector(sentences, gensim_model)
            ids_list.append(ids)
            rank.append(cos_sim(X, Y))
    return sorted(zip(ids_list, rank), key=lambda l:l[1], reverse=True)[:3]