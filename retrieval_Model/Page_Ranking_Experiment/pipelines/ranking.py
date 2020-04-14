from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

def filtering_ranks(ranks, sorted_list, query, vector):
    temp = 0
    filtered = []
    ids_list = []
    filtered_ranks = []
    for page_id, score in ranks:
        temp = np.array2string(score)
        temp = temp.replace("[","").replace("]","")
        filtered_ranks.append([page_id, float(temp)])

    for page_id, score in filtered_ranks:
        if page_id == temp:
            continue
        else:
            ids_list.append(page_id)
            filtered.append([page_id, score])
            temp = page_id

    ############### Need to implement
    # saved_list = sorted_list
    # if len(ids_list) < 3:
    #     for ranked_ids in ids_list:
    #         index = 0
    #         for ids, items in sorted_list:
    #             if ranked_ids == ids:
    #                 saved_list.pop(index)
    #             index += 1
    # extra_ranks = crude_ranks(saved_list, query, vector)
    # collect_item = 3 - len(ids_list)
    # for i in range(collect_item):
    #    extra_ranks

    #     return collect_item
    # else:
    return filtered
