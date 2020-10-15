import pandas as pd
import MeCab
import re
import ast

mecab = MeCab.Tagger('-Owakati')
from pathlib import Path
import difflib
# import spacy
# nlp = spacy.load('ja_ginza')
from collections import OrderedDict


def synonym_matching_from_tags(tags, max_matched, synonyms_dict, input_query):
    tag_synonyms = generate_synonyms_for_tag_chunks(tags, max_matched, synonyms_dict)
    query_synonyms = generate_synonyms_for_tag_chunks(input_query.split(), max_matched, synonyms_dict)
    # print(query_synonyms, "<===>", tag_synonyms)
    return find_common_terms_between_synonym_list(tag_synonyms, query_synonyms)


def find_common_terms_between_synonym_list(tag_synonyms, query_synonyms):
    synonym_saver = []
    for chunks, synonyms in tag_synonyms:
        for chunks_query, synonyms_query in query_synonyms:
            common_terms = list(set(synonyms).intersection(synonyms_query))
            if common_terms:
                synonym_saver.append([chunks, common_terms])

    collector = []
    for chunks, matched in synonym_saver:
        collector.append(chunks)

    return collector


def generate_synonyms_for_tag_chunks(tags, max_matched, synonyms_dict):
    saver = []
    for chunks in tags:
        collector = []
        chunks_full = chunks
        chunks = chunks.replace(max_matched, '')
        tag_chunks = chunks.split()
        for terms in tag_chunks:
            if terms is not '':
                synonyms = synonyms_dict.get(terms)
                if synonyms:
                    collector.append(synonyms)
        if collector:
            saver.append([chunks_full, [x for sublist in collector for x in sublist]])
    return saver


def check_not_matched_present_in_tags(tags, not_matched):
    collector = []
    for chunks in tags:
        for items in not_matched.split():
            if chunks.find(items) != -1:
                collector.append(chunks)
                # print("Not matched Present in: ", chunks)
    return collector


def unique_tag_provider(matched, token_query_word):
    tags = []
    # print(matched)
    for items in matched:
        for match in re.finditer(r'# (.*) #', items):
            tags.append(items[match.start() + 1: match.end()].split('#'))
    all_tag = []
    for tag_chunk in tags:
        for tag in tag_chunk:
            if tag is not '':
                all_tag.append(tag.strip())
    unique_tags = list(OrderedDict.fromkeys(sorted(all_tag, key=all_tag.count, reverse=True)))
    unique_tags = list(set(unique_tags))
    # print(unique_tags)

    try:
        if len(unique_tags) > 1:
            unique_tags.remove(token_query_word)
    except ValueError:
        pass
    return unique_tags


def query_in_middle_position(text, match):
    chunk = text[match.start() - 20: match.end() + 20]
    chunk_list = chunk.split()
    chunk_list.pop(0)
    chunk_list.pop(-1)
    return chunk_list


def unique_recommended_all_tags(pages_tags):
    suggest_tags = []
    for tags in [x for sublist in pages_tags for x in sublist]:
        if tags:
            suggest_tags.append(tags)
    suggest_tags = list(OrderedDict.fromkeys(sorted(suggest_tags, key=suggest_tags.count, reverse=True)))
    recommended_tags = []
    for items in suggest_tags:
        recommended_tags.append(single_character_remover(items))
    return recommended_tags


def query_at_top_at_beginning(text, match):
    chunk = text[match.start(): match.end() + 40]
    chunk_list = chunk.split()
    chunk_list.pop(-1)
    return chunk_list


def longest_match_within_best_matches(best_matches, items):
    longest_content = []
    for content in best_matches:
        longest_content.append(max(all_substrings(content) & all_substrings(items), key=len))
    return max(longest_content, key=len)


def tag_chunks(front_seq_word, rear_seq_word):
    rear_queue = []
    count = 0
    for word in rear_seq_word:
        # if wnt to restrict for English and Japanese Katakana only
        # if re.match(r'[ァ-ン]', word) or re.match(r'[A-Za-z]', word) and count < 3:
        if count < 3:
            rear_queue.append(word)
        else:
            break
        count += 1

    front_queue = []
    count = 0
    for word in front_seq_word[::-1]:
        if count < 3:
            front_queue.append(word)
        else:
            break
        count += 1
    front_queue.reverse()
    return front_queue, rear_queue


def tags_factory(text, match, pattern):
    front_seq_word = text[match.start() - 30: match.end()].split()
    rear_seq_word = text[match.start(): match.end() + 30].split()
    # print(front_seq_word, rear_seq_word)
    front_queue, rear_queue = tag_chunks(front_seq_word, rear_seq_word)
    return front_queue, rear_queue


def hash_tag_generator_for_non_sequence(page_corpus, token_query_word):
    unique_tags = []
    pages_tags = []
    collector = []
    token_query_word = token_query_word
    pattern = token_query_word
    for index, col in page_corpus.iterrows():
        matched = []
        text = col['Data']
        for match in re.finditer(pattern, text):
            if match:
                # print(match)
                if match.start() > 30:
                    chunk_list = query_in_middle_position(text, match)
                    front_queue, rear_queue = tags_factory(text, match, pattern)
                    matched.append(' '.join(chunk_list + ["#"] + rear_queue + ["#"] + front_queue + ["#"]))
                else:
                    matched.append(' '.join(query_at_top_at_beginning(text, match)))

        if matched:
            unique_tags = unique_tag_provider(matched, token_query_word)
            collector.append([col['PageID'], len(matched), unique_tags, matched])
        pages_tags.append(unique_tags)
    tags = unique_recommended_all_tags(pages_tags)

    return tags, sorted(collector, key=lambda l: l[1], reverse=True)[:10]


def hash_tag_generator(page_corpus, token_query_word):
    unique_tags = []
    pages_tags = []
    collector = []
    token_query_word = token_query_word
    pattern = token_query_word
    for index, col in page_corpus.iterrows():
        matched = []
        text = col['Data']
        for match in re.finditer(pattern, text):
            if match:
                print(match)
                if match.start() > 30:
                    chunk_list = query_in_middle_position(text, match)
                    front_queue, rear_queue = tags_factory(text, match, pattern)
                    matched.append(' '.join(chunk_list + ["#"] + rear_queue + ["#"] + front_queue + ["#"]))
                else:
                    matched.append(' '.join(query_at_top_at_beginning(text, match)))
        #print('matched: ', matched)
        # exit()
        if matched:
            unique_tags = unique_tag_provider(matched, token_query_word)
            print('unique_tags: ', unique_tags)
            # exit()
            collector.append([col['PageID'], len(matched), unique_tags, matched])
        pages_tags.append(unique_tags)
    tags = unique_recommended_all_tags(pages_tags)
    print('Tags: ', tags)
    # exit()
    return tags, sorted(collector, key=lambda l: l[1], reverse=True)[:10]


def making_query_collection(query):
    query_parts = query.split()
    question_parts = []
    for i in range(len(query_parts)):
        if len(query_parts) - 1 > i:
            question_parts.append(query_parts[i] + " " + query_parts[i + 1])
            if len(query_parts) - 2 > i:
                question_parts.append(query_parts[i] + " " + query_parts[i + 1] + " " + query_parts[i + 2])
    return question_parts


def query_rewritter_replacing_synonyms(single_token_query, corpus, synonyms_dict):
    # check the synonyms and convert it to base terms
    collector = []
    for items in single_token_query:
        if corpus.find(items) == -1:
            dict_synonyms = get_keys_by_value(synonyms_dict, items)
            if dict_synonyms:
                print("Input Terms: ", items, ' uttered in corpus ', dict_synonyms)
                collector.append(' '.join(dict_synonyms))
        else:
            collector.append(items)
    # rewritten_query = ' '.join([x for sublist in collector for x in sublist])
    # print("Your input becomes: ", ' '.join(collector))
    return collector


def handling_spelling_mistakes(question_parts, vocabulary):
    # Assumed user has spelling mistakes
    collector = []
    for items in question_parts:
        best_matches = difflib.get_close_matches(items, vocabulary, n=5, cutoff=0.6)
        if best_matches:
            max_term = longest_match_within_best_matches(best_matches, items)
            collector.append(max_term)
            # print("Closest", best_matches)
    return collector


def how_long_query_matched(collector, whole_corpus):
    max_matched = ''
    flag = True
    for items in collector:
        if whole_corpus.find(items) != -1 and flag is True:
            max_matched = items
            flag = False
        elif whole_corpus.find(max_matched + " " + items) != -1:
            max_matched += " " + items
        else:
            break

    not_matched = ' '.join(collector).replace(max_matched, '')
    print("Maximum Sequence Matched: ", max_matched, 'not_matched: ', not_matched)
    return max_matched, not_matched


def get_maximum_longest_matched_chunk(input_query, synonyms_dict, corpus):
    whole_corpus = ' '.join(corpus.Data.values)
    single_token_query = input_query.split()
    collector = query_rewritter_replacing_synonyms(single_token_query, whole_corpus, synonyms_dict)
    max_matched, not_matched = how_long_query_matched(collector, whole_corpus)
    return max_matched


def unknown_word_sequence_handler(input_query, vocabulary, synonyms_dict, corpus):
    print('unknown_word_sequence_handler---------------------------')
    not_matched = 0
    whole_corpus = ' '.join(corpus.Data.values)
    single_token_query = input_query.split()
    question_parts = [input_query] + making_query_collection(input_query) + single_token_query
    print("question_parts-----------------")
    print(question_parts)
    collector = query_rewritter_replacing_synonyms(single_token_query, whole_corpus, synonyms_dict)
    print("collector------------------")
    print(collector)
    # exit()
    max_matched, not_matched = how_long_query_matched(collector, whole_corpus)
    #print('Max_matched, Not_matched: ', max_matched, not_matched)
    print('Max_matched: ', max_matched)
    print('Not_matched: ', not_matched)
    voc_hints = handling_spelling_mistakes(question_parts, vocabulary)
    print("Vocab found from corpus: ", list(set(voc_hints)))
    # exit()

    if max_matched not in '' or max_matched not in ' ':
        tags, details = hash_tag_generator_for_non_sequence(corpus, max_matched)
        print('tags: ', tags)
        print('details: ', details)
        # exit()

        if tags:
            print("Suggestions: ", tags)
            return tags, not_matched, max_matched
    else:
        return ['Not Matched'], not_matched, max_matched


def get_keys_by_value(synonyms_dict, items):
    list_keys = list()
    list_items = synonyms_dict.items()
    for item in list_items:
        for synonyms in item[1]:
            if synonyms == items:
                list_keys.append(item[0])
    return list_keys


def load_dictionary():
    file = open("methods/vocabulary_synonyms_all.json", "r")
    contents = file.read()
    synonyms_dict = ast.literal_eval(contents)
    file.close()
    return synonyms_dict


def vocabulary_load():
    with open('methods/vocabulary.txt') as f:
        vocabulary = f.read().splitlines()
    return vocabulary


def all_substrings(string):
    n = len(string)
    return {string[i:j + 1] for i in range(n) for j in range(i, n)}


def query_corpus_processing(corpus):
    cus_ques = pd.read_csv(corpus)
    cus_ques.Question = cus_ques.Question.apply(lambda x: mecab_tokenization(x))
    cus_ques.Question = cus_ques.Question.apply(lambda x: single_character_remover(x))
    cus_ques.Question = cus_ques.Question.apply(lambda x: cleaner(x))
    return cus_ques


def mecab_tokenization(text):
    q = mecab.parse(text)
    q_parts = q.split()
    return ' '.join([word for word in q_parts if not word in get_stop_word_ja()])


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


def cleaner(text):
    collector = []
    for items in text.split():
        cleaned = clean_text(items)
        cleaned = re.sub(r"\s+", '', cleaned)
        if cleaned is not '' or cleaned is not ' ':
            collector.append(clean_text(items))

    return ' '.join(collector)


def clean_text(text):
    replaced = text.replace("\\", "")
    replaced = replaced.replace("+", "")
    replaced = re.sub('_', '', replaced)
    replaced = re.sub('\W+', ' ', replaced)
    replaced = re.sub(r'￥', '', replaced)  # 【】の除去
    replaced = re.sub(r'．', '', replaced)  # ・ の除去
    replaced = re.sub(r'｣', '', replaced)  # （）の除去
    replaced = re.sub(r'｢', '', replaced)  # ［］の除去
    replaced = re.sub(r'～', '', replaced)  # メンションの除去
    replaced = re.sub(r'｜', '', replaced)  # URLの除去
    replaced = re.sub(r'＠', '', replaced)  # 全角空白の除去
    replaced = re.sub(r'？', '', replaced)  # 数字の除去
    replaced = re.sub(r'％', '', replaced)
    replaced = re.sub(r'＝', '', replaced)
    replaced = re.sub(r'！', '', replaced)
    replaced = re.sub(r'｝', '', replaced)
    replaced = re.sub(r'：', '', replaced)
    replaced = re.sub(r'－', '', replaced)
    replaced = re.sub(r'･', '', replaced)
    replaced = re.sub(r'ｔ', '', replaced)
    replaced = re.sub(r'ｋ', '', replaced)
    replaced = re.sub(r'ｄ', '', replaced)
    replaced = re.sub(r'\d+', '', replaced)

    return replaced


def longest_seq_search(query, page_data):
    m = len(query)
    n = len(page_data)
    counter = [[0] * (n + 1) for x in range(m + 1)]
    longest = 0
    lcs_set = set()
    for i in range(m):
        for j in range(n):
            if query[i] == page_data[j]:
                c = counter[i][j] + 1
                counter[i + 1][j + 1] = c
                if c > longest:
                    lcs_set = set()
                    longest = c
                    lcs_set.add(query[i - c + 1:i + 1])
                elif c == longest:
                    lcs_set.add(query[i - c + 1:i + 1])

    return lcs_set


def get_stop_word_ja():
    stop_word_file = Path("stop_word_ja.txt")
    with open(stop_word_file, encoding='utf-8') as f:
        stop_word_list = f.read().splitlines()
    return stop_word_list


def corpus_split(corpus, sentence_length):
    labels = corpus.PageID.unique()
    lines = []
    all_ids = []
    for i in list(labels):
        text_list = corpus[corpus.PageID == i].Data.values
        split_text = fixed_length_sentence(' '.join(text_list), sentence_length)
        ids = [i] * len(split_text)
        lines += split_text
        all_ids += ids
    split_corpus = pd.DataFrame(zip(lines, all_ids), columns=["Data", "PageID"])
    return split_corpus


def fixed_length_sentence(contents, word_limit):
    contents_list = contents.split()
    end = len(contents_list)
    count = 0
    collector = []
    line = []
    for items in contents_list:
        if count < word_limit - 1 and end > 1:
            collector.append(items)
            count += 1
        else:
            collector.append(items)
            line.append(' '.join(collector))
            collector = []
            count = 0
        end -= 1
    return line


def split_joint_word(text):
    pattern = re.compile("[A-Z]")
    index_saver = []
    start = -1
    while True:
        m = pattern.search(text, start + 1)
        if m == None:
            break
        start = m.start()
        index_saver.append(start)

    sorted_list = sorted(index_saver)
    range_list = list(range(min(index_saver), max(index_saver) + 1))
    if sorted_list != range_list:
        temp = 0
        flag = False
        save = []
        for indexes in index_saver:
            if flag:
                if indexes - temp > 1:
                    save.append(indexes)
                    temp = indexes
            else:
                save.append(indexes)
                temp = indexes
                flag = True

        if len(save) > 1:
            chunk = text[save[0]:save[1]]
            return chunk, single_character_remover(text.replace(chunk, ''))
        else:
            return None, None
    else:
        return None, None


def english_joint_word_handler(text):
    saver = []
    while text:
        temp = text
        chunk, text = split_joint_word(text)
        saver.append(chunk)
    saver.append(temp)
    saver.remove(None)
    if len(saver) < 2:
        saver = []
    return saver
