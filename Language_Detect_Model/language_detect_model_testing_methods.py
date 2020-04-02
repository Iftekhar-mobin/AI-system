from os.path import isfile, join
import re
import langcodes
import pycountry
import os
import string


def resolving_language_name_mismatch(language_name):
    if language_name.find('Chinese') != -1:
        language_name = 'Chinese'
    elif language_name.find('Persian') != -1:
        language_name = 'Persian'
    elif language_name.find('Malay') != -1:
        language_name = 'Malay'
    elif language_name.find('Nepali') != -1:
        language_name = 'Nepali'
    elif language_name.find('Norwegian') != -1:
        language_name = 'Norwegian'
    elif language_name.find('Latvian') != -1:
        language_name = 'Latvian'
    return language_name

def find_lang_code_2digit(lang_3digit_code):
    two_digit_code = 0
    language_name = 0
    if pycountry.languages.get(alpha_3=lang_3digit_code) is not None:
        language_name = pycountry.languages.get(alpha_3=lang_3digit_code).name
        language_name = resolving_language_name_mismatch(language_name)
        try:
            two_digit_code = langcodes.find(language_name)
        except LookupError:
            pass
    elif pycountry.languages.get(alpha_2=lang_3digit_code) is not None:
        language_name = pycountry.languages.get(alpha_2=lang_3digit_code).name
        language_name = resolving_language_name_mismatch(language_name)
        two_digit_code = lang_3digit_code

    if len(str(two_digit_code)) != 2:
        language_dict_2digit_code = dict_2digit_code()
        for langnames in language_dict_2digit_code.keys():
            if langnames == language_name:
                two_digit_code = language_dict_2digit_code[langnames]

    return two_digit_code, language_name

def converter_scorer(language_code_collector, language_code_2digit_from_file, query_collector):
    sample_size = len(query_collector)
    code3_code2_convert = []
    score_flag_collector = []
    success_count = 0
    for lang_3digit_code in language_code_collector:
        code_2digit, lang_name = find_lang_code_2digit(lang_3digit_code)
        code3_code2_convert.append([str(code_2digit), language_code_2digit_from_file, lang_name])
        if language_code_2digit_from_file == str(code_2digit):
            score_flag = "1"
            score_flag_collector.append(score_flag)
            success_count += 1
        else:
            score_flag = "0"
            score_flag_collector.append(score_flag)

    score_percentage = (success_count/sample_size)*100
    cumulative_result = list([success_count, score_percentage, language_code_2digit_from_file,
                              find_language_name(language_code_2digit_from_file)])
    details_record = list(zip(score_flag_collector, query_collector, code3_code2_convert))

    return cumulative_result, details_record

def cleaning(replaced_text):
    replaced_text = replaced_text.lower()
    replaced_text = replaced_text.translate(str.maketrans('', '', string.punctuation))
    replaced_text = re.sub(r',', '', replaced_text)
    replaced_text = re.sub(r'】', '', replaced_text)
    replaced_text = re.sub(r'【', '', replaced_text)
    replaced_text = re.sub(r'\d+', '', replaced_text)
    replaced_text = re.sub(r'-', '', replaced_text)
    replaced_text = re.sub(r':', '', replaced_text)

    return replaced_text


def find_language_name(language_code_2digit):
    two_digit_code = 0
    language_name = 0
    if pycountry.languages.get(alpha_2=language_code_2digit) is not None:
        language_name = pycountry.languages.get(alpha_2=language_code_2digit).name
    else:
        language_dict_2digit_code = dict_2digit_code()
        for langnames, codes in language_dict_2digit_code.values():
            if language_code_2digit == codes:
                language_name = language_dict_2digit_code[codes]

    return language_name

def read_file(file_name_only, mypath):
    file_name_with_location = os.path.join(mypath, file_name_only)
    with open(file_name_with_location) as f:
        test_file_reader = f.read().replace("\n", "")
        return test_file_reader.split()

def model_query(query, model_min1_max5):
    k = model_min1_max5.predict(query, k=1)
    language_code = str(k[0][0]).replace('__label__','')
    return language_code

def dict_2digit_code():
    with open("/home/iftekhar/AI-system/Language_Detect_Model/language codes.txt") as f:
        lang_2digit_code = f.read().replace("\n","")
    language_dict_2digit_code = eval(lang_2digit_code)
    return language_dict_2digit_code

def split_lines_corpus(per_page_corpus):
    chunk_size = 5
    max_size = 50
    word_limit = 5

    line_collector = []
    collector = []
    count = 0

    end_mark = len(per_page_corpus)
    for items in per_page_corpus:
        if len(str(items)) < max_size:
            if count < word_limit and end_mark > 1:
                collector.append(items)
                count += 1
            else:
                collector.append(items)
                line_collector.append(' '.join(collector))
                collector = []
                count = 0
        else:
            if len(collector) > 1:
                line_collector.append(' '.join(collector))
                collector = []
                count = 0
            chunks = [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]
            end_mark_internal = len(chunks)
            count_internal = 0
            for items in chunks:
                if count_internal < word_limit and end_mark_internal > 1:
                    collector.append(items)
                    count_internal += 1
                else:
                    collector.append(items)
                    line_collector.append(' '.join(collector))
                    collector = []
                    count_internal = 0
                end_mark_internal -= 1
        end_mark -= 1
    return line_collector
