from difflib import get_close_matches
import re
import os
from collections import Counter
from enchant import DictWithPWL
from constants.fixed_names import PROCESSED_TEXTS_FILE, DEEP_CLEAN_DATA_FILE, VOCAB_FILE


def load_data(data_dir):
    data_file = os.path.join(data_dir, DEEP_CLEAN_DATA_FILE)
    vocab_file = os.path.join(data_dir, VOCAB_FILE)

    if (os.path.isfile(data_file) is False or os.path.isfile(vocab_file) is False):
        with open(os.path.join(data_dir, PROCESSED_TEXTS_FILE), encoding='utf-8') as f:
            data_list = f.read().splitlines()

        data = single_character_remover(" ".join(data_list))
        with open(data_file, 'w') as f:
            f.write(data)

        vocabulary = list(set(data.split()))
        with open(vocab_file, 'w') as out:
            out.writelines("%s\n" % vocab for vocab in vocabulary)
    else:
        with open(vocab_file) as f:
            vocabulary = f.read().split()

    return data_file, vocabulary


def load_words_letters(corpus):
    hira_kata = 'かめふアうチパズヅさモぴグゆごヒサもシマりはゲひべヘイヤづペユへぽのほけエこツぺぢだをデど' \
                'ヨギぜミキリるろヌばむょラゴにウずしてすぬつスコネせムロたちゾぎゃおピぶンねガヲぱらカダメュ' \
                'ョぷそナみノなんクホハニぞトでげワいャぐとザビやソプれぼきバベブジゅじゼまレセルびポがくわタドオケヂフボテえざあよ '
    eng_letters = 'abcdefghijklmnopqrstuvwxyz'
    letters = hira_kata + eng_letters
    word_dict = Counter(words(open(corpus).read()))
    return word_dict, letters


def edits1(word, letters):
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word, letters):
    return (e2 for e1 in edits1(word, letters) for e2 in edits1(e1, letters))


def words(text):
    return re.findall(r'\w+', text.lower())


def all_substrings(string):
    n = len(string)
    return {string[i:j + 1] for i in range(n) for j in range(i, n)}


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


def longest_match(best_matches, items):
    longest_content = []
    for content in best_matches:
        longest_content.append(max(all_substrings(content) & all_substrings(items), key=len))
    return max(longest_content, key=len)


def handling_spelling_mistakes(misspelled_word, vocabulary):
    max_term = []
    best_matches = get_close_matches(misspelled_word, vocabulary, n=5, cutoff=0.6)
    if best_matches:
        max_term = longest_match(best_matches, misspelled_word)
    return max_term


class SpellingCheckerCorrector:
    def __init__(self, data_dir):
        corpus_file, self.vocabulary = load_data(data_dir)
        self.word_count_dict, self.all_letters = load_words_letters(corpus_file)
        vocab_file = os.path.join(data_dir, VOCAB_FILE)
        self.dictionary = DictWithPWL('en_US', vocab_file)
        print("Data Loaded, Vocabulary and Dictionary created")

    def probability(self, word):
        return self.word_count_dict[word] / sum(self.word_count_dict.values())

    def correction(self, word, letters):
        """Most probable spelling correction for word."""
        return max(self.candidates(word, letters), key=self.probability)

    def candidates(self, word, letters):
        """Generate possible spelling corrections for word."""
        return self.known([word]) or self.known(edits1(word, letters)) or self.known(
            edits2(word, letters)) or [word]

    def known(self, word_list):
        return set(w for w in word_list if w in self.word_count_dict)

    def spelling_checker_suggester(self, sent):
        correct_sentence = {}
        for word in sent.split():
            word = word.lower()
            if word in self.vocabulary:
                correct_sentence[word] = [word]

            else:
                # First find words suggestion from library
                suggested_words = self.dictionary.suggest(word)
                if suggested_words:
                    correct_sentence[word] = suggested_words

                # find longest chunk match using difflib library
                else:
                    matches = handling_spelling_mistakes(word, self.vocabulary)
                    match_len = len(matches)/len(word)

                    # Matched chunk length is <25% distance away
                    if match_len > 0.75:
                        correct_sentence[word] = self.dictionary.suggest(matches)

                    # Matched chunk length is <35% distance away
                    elif match_len > 0.65:
                        suggested_words = self.correction(word, self.all_letters).split() + list(
                            self.known(edits2(word, self.all_letters)))
                        correct_sentence[word] = list(set(suggested_words))
                    else:
                        correct_sentence[word] = [word]

        return correct_sentence


def spelling_check(output_dir, data_dir, input_query):
    # 漢字以外の一文字は除外
    if len(input_query) <= 1 and re.match("[一-龥]", input_query) is None:
        return {input_query: []}

    checker_object = SpellingCheckerCorrector(data_dir)

    spelling = checker_object.spelling_checker_suggester(input_query)

    return spelling
