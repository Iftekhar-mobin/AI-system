import re
import os
from collections import Counter
from enchant import DictWithPWL
from logger import logger
import _pickle as pickle
from constants import PROCESSED_TEXTS_FILE, FREQUENCY_DICTIONARY, DEEP_CLEAN_DATA_FILE, VOCAB_FILE
from user_input_correction.keyword_corrector.spellcorrector import Corrector, Verbosity


def dump_to_file(file_path, data):
    with open(file_path, 'wb') as output_file:
        pickle.dump(data, output_file)


def dump_frequency_dict(word_dict, file_name, directory):
    with open(os.path.join(directory, file_name), 'w', encoding="utf8", errors='ignore') as f:
        for k, v in word_dict:
            f.write("{} {}\n".format(k, v))


def load_data(data_dir):
    data_file = os.path.join(data_dir, DEEP_CLEAN_DATA_FILE)
    frequency_file = os.path.join(data_dir, FREQUENCY_DICTIONARY)
    vocab_file = os.path.join(data_dir, VOCAB_FILE)

    if os.path.isfile(data_file) is False or os.path.isfile(vocab_file) is False \
            or os.path.isfile(frequency_file) is False:
        with open(os.path.join(data_dir, PROCESSED_TEXTS_FILE), encoding='utf-8') as f:
            data_list = f.read().splitlines()

        data = single_character_remover(" ".join(data_list))
        dump_to_file(data_file, data)

        # create frequency dictionary
        word_dict = Counter(re.findall(r'\w+', data.lower())).most_common()
        dump_frequency_dict(word_dict, FREQUENCY_DICTIONARY, data_dir)
        logger.debug('Frequency dictionary created for spelling checker.')

        # Create vocabulary
        vocabulary = list(set(data.split()))
        with open(vocab_file, 'w') as out:
            out.writelines("%s\n" % vocab for vocab in vocabulary)
        logger.debug('Vocabulary produced for spelling checker.')

    else:
        with open(vocab_file) as f:
            vocabulary = f.read().split()

    return frequency_file, vocabulary


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


class SpellingCheckerCorrector:
    def __init__(self, data_dir):
        frequency_dict, self.vocabulary = load_data(data_dir)
        self.dictionary = DictWithPWL('en_US', os.path.join(data_dir, VOCAB_FILE))
        self.corrector = Corrector(max_dictionary_edit_distance=2, prefix_length=7)
        try:
            self.corrector.load_dictionary(frequency_dict, term_index=0, count_index=1, separator=" ", encoding='utf-8')
            # For debug
            # print(list(islice(self.corrector.words.items(), 5)))
        except UnicodeDecodeError:
            logger.info('UnicodeDecodeError from the spelling checker')
        finally:
            logger.debug("Data Loaded, Vocabulary and Dictionary created")

    def spelling_checker_suggester(self, sent):
        if len(sent) <= 1 and re.match("[一-龥]", sent) is None:
            return {sent: []}

        correct_sentence = {}
        for word in sent.split():
            word = word.lower()
            if word in self.vocabulary:
                correct_sentence[word] = [word]

            elif len(word) > 1:
                try:
                    suggestions = self.corrector.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
                    # for Debug
                    # print(suggestions)
                    if suggestions:
                        collector = []
                        [collector.append(str(suggestion).split(',', 1)[0]) for suggestion in suggestions]
                        correct_sentence[word] = collector
                    else:
                        correct_sentence[word] = sorted(self.dictionary.suggest(word), key=len, reverse=True)
                except:
                    logger.exception('library Error from the spelling checker')
                    pass
        return correct_sentence
