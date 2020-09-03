import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
lemmer = WordNetLemmatizer()
st_words = stopwords.words('english')


def intent_chunks(f_name):
    with open(f_name, encoding='utf-8') as f:
        intents = f.read()
    return intents


def remove_punctuation(words):
    table = str.maketrans('', '', string.punctuation)
    return [w.translate(table) for w in words]


def text_processing(items):
    tokens = [w for w in [lemmer.lemmatize(t) for t in word_tokenize(items)] if w not in st_words]
    return remove_punctuation([word.lower() for word in tokens])


def page_text_split(page_text, word_limit):
    page_text = page_text.split()
    return [' '.join(page_text[i:i + word_limit]) for i in range(0, len(page_text), word_limit)]
