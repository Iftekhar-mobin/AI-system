# %%
import spacy
from spacy_langdetect import LanguageDetector
nlp = spacy.load("en")
nlp.add_pipe(LanguageDetector(), name="language_detector", last=True)
text = "This is English text. Er lebt mit seinen Eltern und seiner Schwester in Berlin. Yo me divierto todos los días en el parque. Je m'appelle Angélica Summer, j'ai 12 ans et je suis canadienne."

# %%
import MeCab
mecab = MeCab.Tagger('-Owakati')

text = 'AndroidEnteriseの登録方法は？'
text = mecab.parse(text).strip("\n").rstrip()

# %%
doc = nlp(text)
# document level language detection. Think of it like average language of document!
print(doc._.language)
print("-----------------------------------------------------------------")
# sentence level language detection
for i, sent in enumerate(doc.sents):
    print(sent, sent._.language)

print("-----------------------------------------------------------------")
# Token level language detection from version 0.1.2
# Use this with caution because, in some cases language detection will not make sense for individual tokens
for token in doc:
    print(token, token._.language)

# %%
text

# %%
'''
# Trying With fastText library
'''

# %%
import fasttext
model = fasttext.load_model('/home/ifte/Downloads/lid.176.bin')
print(model.predict(text, k=1))  # top 2 matching languages

#(('__label__ar', '__label__fa'), array([0.98124713, 0.01265871]))

# %%
k = model.predict(text, k=1)

# %%
ln1 = str(k[0][0]).replace('__label__','')
ln2 = str(k[0][1]).replace('__label__','')
ln2

# %%
from pycountry import languages

language_name = languages.get(alpha_2=ln1).name
print(language_name)
# french

# %%
