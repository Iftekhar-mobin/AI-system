# %%
lang_name = {
"Abkhazian":"ab",
"Afar":"aa",
"Afrikaans":"af",
"Akan":"ak",
"Albanian":"sq",
"Amharic":"am",
"Arabic":"ar",
"Aragonese":"an",
"Armenian":"hy",
"Assamese":"as",
"Avaric":"av",
"Avestan":"ae",
"Aymara":"ay",
"Azerbaijani":"az",
"Bambara":"bm",
"Bashkir":"ba",
"Basque":"eu",
"Belarusian":"be",
"Bengali":"bn",
"Bihari languages":"bh",
"Bislama":"bi",
"Bosnian":"bs",
"Breton":"br",
"Bulgarian":"bg",
"Burmese":"my",
"Catalan":"ca",
"Chamorro":"ch",
"Chechen":"ce",
"Chichewa":"ny",
"Chinese":"zh",
"Chuvash":"cv",
"Cornish":"kw",
"Corsican":"co",
"Cree":"cr",
"Croatian":"hr",
"Czech":"cs",
"Danish":"da",
"Divehi":"dv",
"Dutch":"nl",
"Dzongkha":"dz",
"English":"en",
"Esperanto":"eo",
"Estonian":"et",
"Ewe":"ee",
"Faroese":"fo",
"Fijian":"fj",
"Finnish":"fi",
"French":"fr",
"Fulah":"ff",
"Galician":"gl",
"Georgian":"ka",
"German":"de",
"Greek":"el",
"Guarani":"gn",
"Gujarati":"gu",
"Hausa":"ha",
"Hebrew":"he",
"Herero":"hz",
"Hindi":"hi",
"Hiri Motu":"ho",
"Hungarian":"hu",
"Indonesian":"id",
"Irish":"ga",
"Igbo":"ig",
"Inupiaq":"ik",
"Ido":"io",
"Icelandic":"is",
"Italian":"it",
"Inuktitut":"iu",
"Japanese":"ja",
"Javanese":"jv",
"Kalaallisut":"kl",
"Kannada":"kn",
"Kanuri":"kr",
"Kashmiri":"ks",
"Kazakh":"kk",
"Central Khmer":"km",
"Kikuyu":"ki",
"Kinyarwanda":"rw",
"Kirghiz":"ky",
"Komi":"kv",
"Kongo":"kg",
"Korean":"ko",
"Kurdish":"ku",
"Kuanyama":"kj",
"Latin":"la",
"Ganda":"lg",
"Limburgan":"li",
"Lingala":"ln",
"Lao":"lo",
"Lithuanian":"lt",
"Luba-Katanga":"lu",
"Latvian":"lv",
"Manx":"gv",
"Macedonian":"mk",
"Malagasy":"mg",
"Malay":"ms",
"Malayalam":"ml",
"Maltese":"mt",
"Maori":"mi",
"Marathi":"mr",
"Marshallese":"mh",
"Mongolian":"mn",
"Nauru":"na",
"Navajo":"nv",
"North Ndebele":"nd",
"Nepali":"ne",
"Ndonga":"ng",
"Norwegian Bokmål":"nb",
"Norwegian Nynorsk":"nn",
"Norwegian":"no",
"Sichuan Yi":"ii",
"South Ndebele":"nr",
"Occitan":"oc",
"Ojibwa":"oj",
"Oromo":"om",
"Oriya":"or",
"Ossetian":"os",
"Punjabi":"pa",
"Pali":"pi",
"Persian":"fa",
"Polish":"pl",
"Pashto":"ps",
"Portuguese":"pt",
"Quechua":"qu",
"Romansh":"rm",
"Rundi":"rn",
"Romanian":"ro",
"Russian":"ru",
"Sanskrit":"sa",
"Sardinian":"sc",
"Sindhi":"sd",
"Northern Sami":"se",
"Samoan":"sm",
"Sango":"sg",
"Serbian":"sr",
"Gaelic":"gd",
"Shona":"sn",
"Sinhala":"si",
"Slovak":"sk",
"Slovenian":"sl",
"Somali":"so",
"Southern Sotho":"st",
"Spanish":"es",
"Sundanese":"su",
"Swahili":"sw",
"Swati":"ss",
"Swedish":"sv",
"Tamil":"ta",
"Telugu":"te",
"Tajik":"tg",
"Thai":"th",
"Tigrinya":"ti",
"Tibetan":"bo",
"Turkmen":"tk",
"Tagalog":"tl",
"Tswana":"tn",
"Tonga":"to",
"Turkish":"tr",
"Tsonga":"ts",
"Tatar":"tt",
"Twi":"tw",
"Tahitian":"ty",
"Uighur":"ug",
"Ukrainian":"uk",
"Urdu":"ur",
"Uzbek":"uz",
"Venda":"ve",
"Vietnamese":"vi",
"Volapük":"vo",
"Walloon":"wa",
"Welsh":"cy",
"Wolof":"wo",
"Western Frisian":"fy",
"Xhosa":"xh",
"Yiddish":"yi",
"Yoruba":"yo",
"Zhuang":"za",
"Zulu":"zu"
}

# %%
import spacy
from spacy_cld import LanguageDetector
nlp = spacy.load('en')
language_detector = LanguageDetector()
nlp.add_pipe(language_detector)

# %%
import MeCab
mecab = MeCab.Tagger('-Owakati')

text = 'AndroidEnteriseの登録方法は？'
text = mecab.parse(text).strip("\n").rstrip()



# %%
doc = nlp(text)

print(doc._.languages)  # ['en']
print(doc._.language_scores)  # 0.96

# %%
for i, sent in enumerate(doc.sents):
    print(i,sent._.language)

# %%
for i, sent in enumerate(doc.sents):
    print('Text: <start>',sent,"<End> : ")
    l = sent._.language['language']
    
    for name, code in lang_name.items():
        if code == l:
            n = name
    
    print(n,":",sent._.language_scores)

# %%
import fasttext
import ast

def language_detect(msg):
    #print("Message is: ",msg)
    model = fasttext.load_model('/home/ifte/Downloads/lid.176.bin')
    ln_array = model.predict(msg, k=1)  # top 2 matching languages
    ln = str(ln_array[0][0]).replace('__label__', '')
    #print("Language in func: ", ln)

    LANGUAGES_NANE_FILE = "/home/ifte/amiebot_project/amie-HelpBot/amie_helpbot/" + '/assets/learning/' + "languages_name.txt"

    dict_file = open(LANGUAGES_NANE_FILE, "r")
    dict_string = dict_file.readline().strip()
    dict_file.close()

    lang_name = ast.literal_eval(dict_string)

    language_name = None
    for name, code in lang_name.items():
        if code == ln:
            language_name = name

    return language_name

language_detect(text.strip("\n").rstrip())

# %%
LANGUAGES_NAME_FILE = "/home/ifte/amiebot_project/amie-HelpBot/amie_helpbot/" + '/assets/learning/' + "languages_name.txt"

# %%
dict_file = open(LANGUAGES_NANE_FILE, "r")
dict_string = dict_file.readline().strip()
dict_file.close()

# %%
dict_string

# %%
with open(LANGUAGES_NANE_FILE, "r") as data:
    dictionary = ast.literal_eval(data.read())

# %%
dictionary

# %%


# %%
