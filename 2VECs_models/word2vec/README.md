Install MeCab on Ubuntu is very easy:
python process_wiki.py jawiki-latest-pages-articles.xml.bz2 wiki.ja.text

$ sudo apt-get install mecab libmecab-dev mecab-ipadic
$ sudo apt-get install mecab-ipadic-utf8
$ sudo apt-get install python-mecab

mecab -O wakati wiki.ja.text -o wiki.ja.text.seg -b 10000000

python train_word2vec_model.py wiki.ja.text.seg wiki.ja.text.model wiki.ja.text.vector
