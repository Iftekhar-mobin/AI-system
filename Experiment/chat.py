from predictor.predict import Predictor
from flask import Flask, request, render_template
from faq.faq_chatter import FaqBot
from preprocessor import text_preprocessing
# from chatterbot import ChatBot
# from chatterbot.trainers import ChatterBotCorpusTrainer
# from chatterbot.trainers import ListTrainer
import json

app = Flask(__name__)

# english_bot = ChatBot("Chatterbot", storage_adapter="chatterbot.storage.SQLStorageAdapter")
# trainer = ChatterBotCorpusTrainer(english_bot)
# trainer.train("chatterbot.corpus.english")


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/chatting', methods=["GET", "POST"])
def chat():
    request_param = request.form
    query = request_param['query']
    caller = FaqBot().generate_reply(query)
    return caller


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    request_param = request.form
    query = request_param['query']
    obj = Predictor('/home/iftekhar/AI-system/Experiment/data')
    out1 = obj.driver(text_preprocessing.text_processing(query))
    out2 = obj.predict_sentence(text_preprocessing.text_processing(query))

    collector = []
    if out1:
        collector += out1
    if out2:
        if len(out2) > 1 or out2[0].lower() != query.lower():
            collector += out2
    if collector:
        return json.dumps(collector)
    else:
        return ''


@app.route('/process', methods=['POST', 'GET'])
def process():
    query = request.args.get('msg')
    return str(FaqBot().generate_reply(query))


# @app.route("/process")
# def getBotResponse():
#     q = request.args.get('msg')
#     return str(english_bot.get_response(q))


# @app.route('/words/', methods=["POST"])
# def predict():
#     request_param = request.form
#     query = request_param['query']
#     data_dir = request_param['data_dir']
#     recommend_words = Predictor(data_dir, query).driver()
#
#     return json.dumps(recommend_words)
#
#
# @app.route('/reply/', methods=["POST"])
# def reply():
#     request_param = request.form
#     query = request_param['query']
#     data_dir = request_param['data_dir']
#     page_id = request_param['page_id']
#     caller = Predictor(data_dir, query)
#
#     return json.dumps(caller.chat_reply(page_id))
#
#
# @app.route('/sentence/', methods=["POST"])
# def sentence():
#     request_param = request.form
#     query = request_param['query']
#     data_dir = request_param['data_dir']
#     caller = Predictor(data_dir, query)
#
#     return json.dumps(caller.predict_sentence())


if __name__ == '__main__':
    app.run()
