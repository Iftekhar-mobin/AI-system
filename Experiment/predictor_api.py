from flask import request
from amie_core.core.retriever.sequence_predictor.predict import Predictor
from flask_restplus import Namespace, Resource
from amie_core.core.utils import logger

from amie_core.core.tokenizer.mecab_tokenizer import MecabTokenizer

# register in __init__.py
api = Namespace('chat', description='chat api')


def tokenize(query):
    tokenizer = MecabTokenizer()
    tokenizer._load_mecab()
    return tokenizer.wakati_base_form(query)


@api.route('/words/', methods=["POST"])
class Chat(Resource):
    @api.doc('chat')
    def post(self):
        request_param = request.form
        logger.debug(request_param)
        query = request_param['query']
        data_dir = request_param['data_dir']
        caller = Predictor(data_dir, query).driver()


        return caller


@api.route('/reply/', methods=["POST"])
class ChatReply(Resource):
    @api.doc('chat')
    def post(self):
        request_param = request.form
        logger.debug(request_param)
        query = request_param['query']
        data_dir = request_param['data_dir']
        page_id = request_param['page_id']
        caller = Predictor(data_dir, query)

        return caller.chat_reply(page_id)


@api.route('/sentences/', methods=["POST"])
class Chat(Resource):
    @api.doc('chat')
    def post(self):
        request_param = request.form
        logger.debug(request_param)
        query = request_param['query']
        data_dir = request_param['data_dir']
        caller = Predictor(data_dir, query)

        return caller.predict_sentence()


@api.route('/test/', methods=["POST"])
class Chat(Resource):
    @api.doc('chat')
    def post(self):
        request_param = request.form
        logger.debug(request_param)
        query = request_param['query']
        return tokenize(query)
