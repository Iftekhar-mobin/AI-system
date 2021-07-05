import time
from logger import logger
from flask import Flask, request
from google_assistant import google_assistant_client as google_assistant

app = Flask("Google Assistant")


# just for testing
@app.route("/", methods=["GET", "POST"])
def index():
    return "Hello World !"


@app.route('/assistant', methods=["POST"])
def run():
    res = {}
    try:
        request_param = request.json
        logger.debug(request_param)
        query = request_param['query']
        user_nm = request_param['user_nm']
        start = time.time()
        res['answer'] = google_assistant.request_google_assistant(query, user_nm)
        logger.info(res['answer'])
        end = time.time()
        logger.info("predict-Done: {}".format(end - start))
        res['status_code'] = 1
    except:
        logger.exception(f'{__name__} Exception occurred')
        res = {
            "status_code": -1,
            "error_message": "An exception occurred",
        }
    logger.debug(res)
    return res


app.run(host="0.0.0.0", port=8004, debug=True)
