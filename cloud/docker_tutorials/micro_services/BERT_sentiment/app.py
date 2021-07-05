from logger import logger
from PosNegDetector import agent_manager
from flask import Flask, request

app = Flask("sentiment detection")


# just for testing
@app.route("/", methods=["GET", "POST"])
def index():
    return "Hello World !"


@app.route('/sentiment', methods=["POST"])
def run():
    res = {}
    try:
        request_param = request.get_json()
        logger.debug(request_param)
        model_path = request_param['model_path']
        agent_id = request_param.get('agent_id', False)
        if not agent_id:
            agent_id = -1
        query = request_param['query']
        current_agent = agent_manager.get_agent(agent_id, model_path)
        res["sentiment"] = current_agent.sentiment_analyzer(query)
    except:
        logger.exception(f'{__name__} Exception occurred')
        res["error_message"] = "An exception occurred"

    return res


app.run(host="0.0.0.0", port=8001, debug=True)
