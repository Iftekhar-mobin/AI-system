from logger import logger
from QAGenerator import agent_manager
from flask import Flask, request
import requests

app = Flask("QA Generation")


# just for testing
@app.route("/", methods=["GET", "POST"])
def index():
    return "Hello World !"


@app.route('/qa', methods=["POST"])
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

        requests.get("http://0.0.0.0:8000/spelling")

        res["QAs"] = current_agent.qa_model(query)
    except:
        logger.exception(f'{__name__} Exception occurred')
        res["error_message"] = "An exception occurred"

    return res


app.run(host="0.0.0.0", port=8001, debug=True)
