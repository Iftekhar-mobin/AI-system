from logger import logger
from user_input_correction import agent_manager
from flask import Flask, request

app = Flask("Spelling correction")


# just for testing
@app.route("/")
def index():
    return "Hello World !"


@app.route('/spelling', methods=["POST"])
def run():
    res = {}
    try:
        request_param = request.get_json()
        logger.debug(request_param)
        data_dir = request_param['data_dir']
        agent_id = request_param.get('agent_id', False)
        if not agent_id:
            agent_id = -1
        query = request_param['query']
        current_agent = agent_manager.get_agent(agent_id, data_dir).load_spelling_check()
        res["spelling"] = current_agent.spelling_checker_suggester(query)
    except:
        logger.exception(f'{__name__} Exception occurred')
        res["error_message"] = "An exception occurred"

    return res


app.run(host="0.0.0.0", port=8000, debug=True)
