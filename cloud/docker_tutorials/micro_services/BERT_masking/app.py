from logger import logger
from masking import agent_manager
from flask import Flask, request

app = Flask("Masking Model")


# just for testing
@app.route("/", methods=["GET", "POST"])
def index():
    return "Hello World !"


@app.route('/mask', methods=["POST"])
def run():
    res = {}
    try:
        request_param = request.get_json()
        logger.debug(request_param)
        model_path = request_param['model_path']
        data_dir = request_param['data_dir']
        model_dir = request_param['model_dir']
        agent_id = request_param.get('agent_id', False)
        if not agent_id:
            agent_id = -1
        query = request_param['query']
        current_agent = agent_manager.get_agent(agent_id, model_path, data_dir, model_dir)
        res["mask_word"] = current_agent.rewrite_word.question_rewriter(
            query,
            model=current_agent.mask_analyzer['model'],
            tokenizer=current_agent.mask_analyzer['tokenizer']
        )
    except:
        logger.exception(f'{__name__} Exception occurred')
        res["error_message"] = "An exception occurred"

    return res


app.run(host="0.0.0.0", port=8002, debug=True)
