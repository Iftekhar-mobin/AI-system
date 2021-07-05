from logger import logger
from flask import Flask, request
from Tag_recommender import agent_manager

app = Flask("Tag Recommendation")


# just for testing
@app.route("/", methods=["GET", "POST"])
def index():
    return "Hello World !"


@app.route('/recommend', methods=["POST"])
def run():
    res = {}
    try:
        request_param = request.json
        logger.debug(request_param)
        model_dir = request_param['output_dir']
        data_dir = request_param['data_dir']
        query = request_param['query']
        agent_id = request_param.get('agent_id', False)
        if not agent_id:
            agent_id = -1
        current_agent = agent_manager.get_agent(agent_id, model_dir, data_dir).load_recommender()

        data = {
            "output_dir": model_dir,
            "data_dir": data_dir,
            "agent_id": -1,
            "query": query,
            "max_answer": 5
        }

        res["recommend_tag"] = current_agent.get_recommend(data)

    except:
        logger.exception(f'{__name__} Exception occurred')
        res["error_message"] = "An exception occurred"

    return res


app.run(host="0.0.0.0", port=8000, debug=True)
