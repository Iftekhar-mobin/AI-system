from logger import logger
import ast
from flask import Flask, request
from synonym_generator.synonyms import add_users_synonyms


app = Flask("Synonym Generator")


# just for testing
@app.route("/", methods=["GET", "POST"])
def index():
    return "Hello World !"


@app.route('/synonym', methods=["POST"])
def run():
    res = {}
    try:
        request_param = request.json
        logger.debug(request_param)
        data_dir = request_param['data_dir']
        synonyms = ast.literal_eval(request_param['synonyms'])
        res["message"] = add_users_synonyms(data_dir, synonyms)

    except:
        logger.exception(f'{__name__} Exception occurred')
        res["error_message"] = "An exception occurred"
    finally:
        logger.info("Synonyms registered successfully.")

    return res


app.run(host="0.0.0.0", port=8003, debug=True)
