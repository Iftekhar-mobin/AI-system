from QAGenerator.qa_maker import qa_retriever_load

DEFAULT_PRIORITY = 1


def load_qa_model(model_path):
    return qa_retriever_load(model_path)


class Agent:
    def __init__(self, agent_id, model_path):
        self.agent_id = str(agent_id)
        self._priority = DEFAULT_PRIORITY
        self.qa_model = load_qa_model(model_path)

    def reset_priority(self):
        self._priority = DEFAULT_PRIORITY

    def update_priority(self, priority: int = 1):
        self._priority += priority

    def priority(self):
        return self._priority
