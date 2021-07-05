from PosNegDetector.pos_neg_retrieve import load_sent_retriever

DEFAULT_PRIORITY = 1


def load_sentiment_analyzer(model_path):
    return load_sent_retriever(model_path)


class Agent:
    def __init__(self, agent_id, model_path):
        self.agent_id = str(agent_id)
        self._priority = DEFAULT_PRIORITY
        self.sentiment_analyzer = load_sentiment_analyzer(model_path)

    def reset_priority(self):
        self._priority = DEFAULT_PRIORITY

    def update_priority(self, priority: int = 1):
        self._priority += priority

    def priority(self):
        return self._priority
