from masking.mask import load_bert, Rewriter

DEFAULT_PRIORITY = 1


def load_bert_model(model_path):
    return load_bert(model_path)


class Agent:
    def __init__(self, agent_id, model_path, data_dir, model_dir):
        self.agent_id = str(agent_id)
        self._priority = DEFAULT_PRIORITY
        self.mask_analyzer = load_bert_model(model_path)
        self.rewrite_word = Rewriter(data_dir, model_dir)

    def reset_priority(self):
        self._priority = DEFAULT_PRIORITY

    def update_priority(self, priority: int = 1):
        self._priority += priority

    def priority(self):
        return self._priority
