from Tag_recommender.recommend import Recommender

DEFAULT_PRIORITY = 1


class Agent:
    def __init__(self, agent_id, model_dir, data_dir):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.agent_id = str(agent_id)
        self._priority = DEFAULT_PRIORITY

    def reset_priority(self):
        self._priority = DEFAULT_PRIORITY

    def update_priority(self, priority: int = 1):
        self._priority += priority

    def priority(self):
        return self._priority

    def load_recommender(self):
        return Recommender(self.model_dir, self.data_dir)
