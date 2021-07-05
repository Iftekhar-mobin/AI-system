from user_input_correction.spelling_correction import SpellingCheckerCorrector

DEFAULT_PRIORITY = 1


class Agent:
    def __init__(self, agent_id, data_dir):
        self.data_dir = data_dir
        self.agent_id = str(agent_id)
        self._priority = DEFAULT_PRIORITY

    def reset_priority(self):
        self._priority = DEFAULT_PRIORITY

    def update_priority(self, priority: int = 1):
        self._priority += priority

    def priority(self):
        return self._priority

    def load_spelling_check(self):
        return SpellingCheckerCorrector(self.data_dir)
