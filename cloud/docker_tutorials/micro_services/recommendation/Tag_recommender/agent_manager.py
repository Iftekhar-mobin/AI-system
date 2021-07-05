from logger import logger
from .agent import Agent

NUM_SPELLING_AGENTS = 1


class AgentManager:
    def __init__(self):
        logger.info("Recommender: init()")
        self.agents = []

    def get_agent(self, agent_id, model_dir, data_dir):
        self._set_low_priority_to_agent()
        current_agent = self._get_current_agent(agent_id)
        if current_agent is None:
            current_agent = self._add_new(agent_id, model_dir, data_dir)

        return current_agent

    def remove_agent_by_id(self, agent_id):
        for idx, agent in enumerate(self.agents):
            if agent.agent_id == agent_id:
                self.agents.pop(idx)

    def _get_current_agent(self, agent_id):
        for agent in self.agents:
            if agent.agent_id == agent_id:
                agent.reset_priority()
                return agent
        return None

    def _set_low_priority_to_agent(self):
        for agent in self.agents:
            agent.update_priority()

    def _add_new(self, agent_id, model_dir, data_dir):
        agent = Agent(agent_id, model_dir, data_dir)
        if len(self.agents) == NUM_SPELLING_AGENTS:
            self._remove_low_priority_agent()
        self.agents.append(agent)
        return agent

    def _remove_low_priority_agent(self):
        self.agents = sorted(self.agents, key=lambda k: k.priority())
        del self.agents[-1]
