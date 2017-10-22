from .agent import Agent
import random


class RandomAgent(Agent):
    def decide(self, env, state):
        actions = env.valid_actions(state)
        if len(actions) == 0:
            return None
        return random.choice(list(actions))