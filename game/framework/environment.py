class Environment(object):
    def is_active(self, state):
        if len(self.valid_actions(state)) > 0:
            return True
        return len(self.valid_actions(state.opposite())) > 0

    def valid_actions(self, state):
        return []

    def apply(self, state, action):
        return None

    def winner(self, state):
        if self.is_active(state) or state.agent_score == state.opponent_score:
            return None
        return state.agent if state.agent_score > state.opponent_score else state.opponent

    def print_summary(self, state):
        pass