from .agent import Agent
import time
import random
from math import log, sqrt


class StateInfo(object):
    def __init__(self):
        self.wins = 0
        self.plays = 0


class StateInfoMap(object):
    def __init__(self):
        self._states = {}
        self._c = sqrt(2.0)  # theoretically sqrt(2); in practice usually chosen empirically

    def add(self, state):
        if state not in self._states:
            self._states[state] = StateInfo()

    def exists(self, state):
        return state in self._states

    def unexplored(self, states):
        return [state for state in states if not self.exists(state)]

    def update(self, states, won):
        for state in states:
            if self.exists(state):
                info = self._states[state]
                info.plays += 1
                if won:
                    info.wins += 1

    def wins(self, state):
        if not self.exists(state):
            return 0
        return self._states[state].wins

    def plays(self, state):
        if not self.exists(state):
            return 0
        return self._states[state].plays

    def win_ratio(self, state):
        if not self.exists(state):
            return 0.0
        info = self._states[state]
        if info.plays == 0:
            return 0.0
        return info.wins * 1.0 / info.plays

    def best_state(self, states):
        """ Using UCB1 select the best state
        """
        log_sum = log(sum(self.plays(state) for state in states))
        _, best_state = max([(self.win_ratio(state)
                              + self._c * sqrt(log_sum / self.plays(state)), state)
                             for state in states])
        return best_state

    def best_action(self, transitions):
        ret1 = (action1, win_ratio1, wins1, plays1) = self.action_by_visits(transitions)
        ret2 = (action2, win_ratio2, wins2, plays2) = self.action_by_win_ratio(transitions)
        if plays1 == plays2:
            return ret1 if win_ratio1 > win_ratio2 else ret2
        else:
            return ret1 if plays1 > plays2 else ret2

    def action_by_visits(self, transitions):
        plays, state, action = max((self.plays(s), s, a) for s, a in transitions)
        wins, win_ratio = self.wins(state), self.win_ratio(state)
        return action, win_ratio, wins, plays

    def action_by_win_ratio(self, transitions):
        win_ratio, state, action = max((self.win_ratio(s), s, a) for s, a in transitions)
        wins, plays = self.wins(state), self.plays(state)
        return action, win_ratio, wins, plays


class MCTSAgent(Agent):
    def __init__(self, max_seconds):
        self._max_seconds = max_seconds
        self._state_info_map = StateInfoMap()

    def decide(self, env, state):
        valid_actions = env.valid_actions(state)
        if len(valid_actions) == 0:
            return None
        if len(valid_actions) == 1:
            return list(valid_actions)[0]

        count = 0
        start = time.time()
        while time.time() - start < self._max_seconds:
            self._simulate(env, state)
            count += 1

        transitions = [(env.apply(state, action), action) for action in valid_actions]
        action, win_ratio, wins, plays = self._state_info_map.best_action(transitions)
        print('Win Ratio: {:.2f}% ({}/{})[{}]'.format(win_ratio * 100.0, wins, plays, count))
        return action

    def _simulate(self, env, state):
        expand = True
        visited = set()

        while env.is_active(state):
            agent = state.agent  # agent that is acting on this state

            # selection
            states = [env.apply(state, action) for action in env.valid_actions(state)]
            if len(states) > 0:
                unexplored = self._state_info_map.unexplored(states)
                if len(unexplored) == 0:
                    # exploitation
                    state = self._state_info_map.best_state(states)
                else:
                    # exploration
                    state = random.choice(unexplored)
            else:
                state = env.apply(state, None)  # pass to the opponent

            # expansion of tree / node
            if agent == self:
                visited.add(state)
                if expand:
                    if not self._state_info_map.exists(state):
                        expand = False
                        self._state_info_map.add(state)

        # back propagation of wins / plays
        self._state_info_map.update(visited, env.winner(state) == self)