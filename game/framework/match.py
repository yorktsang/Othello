import time


class SimpleMatch(object):
    def __init__(self, env, logging_on=True):
        self._env = env
        self._logging_on = logging_on

    def run(self, state):
        """ A simple implementation of turn-based game match
        """
        self.on_game_start(state)

        turn = 1
        while self._env.is_active(state):
            start = self.on_turn_start(state, turn)
            action = state.agent.decide(self._env, state)
            state = self._env.apply(state, action)
            self.on_turn_end(action, start)
            turn += 1

        return self.on_game_end(state)

    def on_game_start(self, state):
        if self._logging_on:
            print_message("Game start")
        state.agent.start()
        state.opponent.start()

    def on_turn_start(self, state, turn):
        start = time.time()
        agent = state.agent
        if self._logging_on:
            print('[{}] Turn: {} ({})'.format(time.strftime("%H:%M:%S", time.localtime(start)), turn, agent))
            print(state.board)
            self._env.print_summary(state)
        return start

    def on_turn_end(self, action, start):
        elapsed = time.time() - start
        if self._logging_on:
            print('Move: {} Elapsed: {:.2f}s\n'.format(action, elapsed))

    def on_game_end(self, state):
        winner = self._env.winner(state)
        if self._logging_on:
            print('[{}] Winner: {}'.format(
                time.strftime("%H:%M:%S", time.localtime(time.time())), winner))
        if self._logging_on:
            print(state.board)
        state.agent.end(winner)
        state.opponent.end(winner)
        if self._logging_on:
            self._env.print_summary(state)
            print_message("Game end")
        return winner


def print_message(message, width=40):
    print
    print('-' * width)
    print(message)
    print('-' * width)
    print
