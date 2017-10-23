from .agent.dummy import RandomAgent
from .agent.dqn import DQNAgent
from .agent.manual import ManualAgent
from .agent.minimax import MinimaxAgent, MinimaxABAgent
from .agent.mcts import MCTSAgent
from .agent.tldqn import TLDQNAgent


def choose_agent(message, board_shape, sign):
    while True:
        print
        print(message)
        print_horizontal_line()
        print('[1] Manual (Human)')
        print('[2] Random (Dummy)')
        print('[3] MiniMax (Naive)')
        print('[4] MiniMax (Alpha Beta Pruning)')
        print('[5] Monte Carlo Tree Search')
        print('[6] Deep Q Network')
        print('[7] Tensorlayer DQN')
        print_horizontal_line()
        try:
            number = eval(input('Enter [1-6]: '))
            if number == 1:
                return ManualAgent()
            if number == 2:
                return RandomAgent()
            if number == 3:
                number = eval(input('Max Depth: '))
                return MinimaxAgent(number)
            if number == 4:
                number = eval(input('Max Depth: '))
                return MinimaxABAgent(number)
            if number == 5:
                number = eval(input('Max Seconds: '))
                return MCTSAgent(number)
            if number == 6:
                (rows, cols) = board_shape
                return DQNAgent(rows, cols, sign, learning_on=False)
            if number == 7:
                (rows, cols) = board_shape
                return TLDQNAgent(rows, cols, sign, learning_on=False)
        except Exception as e:
            print(e)


def print_horizontal_line(width=40):
    print('-' * width)
