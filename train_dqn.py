from game.framework.match import SimpleMatch
from game.framework.state import State
from game.agent.dqn import DQNAgent
from game.agent.tldqn import TLDQNAgent
from game.console import choose_agent
from reversi import Reversi, create_board, Disc
from collections import deque
import time, random

if __name__ == '__main__':
    board_size = (8, 8)
    (rows, cols) = board_size

    is_dqn_play_white = True
    if random.random() > 0.5:
        is_dqn_play_white = False

    epsilon = input('DQN Epsilon [0.0 - 1.0]: ')

    if(is_dqn_play_white):
        black = choose_agent('Choose a black agent', board_size, Disc.BLACK.value)
        white = TLDQNAgent(rows, cols, Disc.WHITE.value, epsilon=float(epsilon))
    else:
        black = TLDQNAgent(rows, cols, Disc.BLACK.value, epsilon=float(epsilon))
        white = choose_agent('Choose a white agent', board_size, Disc.WHITE.value)

    match = SimpleMatch(Reversi(black, white), logging_on=False)
    win_lose = deque([], 20)
    for i in range(1000):
        winner = match.run(State(create_board(rows,cols), black, white, 2, 2))
        win_lose.append(1.0 if winner == white else 0)
        print('[{}] #{} {} vs {} {} Win={:.2f}%'.format(
            time.strftime("%H:%M:%S", time.localtime(time.time())), i,
            black, white, 'Won ' if winner == white else 'Lost',
            sum(win_lose)*100./len(win_lose)
        ))