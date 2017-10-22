import itertools

from enum import Enum
from game.framework import Board, Environment, SimpleMatch, State
from game.console import choose_agent


class Disc(Enum):
    EMPTY = 0
    BLACK = -1
    WHITE = 1

    def __str__(self):
        return self.name


class Reversi(Environment):
    def __init__(self, black, white):
        self._black = black
        self._white = white
        self._valid_actions = {}  # { action: flips } where flips = [cell1, cell2, ..]

    def valid_actions(self, state):
        if state not in self._valid_actions:
            self._valid_actions[state] = self._calc_valid_actions(state)
        return self._valid_actions[state].keys()

    def apply(self, state, action):
        board, agent = state.board, state.agent
        if action in self.valid_actions(state):
            flips = self._valid_actions[state][action]
            disc_color = self._disc_color(state.agent)
            board = board.apply([(disc_color, flip) for flip in [action] + flips])
            return state.turn(board, len(flips)+1, -len(flips))
        return state.opposite()

    def print_summary(self, state):
        print('{}: {} ({})'.format(Disc.BLACK, state.score(self._black), self._black))
        print('{}: {} ({})'.format(Disc.WHITE, state.score(self._white), self._white))

    def _disc_color(self, agent):
        return Disc.BLACK if agent == self._black else Disc.WHITE

    def _calc_valid_actions(self, state):
        board, agent = state.board, state.agent
        disc_color = self._disc_color(agent)
        valid_actions = {}
        for cell in itertools.product(range(board.rows), range(board.cols)):
            if not board.is_empty(cell):
                continue
            flips = self._calc_flips(disc_color, board, cell)
            if len(flips) > 0:
                valid_actions[cell] = flips
        return valid_actions

    def _calc_flips(self, disc_color, board, cell):
        flips = []
        for dr, dc in itertools.product(range(-1, 2), range(-1, 2)):
            if (dr, dc) == (0, 0):
                continue
            flips.extend(self._find_flips(disc_color, board, cell, dr, dc, []))
        return flips

    def _find_flips(self, disc_color, board, prev_cell, dr, dc, flippable):
        cell = prev_cell[0] + dr, prev_cell[1] + dc
        if not board.in_bounds(cell) or board.is_empty(cell):
            return []
        if board[cell] == disc_color:
            return flippable
        flippable.append(cell)  # not empty and not player's kind ==> opponent kind
        return self._find_flips(disc_color, board, cell, dr, dc, flippable)


def create_board(rows, cols):
    row, col = rows//2-1, cols//2-1
    grid = [[Disc.EMPTY for _ in range(rows)] for _ in range(cols)]
    grid[row][col] = Disc.WHITE
    grid[row][col+1] = Disc.BLACK
    grid[row+1][col] = Disc.BLACK
    grid[row+1][col+1] = Disc.WHITE
    return Board(grid, Disc.EMPTY, {Disc.EMPTY: ' ', Disc.BLACK: 'X', Disc.WHITE: 'O'})


def main():
    board_size = (8, 8)
    black = choose_agent('Choose a black agent type', board_size, Disc.BLACK.value)
    white = choose_agent('Choose a white agent type', board_size, Disc.WHITE.value)
    match = SimpleMatch(Reversi(black, white))
    match.run(State(create_board(8,8), black, white, 2, 2))


if __name__ == "__main__":
    main()
