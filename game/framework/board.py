import copy
import os


class Board(object):
    def __init__(self, grid, empty, mapping):
        self._grid = grid
        self._empty = empty
        self._mapping = mapping

    @property
    def rows(self):
        return len(self._grid)

    @property
    def cols(self):
        return len(self._grid[0])

    def data(self, sign):
        grid = []
        for row in range(self.rows):
            grid.append([])
            for col in range(self.cols):
                grid[row].append(self[row, col].value*sign)
        return grid

    def in_bounds(self, cell):
        row, col = cell
        return 0 <= row < self.rows and 0 <= col < self.cols

    def __getitem__(self, cell):
        row, col = cell
        return self._grid[row][col]

    def is_empty(self, cell):
        return self[cell] == self._empty

    def apply(self, actions):
        grid = copy.deepcopy(self._grid)
        for action in actions:
            row, col = action[1]
            grid[row][col] = action[0]
        return Board(grid, self._empty, self._mapping)

    def __str__(self):
        s = ' | '
        for col in range(self.cols):
            s += str(col) + ' | '
        s += os.linesep + '-' * (self.cols * 4 + 2)
        for row in range(self.rows):
            s += os.linesep + str(row) + '| '
            for col in range(self.cols):
                s += self._mapping[self._grid[row][col]] + ' | '
            s += os.linesep
            s += '-' * (self.cols * 4 + 2)
        return s

    def __eq__(self, other):
        return self._grid == other._grid

    def __lt__(self,other):
        return self.__hash__() > other.__hash__()

    def __hash__(self):
        return hash(str(self._grid))
