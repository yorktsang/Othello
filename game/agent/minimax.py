from .agent import Agent

INFINITY = float('Inf')


def val_max(seq, fn):
    if len(seq) == 0:
        return fn(None)
    return max(map(fn, seq))


def val_min(seq, fn):
    if len(seq) == 0:
        return fn(None)
    return min(map(fn, seq))


def arg_max(seq, fn):
    if len(seq) == 0:
        return None
    values = map(fn, seq)
    valuesList = list(values)
    #return seq[values.index(max(values))]
    return list(seq)[valuesList.index(max(valuesList))]


class MinimaxAgent(Agent):
    """ Naive minimax agent
    Minimax agent iterate though all the valid moves to find the best (highest value) move.
    For each, it needs to evaluate all the possible following moves by the opponent
    who tries to choose the best move which is the worst move for this agent.
    This traversing of possible move chains will be continued until we reach the max depth
    or there is no more move for both agents.
    """
    def __init__(self, max_depth):
        self._max_depth = max_depth

    def decide(self, env, state):
        return arg_max(env.valid_actions(state),
                       lambda action: self._min_play(env, env.apply(state, action), 1))

    def _max_play(self, env, state, depth):
        if not env.is_active(state) or depth > self._max_depth:
            return state.score(self)
        return val_max(env.valid_actions(state),
                       lambda action: self._min_play(env, env.apply(state, action), depth+1))

    def _min_play(self, env, state, depth):
        if not env.is_active(state) or depth > self._max_depth:
            return state.score(self)
        return val_min(env.valid_actions(state),
                       lambda action: self._max_play(env, env.apply(state, action), depth+1))


class MinimaxABAgent(Agent):
    """ An improved version of minimax agent.
    The agent (=max agent) will choose the best of the worst moves chosen by the
    min agent, and the opponent (=min agent) will choose the worst of the best
    moves chosen by the max agent.
    The max agent can stop traversing child nodes if the current node value is already
    bigger than other sibling nodes (that were evaluated before that node) as the min
    agent will not choose the node anyway.
    The min agent can stop traversing child nodes if the current node value is already
    smaller than other sibling nodes (that were evaluated before that node) as the max
    agent will not choose the node anyway.
    Alpha is the best of the worst values chosen by the min agent from the child nodes.
    If the min agent finds that the current node value is less than the alpha, it stops.
    Beta is the worst of the best values chosen by the max agent from the child nodes.
    If the max agent finds that the current node value is more than the beta, it stops.
    """
    def __init__(self, max_depth):
        self._max_depth = max_depth

    def decide(self, env, state):
        return arg_max(env.valid_actions(state),
                       lambda action: self._min_play(env, env.apply(state, action),
                                                     -INFINITY, INFINITY, 1))

    def _max_play(self, env, state, alpha, beta, depth):
        if not env.is_active(state) or depth > self._max_depth:
            return state.score(self)
        actions = env.valid_actions(state)
        if len(actions) == 0:
            return self._min_play(env, env.apply(state, None),
                                  alpha, beta, depth)
        value = -INFINITY
        for action in actions:
            value = max(value, self._min_play(env, env.apply(state, action),
                                              alpha, beta, depth+1))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    def _min_play(self, env, state, alpha, beta, depth):
        if not env.is_active(state) or depth > self._max_depth:
            return state.score(self)
        actions = env.valid_actions(state)
        if len(actions) == 0:
            return self._max_play(env, env.apply(state, None),
                                  alpha, beta, depth)
        value = INFINITY
        for action in actions:
            value = min(value, self._max_play(env, env.apply(state, action),
                                              alpha, beta, depth+1))
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value