from .agent import Agent
from .dqn import DQNAgent
from .dummy import RandomAgent
from .manual import ManualAgent
from .mcts import MCTSAgent
from .minimax import MinimaxAgent, MinimaxABAgent

__all__ = ['Agent',
           'DQNAgent',
           'RandomAgent',
           'ManualAgent',
           'MCTSAgent',
           'MinimaxAgent',
           'MinimaxABAgent']