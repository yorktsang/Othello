from .agent import Agent


class ManualAgent(Agent):
    def decide(self, env, state):
        valid_actions = env.valid_actions(state)
        if len(valid_actions) == 0:
            return None
        while True:
            command = input('Enter a move row, col: ')
            if command == 'quit':
                exit(0)
            elif command == 'show':
                print('Valid moves: {}'.format(valid_actions))
            elif command == 'help':
                print('quit: terminate the match')
                print('help: show this help')
            else:
                try:
                    action = eval(command)
                except Exception as e:
                    print('Invalid input: {}'.format(e))
                if action in valid_actions:
                    return action
                print('Invalid move: {}'.format(action))