class Agent(object):
    def start(self):
        pass

    def decide(self, env, state):
        """ Returns an action to take (or None to skip)
        """
        return None

    def end(self, winner):
        pass

    def __str__(self):
        return self.__class__.__name__