import numpy as np

class AgenteDet():

    def __init__(self):
        self.curr_act = 0
        self.curr_state = 0
    
    def predict(self,observation):
        action = np.random.randint(observation.shape[0])
        return action, self.curr_state
    