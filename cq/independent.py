import random
from .action import Action

class IndependentAgent:
    """An independent agent using Q-learning"""

    def update_state(self, new_state, reward, previous_action):
        """Update the state according to the previous action
        
        This method will also update the Q-table based on the reward.
        """
        # TODO
        
        

    def select_epsilon_greedy_action(self):
        """Uses an epsilon-greedy policy to select an action.
        
        A greedy action is selected with probability 1 - epsilon.
        A random action is elected with probability epsilon.
        """

        if random.random() < self.epsilon:
            return self.select_random_action()
        else:
            return self.select_greedy_action()
        

    def select_greedy_action(self):
        """Greedily selects an action based on the current Q-table"""
        
        action_table = self.q_table.get(self.state)

        if not action_table:
            # The Q table hasn't been initialized yet for this state, so we select an arbitrary action
            return Action.NORTH

        return max(action_table, key=action_table.get)

    def select_random_action(self):
        return random.choice([Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST])
        

    def __init__(self, learning_rate, epsilon, discount_factor, state):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.q_table = {}
        self.state = state
