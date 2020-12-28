import random
from .action import Action

class IndependentAgent:
    """An independent agent using Q-learning"""

    def update_state(self, global_state, reward):
        """Update the state according to the previous action
        
        This method will also update the Q-table based on the reward.
        """
        new_state = global_state[self.name]
        
        if not self.q_table.get(self.local_state):
            # The Q-values for self.state haven't been initialized yet
            self.q_table[self.local_state] = dict.fromkeys(self.possible_actions, 0)

        previous_q_value = self.q_table[self.local_state][self.previous_action]
        max_next_action = max(self.q_table.get(new_state).values()) if self.q_table.get(new_state) else 0
        self.q_table[self.local_state][self.previous_action] = previous_q_value + \
            self.learning_rate(self.time_step) * (reward + self.discount_factor * max_next_action - previous_q_value)
        self.local_state = new_state
        self.time_step += 1

    def select_action(self):
        """Uses an epsilon-greedy policy to select an action.
        
        A greedy action is selected with probability 1 - epsilon.
        A random action is elected with probability epsilon.
        """

        if random.random() < self.epsilon:
            action = self.__select_random_action()
            self.previous_action = action
            return action
        else:
            action = self.__select_greedy_action()
            self.previous_action = action
            return action

    def reset_state(self):
        """Reset the state between episodes"""
        self.local_state = self.initial_state
        
    def __select_greedy_action(self):
        """Greedily selects an action based on the current Q-table"""
        
        action_table = self.q_table.get(self.local_state)

        if not action_table:
            # The Q table hasn't been initialized yet for this state,
            # so we select an arbitrary action
            return Action.NORTH

        return max(action_table, key=action_table.get)

    def __select_random_action(self):
        return random.choice(self.possible_actions)
        

    def __init__(self, learning_rate, epsilon, discount_factor, state, name):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.q_table = {}
        self.local_state = state
        self.possible_actions = [Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST]
        self.time_step = 1
        self.name = name
        self.previous_action = None
        self.initial_state = state
