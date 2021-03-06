import random
from scipy import stats
from .action import Action
from collections import deque

class CQLearner: 
        
    def __init__(self, name, initial_state, sliding_window_size=60, epsilon=0.1, discount_factor=0.9, learning_rate=0.1):
        self.name = name
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.sliding_window_size = sliding_window_size
        self.possible_actions = [Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST]
        
        self.coordination_states_confidence = {}

        self.local_q_table = {}
        self.global_q_table = {}
        
        self.initial_rewards = {}
        self.latest_rewards = {}
        self.initial_state = initial_state
        self.local_state = initial_state[self.name]
        self.global_state = frozenset(initial_state.items())
        self.previous_action = None

    def select_action(self):
        use_global = self.global_state in self.coordination_states_confidence
        if use_global:
            self.previous_action = self.__select_action_with_q_table(self.global_q_table, self.global_state)
            self.__increment_coordination_confidence()
        else:
            self.previous_action = self.__select_action_with_q_table(self.local_q_table, self.local_state)
            self.__decrement_coordination_confidence()
        return self.previous_action

    def update_state(self, global_state, reward):
        """Update the state according to the previous action."""    

        old_local_state = self.local_state
        old_global_state = self.global_state
        # By using a frozenset, global_state can be used as a key in self.global_q_table
        self.global_state = frozenset(global_state.items()) 
        self.local_state = global_state[self.name]
        
        self.__update_sliding_windows(old_local_state, reward)
        self.__update_conflicting_states(old_global_state, old_local_state, reward)
        self.__update_q_values(self.local_state, old_global_state, old_local_state, reward)

    def reset_state(self):
        """Reset the state between episodes"""
        self.global_state = frozenset(self.initial_state.items())
        self.local_state = self.initial_state[self.name]

    def get_state_space_size(self):
        return len(self.local_q_table) + len(self.coordination_states_confidence)
            
    def __select_action_with_q_table(self, q_table, state):
        """Selects an action using the given Q table and state.
        The Q table and state could be either local or global
        """
        if random.random() < self.epsilon:
            return random.choice(self.possible_actions)
        else:
            action_table = q_table.get(state)
            if action_table:
                max_value = max(action_table.values())
                max_actions = [ action for action,val in action_table.items() if val==max_value ]
                # If there are multiple actions with the same value, we select one randomly
                # This should help add additional exploration
                return random.choice(max_actions)
            else:
                # If the Q table hasn't been initialized yet for this state, select an arbitrary action
                return random.choice(self.possible_actions)
        
    def __update_q_values(self, new_local_state, old_global_state, old_local_state, reward):
        use_global = old_global_state in self.coordination_states_confidence

        q_table = self.global_q_table if use_global else self.local_q_table
        old_state = old_global_state if use_global else old_local_state

        if not q_table.get(old_state):
            # The Q-values for the state haven't been initialized yet
            q_table[old_state] = dict.fromkeys(self.possible_actions, 0)
            
        max_q_value_next_action = max(self.local_q_table.get(new_local_state).values()) if self.local_q_table.get(new_local_state) else 0
        old_q_value = q_table[old_state][self.previous_action]
        new_q_value = reward + self.discount_factor * max_q_value_next_action
        q_table[old_state][self.previous_action] = old_q_value + self.learning_rate * (new_q_value - old_q_value)
        

    def __update_conflicting_states(self, old_global_state, old_local_state, reward):
        if old_global_state in self.coordination_states_confidence:
            return
        
        conflict_detected = self.__is_conflict_detected(old_local_state, self.previous_action)
        reward_lower = self.__is_reward_maybe_less_than_average(old_local_state, self.previous_action, reward)
        if conflict_detected and reward_lower and old_global_state not in self.coordination_states_confidence.keys():
            self.coordination_states_confidence[old_global_state] = 30
    
    def __increment_coordination_confidence(self):
        for state in self.coordination_states_confidence:
            state_dict = dict(state)
            if state_dict[self.name] == self.local_state:
                self.coordination_states_confidence[state] += 2
                
    def __decrement_coordination_confidence(self):
        for state in self.coordination_states_confidence:
            state_dict = dict(state)
            if state_dict[self.name] == self.local_state:                
                self.coordination_states_confidence[state] -= 1

        # Remove coordination states that have low confidence
        self.coordination_states_confidence = {
            state: val for state, val in self.coordination_states_confidence.items() if val > 0
        }
    
    def __update_sliding_windows(self, old_local_state, reward):

        if not self.initial_rewards.get(old_local_state):
            self.initial_rewards[old_local_state] = { action:[] for action in self.possible_actions }

        if len(self.initial_rewards[old_local_state][self.previous_action]) < self.sliding_window_size:
            self.initial_rewards[old_local_state][self.previous_action].append(reward)
        else:
            if not self.latest_rewards.get(old_local_state):
                self.latest_rewards[old_local_state] = { action:deque(maxlen=self.sliding_window_size) for action in self.possible_actions }
            self.latest_rewards[old_local_state][self.previous_action].append(reward)
    
    def __is_conflict_detected(self, local_state, action):
        initial_rewards = self.initial_rewards.get(local_state)
        latest_rewards = self.latest_rewards.get(local_state)
        if not (initial_rewards and latest_rewards
                and len(initial_rewards[action]) == self.sliding_window_size
                and len(latest_rewards[action]) == self.sliding_window_size):
            return False
        test_result = stats.ttest_ind(initial_rewards[action], latest_rewards[action])
        # Low p value -> unlikely they are from the same distribution
        return test_result.pvalue < 0.2

    def __is_reward_maybe_less_than_average(self, local_state, action, reward):
        latest_rewards = self.latest_rewards.get(local_state)
        if not (latest_rewards and len(latest_rewards[action]) == self.sliding_window_size):
                return False
    
        t_test = stats.ttest_1samp(latest_rewards[action], popmean=reward, alternative="less")
        # is_reward_greater = t_test.statistic < 0

        is_confident = t_test.pvalue < 0.1
        
        return not is_confident
