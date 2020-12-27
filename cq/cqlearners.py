# -*- coding: utf-8 -*-

import numpy as np
import random
from scipy import stats
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
        
        self.initial_rewards = {}
        self.latest_rewards = {}
        self.state = initial_state

    def select_action(self):
        use_global = self.state is in self.coordination_states_confidence
        if use_global:
            self.__select_action_with_table(self.global_q_table, self.state)
            self.__increment_coordination_confidence()
        else:
            self.__select_action_with_table(self.local_q_table, self.state[self.name])
            self.__decrement_coordination_confidence()

    def update_state(self, global_state, reward):
        """Update the state according to the previous action."""    

        self.__update_sliding_windows(global_state, reward)
        self.__update_conflicting_states(global_state, reward)
        self.__update_q_values(global_state, reward)
            
    def __select_action_with_q_table(self, q_table, state):
        """Selects an action using the given Q table and state.
        The Q table and state could be either local or global
        """
        if random.random() < self.epsilon:
            return random.choice(self.possible_actions)
        else:
            action_table = q_table.get(state)
            # If the Q table hasn't been initialized yet for this state, select an arbitrary action (NORTH)
            return max(action_table, key=action_table.get) if action_table else Action.NORTH
        
    def __upate_q_values(self, global_state, reward):
        new_local_state = global_state[self.name]
        use_global = global_state is in self.coordination_states_confidence

        q_table = self.global_q_table if use_global else self.local_q_table
        old_state = self.state if use_global else self.state[self.name]

        if not q_table.get(old_state):
            # The Q-values for the state haven't been initialized yet
            q_table[old_state] = dict.fromkeys(self.possible_actions, 0)
            
        max_q_value_next_action = max(self.local_q_table.get(new_local_state).values()) if self.q_table.get(new_local_state) else 0
        old_q_value = q_table[old_state][self.previous_action]
        new_q_value = reward + self.discound_factor * max_q_value_next_action
        q_table[old_state][self.previous_action] = old_q_value + self.learning_rate * (new_q_value - old_q_value)
        
        self.state = global_state

    def __update_conflicting_states(self, global_state, reward):
        local_state = global_state[self.name]
        conflict_detected = self.__is_conflict_detected(local_state, self.previous_action)
        reward_lower = self.__is_reward_less_than_average(local_state, self.previous_action, reward)
        if conflict_detected and reward_lower and global_state not in self.coordination_states_confidence.keys():
            self.coordination_states_confidence[global_state] = 10
    
    def __isCoordinationNecessary(self, state):
        joint_actions = [joint_state for joint_state in self.coordination_joint_states if joint_state[0] == state]
        return len(joint_actions) > 0, joint_actions
            
    def __increment_coordination_confidence(self):
        for state in self.coordination_states_confidence:
            if state[self.name] == self.state[self.name]:
                self.coordination_states_confidence[state] += 2
                
    def __decrement_coordination_confidence(self):
        for state in self.coordination_states:
            if state[self.name] != self.state[self.name]:
                continue
            self.coordination_states_confidence[state] -= 1
            if self.coordination_states_confidence[state] < 0:
                del self.coordination_states_confidence[state]
    
    def __update_sliding_windows(self, global_state, reward):
        local_state = global_state[self.name]
        action = self.previous_action

        if not self.initial_rewards.get(local_state):
            self.initial_rewards[local_state] = { action:[] for action in self.possible_actions }
            
        if len(self.initial_rewards[local_state][action]) < self.sliding_window_size:
            self.initial_rewards[local_state][action].append(reward)
        else:
            if not self.latest_rewards.get(local_state):
                self.latest_rewards[local_state] = { action:[] for action in self.possible_actions }
            self.latest_rewards[local_state][action].append(reward)
    
    def __is_conflict_detected(self, local_state, action):
        test_result = stats.ttest_ind(self.initial_rewards[local_state][action], self.latest_rewards[local_state][action])
        return test_result.pvalue < 0.10

    def __is_reward_less_than_average(self, local_state, action, reward):
        # TODO: Fix
        return True
        # return stats.ttest_1samp(self.latest_rewards[local_state][action], popmean=reward)                
