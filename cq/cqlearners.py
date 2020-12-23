# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 10:26:22 2020

@author: as
"""


import numpy as np
import random
from scipy import stats
from collections import deque

class ActionSelector: 
    epison = None
    discount_factor = None
    learning_rate = None

    q_values = None
    no_of_states = None
    no_of_actions = None
    name = None
    
    def __init__(self, name, no_of_states, no_of_actions, epison=0.1, discount_factor=0.9, learning_rate=0.1):
        self.epison = epison
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.no_of_states = no_of_states
        self.no_of_actions = no_of_actions
        self.name = name
        
    def selectAction(self, state):
        if (random.random() < self.epison):
            return self.randomAction()
        else:
            return self.bestAction(state)
    
    def randomAction(self):
        return np.random.choice(self.no_of_actions)
        
    def bestAction(self, state):
        return np.random.choice(np.flatnonzero(self.q_values[state] == max(self.q_values[state])))
    
        
class LocalActionSelector(ActionSelector): 
    def setup(self):
        self.q_values = np.zeros((self.no_of_states, self.no_of_actions))
        
    def maxQValue(self, state):
        return max(self.q_values[state])

    def updateQValues(self, state, action, next_state, reward): 
        old_q_value = self.q_values[state][action]
        new_q_value = reward + self.discount_factor * max(self.q_values[next_state])
        self.q_values[state][action] = old_q_value + self.learning_rate * (new_q_value - old_q_value)
        
        
class GlobalActionSelector(ActionSelector): 
    def setup(self):
        self.q_values = {}
    
    def addState(self, state):
        if (state not in self.q_values.keys()):
            self.q_values[state] = np.zeros(self.no_of_actions)
        
    def updateQValuesWithMaxQ(self, state, action, reward, maxQValue): 
        if (state not in self.q_values.keys()):
            self.q_values[state] = np.zeros(self.no_of_actions)
            
        old_q_value = self.q_values[state][action]
        new_q_value = reward + self.discount_factor * maxQValue
        self.q_values[state][action] = old_q_value + self.learning_rate * (new_q_value - old_q_value)
        
        
        

class CQLearner: 
    
    coordination_joint_states = None
    coordination_joint_states_confidence = None
    
    local_action_selector = None
    global_action_selector = None
    
    initial_rewards = None
    latest_rewards = None
    
    no_of_actions = None
    no_of_states = None
    
    epison = None
    discount_factor = None
    learning_rate = None
    sliding_window_size = None
    name = None
    
    def __init__(self, name, no_of_states, no_of_actions, sliding_window_size=60, epison=0.1, discount_factor=0.9, learning_rate=0.1):
        self.name = name
        self.no_of_actions = no_of_actions
        self.no_of_states = no_of_states
        self.epison = epison
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.sliding_window_size = sliding_window_size
        
        self.coordination_joint_states = []
        self.coordination_joint_states_confidence = {}
        
        self.local_action_selector = LocalActionSelector(name, no_of_states, no_of_actions, epison, discount_factor, learning_rate)
        self.local_action_selector.setup()
        self.global_action_selector = GlobalActionSelector(name, no_of_states, no_of_actions, epison, discount_factor, learning_rate)
        self.global_action_selector.setup()
    
        self.initial_rewards = [ [deque(maxlen=sliding_window_size) for a in range(no_of_actions)] for s in range(no_of_states) ]
        self.latest_rewards = [ [deque(maxlen=sliding_window_size) for a in range(no_of_actions)] for s in range(no_of_states) ]
    
    
    def reset(self):
        self.coordination_joint_states = []
        self.coordination_joint_states_confidence = {}
        #self.latest_rewards = [ [deque(maxlen=self.sliding_window_size) for a in range(self.no_of_actions)] for s in range(self.no_of_states) ]
    
    def takeAction(self, local_state, global_state):
        isNecessary, joint_states_including_local_state = self.__isCoordinationNecessary(local_state)

        if (isNecessary and (global_state in joint_states_including_local_state) 
          #  and self.coordination_joint_states_confidence[global_state] > np.median(list(self.coordination_joint_states_confidence.values()))
          ):
            action = self.global_action_selector.selectAction(global_state)
            is_joint = True
            self.__incrementCoordinationJointStateConfidence(joint_states_including_local_state)
        else:
            action = self.local_action_selector.selectAction(local_state)
            self.__decrementCoordinationJointStateConfidence(joint_states_including_local_state)
            is_joint = False
            
        return (action, is_joint) 
    
    def __isCoordinationNecessary(self, state):
        joint_actions = [joint_state for joint_state in self.coordination_joint_states if joint_state[0] == state]
        return len(joint_actions) > 0, joint_actions
            
    def __incrementCoordinationJointStateConfidence(self, joint_states):
        for js in joint_states:            
            self.coordination_joint_states_confidence[js] += 1
        
    def __decrementCoordinationJointStateConfidence(self, joint_states):
        for js in joint_states:            
            self.coordination_joint_states_confidence[js] -= 13
            # larger discount valuese preferred for game1 (ex: 13) and larger for game2 (30)
            
            if (self.coordination_joint_states_confidence[js] < 5):
                self.coordination_joint_states.remove(js)
                del self.coordination_joint_states_confidence[js]
                
        
        
    
    def enviornmentFeedback(self, local_state, global_state, action, next_state, reward):
        self.__updateSlidingWindows(local_state, action, reward)
    
        if (self.__isConflictDetected(local_state, action) and self.__isRewardLessThanAverage(local_state, action, reward)):
            self.__markStateAsCoordinationNeeded(global_state)
            
        self.__updateQValues(local_state, global_state, action, next_state, reward)
            
    
    def __updateSlidingWindows(self, local_state, action, reward):
        if len(self.initial_rewards[local_state][action]) < self.sliding_window_size:
            self.initial_rewards[local_state][action].append(reward)
        else:
            self.latest_rewards[local_state][action].append(reward)        
    
    def __isConflictDetected(self, local_state, action):
        test_result = stats.ttest_ind(self.initial_rewards[local_state][action], self.latest_rewards[local_state][action])
        return test_result.pvalue < .05
    
    def __isRewardLessThanAverage(self, local_state, action, reward):
        return stats.ttest_1samp(self.latest_rewards[local_state][action], popmean=reward)
        
    def __markStateAsCoordinationNeeded(self, global_state):
        if(global_state not in self.coordination_joint_states):
            self.coordination_joint_states.append(global_state)
            self.coordination_joint_states_confidence[global_state] = 10
            self.global_action_selector.addState(global_state)
        
    def __updateQValues(self, local_state, global_state, action, next_state, reward):
        isNecessary, _ = self.__isCoordinationNecessary(local_state)
        if (isNecessary):
            self.global_action_selector.updateQValuesWithMaxQ(global_state, action, reward, self.local_action_selector.maxQValue(next_state))
        else:
            self.local_action_selector.updateQValues(local_state, action, next_state, reward)


    def global_state_size(self):
        return len(self.coordination_joint_states)
