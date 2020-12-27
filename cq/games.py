# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 10:23:27 2020

@author: as
"""

import random
import sys
import math
import numpy as np 

"""
up -> 0
right -> 1
down -> 2
left -> 3
"""

class Game:
    
    def compute_game_grid(self, agent1_s, agent2_s):
        grid = self.__fill_grid(agent1_s, agent2_s)
        return np.array(grid)
        
    def __fill_grid(self, agent1_s, agent2_s):
        game_grid = [[" " for y in range(self.no_of_rows)] for x in range(self.no_of_columns)]
        agent1_row, agent1_col = self.__convert_state_to_col_rows(agent1_s)
        game_grid[int(agent1_row)][int(agent1_col)] = 'X'
        agent2_row, agent2_col = self.__convert_state_to_col_rows(agent2_s)
        game_grid[int(agent2_row)][int(agent2_col)] = 'Y'
        return game_grid
    
    def __convert_state_to_col_rows(self, state):
        return (math.floor(state/self.no_of_rows), state%self.no_of_columns)
    
    
    

class Game1(Game):
    goal_state = 1
    top_edge_states = [0,1,2]
    right_edge_states = [2,5,8]
    left_edge_states = [0,3,6]
    bottom_edge_states = [6,7,8]
    no_of_states = 9
    no_of_rows = 3
    no_of_columns = 3
    
    no_top_allowed_states = [6,8]
    no_bottom_allowed_states = [3,5]
    
    def next_state_single(self, agent_s, agent_a):
        isGameEnded = False
        agent_s_, agent_r = self.__next_state(agent_s, agent_a)
        if (agent_s_ == 1):
            agent_r = 100
            isGameEnded = True   
        
        return (agent_s_, agent_r, isGameEnded)

    
    def next_state_joint(self, agent1_s, agent1_a, agent2_s, agent2_a):
        agent1_s_, agent1_r = self.__next_state(agent1_s, agent1_a)
        agent2_s_, agent2_r = self.__next_state(agent2_s, agent2_a)
        
        collisionExists = False
        isGameEnded = False
        if (agent1_s_ == agent2_s_ and agent1_s_ == self.goal_state):
            agent1_r = 100
            agent2_r = 100
            isGameEnded = True
        elif (agent1_s_ == agent2_s_ and agent1_s_ != self.goal_state):
            collisionExists = True
            agent1_r = -1
            agent2_r = -1
            agent1_s_ = agent1_s
            agent2_s_ = agent2_s
            
        return (agent1_s_, agent1_r, agent2_s_, agent2_r, collisionExists, isGameEnded)

    def __next_state(self, s, a):
        
        if (s == self.goal_state):
            s_ = s
            return (s_,0)
        
        if (s in self.no_top_allowed_states and a==0):
            if (random.random() > 0.5):
                s_ = s
            else:
                s_ = s-3
            r = 0
            return (s_, r)
        
        if (s in self.no_bottom_allowed_states and a==2):
            if (random.random() > 0.5):
                s_ = s
            else:
                s_ = s+3
            r = 0
            return (s_, r)
        
        if (a == 0):
            if (s in self.top_edge_states):
                s_ = s
                r = -1
            else:
                s_ = s-3
                r = 0
        elif (a == 2):
            if (s in self.bottom_edge_states):
                s_ = s
                r = -1
            else:
                s_ = s+3
                r = 0
        elif (a == 1):
            if (s in self.right_edge_states):
                s_ = s
                r = -1
            else:
                s_ = s+1
                r = 0
        else:
            if (s in self.left_edge_states):
                s_ = s
                r = -1
            else:
                s_ = s-1
                r = 0
        
        return (s_,r)

    
    
            


class Game2(Game):
    goal_state = 10
    top_edge_states = [0,1,2,3,4]
    right_edge_states = [4,9,14,19,24]
    left_edge_states = [0,5,10,15,20]
    bottom_edge_states = [20,21,22,23,24]
    no_of_states = 25
    no_of_rows = 5
    no_of_columns = 5

    no_top_allowed_states = [10,11,12,13] + [15,16,17,18]
    no_bottom_allowed_states = [5,6,7,8] + [10,11,12,13]
    
    def next_state_single(self, agent_s, agent_a):
        isGameEnded = False
        agent_s_, agent_r = self.__next_state(agent_s, agent_a)
        if (agent_s_ == self.goal_state):
            agent_r = 100
            isGameEnded = True   
        
        return (agent_s_, agent_r, isGameEnded)

    
    def next_state_joint(self, agent1_s, agent1_a, agent2_s, agent2_a):
        agent1_s_, agent1_r = self.__next_state(agent1_s, agent1_a)
        agent2_s_, agent2_r = self.__next_state(agent2_s, agent2_a)
        
        collisionExists = False
        isGameEnded = False
        if (agent1_s_ == agent2_s_ and agent1_s_ == self.goal_state):
            agent1_r = 100
            agent2_r = 100
            isGameEnded = True
        elif (agent1_s_ == agent2_s_ and agent1_s_ != self.goal_state):
            collisionExists = True
            agent1_r = -1
            agent2_r = -1
            agent1_s_ = agent1_s
            agent2_s_ = agent2_s
            
        return (agent1_s_, agent1_r, agent2_s_, agent2_r, collisionExists, isGameEnded)

    def __next_state(self, s, a):
        
        if (s == self.goal_state):
            s_ = s
            return (s,0)
        
        if (s in self.no_bottom_allowed_states and a==2):
            s_ = s
            r = -1
            return (s_, r)
        
        if (s in self.no_top_allowed_states and a==0):
            s_ = s
            r = -1
            return (s_, r)

        if (a == 0):
            if (s in self.top_edge_states):
                s_ = s
                r = -1
            else:
                s_ = s-5
                r = 0
        elif (a == 2):
            if (s in self.bottom_edge_states):
                s_ = s
                r = -1
            else:
                s_ = s+5
                r = 0
        elif (a == 1):
            if (s in self.right_edge_states):
                s_ = s
                r = -1
            else:
                s_ = s+1
                r = 0
        else:
            if (s in self.left_edge_states):
                s_ = s
                r = -1
            else:
                s_ = s-1
                r = 0
        
        return (s_,r)
            