# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 21:39:11 2020

@author: as
"""

import numpy as np
import matplotlib.pyplot as plt
import games
import cqlearners

np.set_printoptions(suppress=True)




def playOneRun(game, agent1, agent2, agent1_initial_state, agent2_initial_state):
        
    agent1_performance = playIndividually(agent1, game, agent1_initial_state)      
    agent2_performance = playIndividually(agent2, game, agent2_initial_state)

    print(agent1.local_action_selector.q_values)
    print(agent2.local_action_selector.q_values)
    print("============================")
    
    timesteps_performance = []
    collisions_performance = []
    agent1_joint_plays_performance = []
    agent2_joint_plays_performance = []
    agent1_global_state_size_performance = []
    agent2_global_state_size_performance = []
    
    for episode in range(2000):
        timesteps, no_of_collisions, agent1_joint_plays, agent2_joint_plays = playJointEpisode(agent1, agent2, game, agent1_initial_state, agent2_initial_state)
        timesteps_performance.append(timesteps)
        collisions_performance.append(no_of_collisions)
        agent1_joint_plays_performance.append(agent1_joint_plays)
        agent2_joint_plays_performance.append(agent2_joint_plays)
        agent1_global_state_size_performance.append(agent1.global_state_size())
        agent2_global_state_size_performance.append(agent2.global_state_size())
        

    print(agent1.local_action_selector.q_values)
    print(agent2.local_action_selector.q_values)
    print("============================")
    for state in agent1.global_action_selector.q_values.keys():
        print(state, agent1.global_action_selector.q_values[state])         
    print("============================")
    for state in agent1.coordination_joint_states_confidence.keys():
        print(state, agent1.coordination_joint_states_confidence[state])
    print("============================")
    for state in agent2.global_action_selector.q_values.keys():
        print(state, agent2.global_action_selector.q_values[state])         
    print("============================")
    for state in agent2.coordination_joint_states_confidence.keys():
        print(state, agent2.coordination_joint_states_confidence[state])
    print("============================")
    print("============================")
        
    return (agent1_performance, agent2_performance, 
            timesteps_performance, 
            collisions_performance, 
            agent1_joint_plays_performance, agent2_joint_plays_performance,
            agent1_global_state_size_performance, agent2_global_state_size_performance)
        



def playJointEpisode(agent1, agent2, game, agent1_initial_state, agent2_initial_state):
    
    isGameEnded = False
    agent1_s = agent1_initial_state
    agent2_s = agent2_initial_state
    
    #agent1.reset()
    #agent2.reset()

    timesteps = 1
    no_of_collisions = 0
    agent1_joint_plays = 0
    agent2_joint_plays = 0
    
    while(isGameEnded != True): 
        
        agent1_a, agent1_is_joint = agent1.takeAction(agent1_s, (agent1_s, agent2_s))
        agent2_a, agent2_is_joint = agent2.takeAction(agent2_s, (agent2_s, agent1_s))
        
        if (agent1_is_joint == True):
            agent1_joint_plays += 1
        if (agent2_is_joint == True):
            agent2_joint_plays += 1        
        
        agent1_s_, agent1_r, agent2_s_, agent2_r, collisionExists, isGameEnded = game.next_state_joint(agent1_s, agent1_a, agent2_s, agent2_a)
        
        if (collisionExists):
            no_of_collisions += 1                
         
        agent1.enviornmentFeedback(agent1_s, (agent1_s, agent2_s), agent1_a, agent1_s_, agent1_r)
        agent2.enviornmentFeedback(agent2_s, (agent2_s, agent1_s), agent2_a, agent2_s_, agent2_r)
    
        agent1_s = agent1_s_
        agent2_s = agent2_s_
         
        timesteps += 1
        
    return timesteps, no_of_collisions, agent1_joint_plays, agent2_joint_plays
        


def playIndividually(agent, game, initial_state):
    agent_performance = []
    for episode in range(5000): # ideally 10000
        agent_s = initial_state
        timestep = 1
        isGameEnded = False
        while(isGameEnded != True): 
            agent_a, _ = agent.takeAction(agent_s, (agent_s, None))
            agent_s_, agent_r, isGameEnded = game.next_state_single(agent_s, agent_a)
            
            agent.enviornmentFeedback(agent_s, (agent_s, None), agent_a, agent_s_, agent_r)
            agent_s = agent_s_
            timestep += 1
        agent_performance.append(timestep)
    return agent_performance



if __name__ == "__main__":        
    agent1_performance_avg = []
    agent2_performance_avg = [] 
    timesteps_performance_avg = []
    collisions_performance_avg = []
    agent1_joint_plays_performance_avg = []
    agent2_joint_plays_performance_avg = []
    agent1_global_state_size_performance_avg = []
    agent2_global_state_size_performance_avg = []
    
    
    no_of_runs = 1
    for run in range(no_of_runs):
        
        """
        for game2: 
        game1 = games.Game2()
        set no_of_states to 25
        pass initial states 0,20 to playOneRun instead of 6,8
        """
        
        game1 = games.Game1()
        agent1 = cqlearners.CQLearner(name="agent1", no_of_states=9, no_of_actions=4, sliding_window_size=60, epison=0.1, discount_factor=0.9, learning_rate=0.1)
        agent2 = cqlearners.CQLearner(name="agent2", no_of_states=9, no_of_actions=4, sliding_window_size=60, epison=0.1, discount_factor=0.9, learning_rate=0.1)
  
        agent1_performance, agent2_performance, \
            timesteps_performance, collisions_performance, \
                agent1_joint_plays_performance, agent2_joint_plays_performance, \
                    agent1_global_state_size_performance, agent2_global_state_size_performance = playOneRun(game1, agent1, agent2, 6, 8)
                    
        agent1_performance_avg.append(agent1_performance)
        agent2_performance_avg.append(agent2_performance)
        timesteps_performance_avg.append(timesteps_performance)
        collisions_performance_avg.append(collisions_performance)
        agent1_joint_plays_performance_avg.append(agent1_joint_plays_performance)
        agent2_joint_plays_performance_avg.append(agent2_joint_plays_performance)
        agent1_global_state_size_performance_avg.append(agent1_global_state_size_performance)
        agent2_global_state_size_performance_avg.append(agent2_global_state_size_performance)
    
          
    """      
    plt.figure()  
    plt.plot(np.mean(agent1_performance_avg, axis=0), label="agent1")
    plt.plot(np.mean(agent2_performance_avg, axis=0), label="agent2")
    plt.xlabel('episodes')
    plt.ylabel('timesteps needed to reach goal')
    plt.legend()
    """
    plt.figure()
    plt.plot(np.mean(timesteps_performance_avg, axis=0))
    plt.xlabel('episodes')
    plt.ylabel('timesteps needed to reach goal')
    plt.legend()
    
    plt.figure()
    plt.plot(np.mean(collisions_performance_avg, axis=0), label="collisions_performance")
    plt.xlabel('episodes')
    plt.ylabel('no of collisions')
    plt.legend()
    
    plt.figure()
    plt.plot(np.mean(agent1_joint_plays_performance_avg, axis=0), label="agent 1")
    plt.plot(np.mean(agent2_joint_plays_performance_avg, axis=0), label="agent 2")
    plt.xlabel('episodes')
    plt.ylabel('no of joint plays')
    plt.legend()
    
    
    plt.figure()
    plt.plot(np.mean(agent1_global_state_size_performance_avg, axis=0), label="agent 1")
    plt.plot(np.mean(agent2_global_state_size_performance_avg, axis=0), label="agent 2")
    plt.xlabel('episodes')
    plt.ylabel('global state size')
    plt.legend()
