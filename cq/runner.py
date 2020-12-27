from .experiment_results import ExperimentResults

class Runner:
    
    def __init__(self, num_episodes, num_trials, environment):
        self.num_episodes = num_episodes
        self.num_trials = num_trials
        self.environment = environment

    def run(self, agent_creator):
        """Run the experiment with the given number of trials and episodes"""
        experiment_results = ExperimentResults()
        for trial in range(self.num_trials):
            experiment_results.start_new_trial()
            # New agents are created for each trial, but reused between episodes
            agents = agent_creator()
            for episode in range(self.num_episodes):
                self.__run_episode(experiment_results, agents)
        return experiment_results
        
    def __run_episode(self, experiment_results, agents):
        """Run an episode and record the results using experiment_results
        An episode is considered complete when all agents have reached their goal
        """
        experiment_results.start_new_episode()
        # active_agents tracks which agents haven't yet reached their goal
        # `list` is used to make a shallow copy
        active_agents = list(agents) 
        while len(active_agents) > 0:
            agent_actions = self.__get_agent_actions(active_agents)
            action_results = self.environment.apply_actions(agent_actions)
            experiment_results.record(action_results)
            global_state = { agent_name: result["state"] for (agent_name,result) in action_results.items() }
            for agent in active_agents:
                result = action_results[agent.name]
                reward = self.__calculate_reward(result["wall"], result["collision"], result["goal"])
                agent.update_state(global_state, reward)
            active_agents = list(filter(lambda agent: not action_results[agent.name]["goal"] , active_agents))


    def __get_agent_actions(self, active_agents):
        agent_actions = {}
        for agent in active_agents:
            agent_actions[agent.name] = {
                "state": agent.local_state,
                "action": agent.select_action()
            }
        return agent_actions
            
    def __calculate_reward(self, wall, collision, goal):
        if collision:
            return -50
        elif wall:
            return -10
        elif goal:
            return 1000
        else:
            return 0
            
