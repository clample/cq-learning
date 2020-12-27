from .experiment_results import ExperimentResults

class Runner:

    def __init__(self, num_episodes, num_trials, environment):
        """
        An episode is completed when all agents are in the goal state.

        To give more reliable results, the experiment is run multiple times (specified by `num_trials`)
        The results are then averaged.
        """
        self.num_episodes = num_episodes
        self.num_trials = num_trials
        self.environment = environment

    # TODO: How to handle agent_creator?
    def run_trial(self, experiment_results, agent_creator):
        """ Run a trial and record the results using experiment_results
        
        agent_creator is a function which creates a fresh set of agents
        """
        experiment_results.start_new_trial()
        for episode in range(num_episodes):
            self.run_episode(experiment_results, agent_creator())

    def run_episode(self, experiment_results, agents):
        """Run an episode and record the results using experiment_results
        An episode is considered complete when all agents have reached their goal
        """
        experiment_results.start_new_episode()
        # active_agents tracks which agents haven't yet reached their goal
        # `list` is used to make a shallow copy
        active_agents = list(agents) 
        while len(active_agents) > 0:
            agent_actions = self.get_agent_actions()
            action_results = self.environment.apply_actions(agent_actions)
            experiment_results.record(action_results)
            for agent in active_agents:
                result = action_results[agent.name]
                reward = self.calculate_reward(result["wall"], result["collision"], result["goal"])
                agent.update_state(result["state"], reward)
            active_agents = list(filter(lambda agent: not action_result[agent.name]["goal"] , active_agents))


    def get_agent_actions(self, active_agents):
        agent_actions = {}
        for agent in active_agents:
            agent_actions[agent.name] = {
                "state": agent.state,
                "action": agent.select_action()
            }
            
    def calculate_reward(self, wall, collision, goal):
        if collision:
            return -50
        elif wall:
            return -10
        elif goal:
            return 1000
            
