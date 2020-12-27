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
        
    def run(self, agents):
        experiment_results = ExperimentResults()
        for trial in range(num_trials):
            experiment_results.start_new_trial()
            for episode in range(num_episodes):
                agent_actions = {}
                for agent in agents:
                    agent_actions[agent.name] = {
                        "state": agent.state,
                        "action": agent.select_action()
                    }
                action_results = self.environment.apply_actions(agent_actions)
                experiment_results.record(action_results)
                for agent in agents:
                    result = action_results[agent.name]
                    reward = 0 # TODO: Calculate reward
                    agent.update_state(result["state"], reward)
                # TODO: End episode when all agents are in the goal
