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
        results = ExperimentResults()
        for trial in range(num_trials):
            results.start_new_trial()
            for episode in range(num_episodes):
