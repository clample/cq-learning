class ExperimentResults:

    def get_average_collisions(self):
        num_trials = len(self.results)
        num_episodes = len(self.results[-1])
        average_collisions = []
        for episode in range(num_episodes):
            num_collisions = 0
            for trial in range(num_trials):
                for timestep_result in self.results[trial][episode]:
                    for agent, result in timestep_result.items():
                        if result["collision"]:
                            num_collisions += 1
            average_collisions.append(num_collisions / float(num_trials))
        return average_collisions
    
    def start_new_trial(self):
        self.results.append([])

    def start_new_episode(self):
        self.results[-1].append([])

    def record(self, action_results):
        self.results[-1][-1].append(action_results)
    
    def __init__(self):
        self.results = []
