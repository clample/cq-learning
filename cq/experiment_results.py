import numpy

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

    def get_averaged_num_states(self):
        average_num_states = {}
        for agent_name, state_record in self.num_states.items():
            average = numpy.mean(numpy.array(state_record), axis=0)
            average_num_states[agent_name] = average
        return average_num_states            
    
    def start_new_trial(self):
        self.results.append([])
        for agent_num_states in self.num_states.values():
            agent_num_states.append([])
        
    def start_new_episode(self):
        self.results[-1].append([])

    def record(self, action_results):
        self.results[-1][-1].append(action_results)

    def record_states(self, agent_name, num_states):
        self.num_states[agent_name][-1].append(num_states)
    
    def __init__(self, agent_names):
        self.results = []
        self.num_states = {}
        for agent_name in agent_names:
            self.num_states[agent_name] = []
