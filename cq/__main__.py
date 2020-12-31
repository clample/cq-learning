from .gridworld import GridWorld
from .runner import Runner
from .independent import IndependentAgent
from .cqlearners import CQLearner
from .plot import Plot

def main(args=None):
    run_basic_grid()
    run_tunnel_to_goal()

def run_tunnel_to_goal():
    gridworld = GridWorld.tunnel_to_goal()
    runner = Runner(2000, 10, gridworld)
    plot = Plot("tunnel")
    agent_creator = AgentCreator((0,0), (0,4))
    independent_results = runner.run(agent_creator.create_independent_agents)
    cq_results = runner.run(agent_creator.create_cq_agents)
    plot.collisions_over_time_plot({
        "Independent": independent_results,
        "CQ": cq_results
    })
    plot.states_over_time_plot({
        "Independent": independent_results,
        "CQ": cq_results
    })
    
def run_basic_grid():
    gridworld = GridWorld.basic_grid()
    runner = Runner(2000, 10, gridworld)
    plot = Plot("basic")
    agent_creator = AgentCreator((0,2), (2,2))
    plot.collisions_over_time_plot({
        "Independent": runner.run(agent_creator.create_independent_agents),
        "CQ": runner.run(agent_creator.create_cq_agents)
    })

class AgentCreator:
    def create_independent_agents(self):
        learning_rate = lambda time_step: 0.1
        return [
            IndependentAgent(learning_rate=learning_rate, epsilon=0.1, discount_factor=0.75, state=self.start_1, name="Agent 1"),
            IndependentAgent(learning_rate=learning_rate, epsilon=0.1, discount_factor=0.75, state=self.start_2, name="Agent 2")
        ]

    def create_cq_agents(self):
        initial_state = {
            "Agent 1": self.start_1,
            "Agent 2": self.start_2
        }
        return [ CQLearner(name, initial_state, discount_factor=0.75) for name in initial_state ]

    def __init__(self, start_1, start_2):
        self.start_1 = start_1
        self.start_2 = start_2
    
if __name__ == "__main__":
    main()
