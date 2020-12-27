from .gridworld import GridWorld
from .runner import Runner
from .independent import IndependentAgent
from .cqlearners import CQLearner
from .plot import Plot

def main(args=None):
    gridworld = GridWorld.basic_grid()
    runner = Runner(2000, 30, gridworld)
    plot = Plot()
    plot.collisions_over_time_plot({
        "Independent": runner.run(create_independent_agents),
        "CQ": runner.run(create_cq_agents)
    })

def create_independent_agents():
    learning_rate = lambda time_step: 0.1
    return [
        IndependentAgent(learning_rate=learning_rate, epsilon=0.1, discount_factor=0.9, state=(0,2), name="Agent 1"),
        IndependentAgent(learning_rate=learning_rate, epsilon=0.1, discount_factor=0.9, state=(2,2), name="Agent 2")
    ]

def create_cq_agents():
    initial_state = {
        "Agent 1": (0,2),
        "Agent 2": (2,2)
    }
    return [ CQLearner(name, initial_state) for name in initial_state ]

if __name__ == "__main__":
    main()
