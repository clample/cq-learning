from .gridworld import GridWorld
from .runner import Runner
from .independent import IndependentAgent

def main(args=None):
    gridworld = GridWorld.basic_grid()
    runner = Runner(2000, 2, gridworld)
    results_independent = runner.run(create_independent_agents)

def create_independent_agents():
    learning_rate = lambda time_step: 0.1
    return [
        IndependentAgent(learning_rate=learning_rate, epsilon=0.1, discount_factor=0.7, state=(0,2), name="Agent 1")
    ]
    
if __name__ == "__main__":
    main()
