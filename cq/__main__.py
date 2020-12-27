from .gridworld import GridWorld
from .runner import Runner

def main(args=None):
    gridworld = GridWorld.basic_grid()
    runner = Runner(2000, 2, gridworld)
    

if __name__ == "__main__":
    main()
