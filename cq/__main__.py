from . import learning
from .gridworld import GridWorld

def main(args=None):
    gridworld = GridWorld.basic_grid()
    print(learning.hello())

if __name__ == "__main__":
    main()
