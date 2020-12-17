from enum import Enum, auto
from .action import Action

class GridWorld:

    class GridSquare:
        """Corresponds to a single square on the grid world.
        The square may or may not have walls blocking the agent on its sides.
        """

        def permits_move(action):
            if action == Action.NORTH:
                return not self.wall_north
            elif action == Action.SOUTH:
                return not self.wall_south
            elif action == Action.EAST:
                return not self.wall_east
            elif action == Action.WEST:
                return not self.wall_west
            else:
                raise Exception(f"Unsupported action {action}")

        def __init__(self, wall_north=False, wall_south=False, wall_east=False, wall_west=False):
            self.wall_north = wall_north
            self.wall_south = wall_south
            self.wall_east = wall_east
            self.wall_west = wall_west
            
    # TODO: Handle the goal state
    def apply_action(state, action):
        """Returns the next state and the resulting reward for applying the action"""

        x, y = state
        current_square = self.grid[y][x]
        if current_square.permits_move(action):
            next_state = self.__get_next_state(state, action)
            # The next square is open, so the agent may move to it.
            return (next_state, 0)
        else:
            # The next square is blocked. The agent remains in place and receives a penalty.
            return (state, -100)

    def __get_next_state(state, action):
        x,y = state
        if action == Action.NORTH:
            return (x, y+1)
        elif action == Action.SOUTH:
            return (x, y-1)
        elif action == Action.EAST:
            return (x+1, y)
        elif action == Action.WEST:
            return (x-1, y)
        else:
            raise Exception(f"Unsupported action {action}")
    
    def __init__(self, grid, goal_state):
        self.grid = grid
        self.goal_state = goal_state


    @classmethod
    def basic_grid(cls):
        return GridWorld([
            [ cls.GridSquare(wall_north=True, wall_west=True),                  cls.GridSquare(wall_north=True), cls.GridSquare(wall_north=True, wall_east=True)                  ],
            [ cls.GridSquare(wall_south=True, wall_west=True),                  cls.GridSquare(),                cls.GridSquare(wall_south=True, wall_east=True)                  ],
            [ cls.GridSquare(wall_north=True, wall_south=True, wall_west=True), cls.GridSquare(wall_south=True), cls.GridSquare(wall_north=True, wall_south=True, wall_east=True) ]
        ], (1,0))
