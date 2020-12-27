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

    def apply_actions(agent_actions):
        """Applies all of the actions of the agents at the same time.
        Penalties will be given if the agents collide or hit a wall.
        """
        result = {}
        # Find the next states and check if the agents hit a wall
        for agent in agent_actions:
            state = agent_actions[agent]["state"]
            action = agent_actions[agent]["action"]
            current_square = self.__get_square(state)
            if current_square.permits_move(action):
                result[agent] = { "state": self.__get_next_state(state, action), "wall": False }
            else:
                result[agent] = { "state": state, "wall": True }

        # Check if the agents collide
        for agent in agent_actions:
            state = result[agent]["state"]
            for other_agent in agent_actions:
                if agent == other_agent:
                    continue
                other_state = result[agent]["state"]
                if state == other_state:
                    result[agent][state] = agent_actions[agent]["state"]
                    result[agent]["collision"] = True
        return result

    def __get_next_state(state, action):
        """Find the next state by applying the given action.
        Note that this method doesn't check if the action is actually possible.
        There might be a wall, or the agents might collide.
        These conditions should be checked by the caller.
        """
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

    def __get_square(state):
        x,y = state
        return self.grid[y][x]
    
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
