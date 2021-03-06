from enum import Enum, auto
from .action import Action

class GridWorld:

    class GridSquare:
        """Corresponds to a single square on the grid world.
        The square may or may not have walls blocking the agent on its sides.
        """

        def permits_move(self, action):
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

    def apply_actions(self, agent_actions):
        """Applies all of the actions of the agents at the same time.
        Returns if there is a collision, the agents hit a wall, or the agents reach the goal.
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

        # Check if the agents are in the goal state
        for agent in agent_actions:
            state = result[agent]["state"]
            result[agent]["goal"] = state == self.goal_state

                
        # Check if the agents collide
        for agent in agent_actions:
            state = result[agent]["state"]
            result[agent]["collision"] = False
            for other_agent in agent_actions:
                if agent == other_agent:
                    continue
                other_state = result[other_agent]["state"]
                # We don't consider it a collision if both agents end in the goal
                if state == other_state and state != self.goal_state:                    
                    result[agent]["collision"] = True

        # Reset the states of the collided agents
        for agent in agent_actions:
            if result[agent]["collision"]:
                result[agent]["state"] = agent_actions[agent]["state"]
        
        return result

    def __get_next_state(self, state, action):
        """Find the next state by applying the given action.
        Note that this method doesn't check if the action is actually possible.
        There might be a wall, or the agents might collide.
        These conditions should be checked by the caller.
        """
        x,y = state
        if action == Action.NORTH:
            return (x, y-1)
        elif action == Action.SOUTH:
            return (x, y+1)
        elif action == Action.EAST:
            return (x+1, y)
        elif action == Action.WEST:
            return (x-1, y)
        else:
            raise Exception(f"Unsupported action {action}")

    def __get_square(self, state):
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

    @classmethod
    def tunnel_to_goal(cls):
        return GridWorld([
            [ cls.GridSquare(wall_north=True, wall_west=True), cls.GridSquare(wall_north=True), cls.GridSquare(wall_north=True), cls.GridSquare(wall_north=True), cls.GridSquare(wall_north=True, wall_east=True)],
            [ cls.GridSquare(wall_south=True, wall_west=True), cls.GridSquare(wall_south=True), cls.GridSquare(wall_south=True), cls.GridSquare(wall_south=True), cls.GridSquare(wall_east=True)                 ],
            [ cls.GridSquare(wall_north=True, wall_south=True, wall_west=True), cls.GridSquare(wall_north=True, wall_south=True), cls.GridSquare(wall_north=True, wall_south=True), cls.GridSquare(wall_north=True, wall_south=True), cls.GridSquare(wall_east=True) ],
            [ cls.GridSquare(wall_north=True, wall_west=True), cls.GridSquare(wall_north=True), cls.GridSquare(wall_north=True), cls.GridSquare(wall_north=True), cls.GridSquare(wall_east=True)],
            [ cls.GridSquare(wall_south=True, wall_west=True), cls.GridSquare(wall_south=True), cls.GridSquare(wall_south=True), cls.GridSquare(wall_south=True), cls.GridSquare(wall_south=True, wall_east=True)]
        ], (0, 2))
