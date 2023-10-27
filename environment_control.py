import numpy as np
import random
import time

class EnvironmentControl():
    def __init__(self, cfg):
        self.cfg = cfg
        self.spawn_position = self._generate_random_coords()
        self.local_x = 0  # Local x-coordinate initialized to 0
        self.local_z = 0  # Local z-coordinate initialized to 0
        self.goals = self._generate_goals_list()
        self.previous_positions = []
        self.goals_reached = 0
        self.iter = 0
        self.env_iter = 0

    def _generate_random_coords(self):
        x = random.randint(-self.cfg.minedojo.minecraft_world_radius, self.cfg.minedojo.minecraft_world_radius)
        z = random.randint(-self.cfg.minedojo.minecraft_world_radius, self.cfg.minedojo.minecraft_world_radius)
        return (x, z)

    def _generate_goals_list(self):
        goals = []
        current_goal = (0, 0)  # Start at local (0, 0)
        for i in range(3):
            next_goal_distance = random.randint(0, self.cfg.env_control.max_next_goal_distance)
            next_goal = (current_goal[0] + next_goal_distance, current_goal[1] + next_goal_distance)
            goals.append(next_goal)
            current_goal = next_goal
        return goals

    def _generate_next_goal(self):
        next_goal_distance = random.randint(0, self.cfg.env_control.max_next_goal_distance)
        last_goal = self.goals[-1]
        next_goal = (last_goal[0] + next_goal_distance, last_goal[1] + next_goal_distance)
        return next_goal

    def _is_goal_reached(self, pos):
        # Only looking at X and Z coordinates
        pos_xz = np.array([self.local_x, self.local_z])
        if np.linalg.norm(pos_xz - np.array(self.goals[0])) < 1:
            return True
        return False

    def _check_stuck(self, pos):
        self.previous_positions.append(pos)  # Using all three coordinates now: x, y, z
        
        # Check using the last 180 positions
        if len(self.previous_positions) >= self.cfg.env_control.small_stuck_treshold_items:
            # Making positions relative to the first position in the list
            relative_positions = np.array(self.previous_positions[-self.cfg.env_control.small_stuck_treshold_items:]) - self.previous_positions[-self.cfg.env_control.small_stuck_treshold_items]

            # Calculating variance of the relative positions
            variance = np.var(relative_positions, axis=0)

            # Check variance against a threshold
            if all(variance < self.cfg.env_control.small_stuck_treshold):
                print("Small variance hit", variance)
                return True 
            else:
                return False
            

        # Remove oldest position if the list grows too long
        if len(self.previous_positions) > self.cfg.env_control.stuck_treshold_items:
            self.previous_positions.pop(0)

            # Making positions relative to the first position in the list
            relative_positions = np.array(self.previous_positions) - self.previous_positions[0]

            # Calculating variance of the relative positions
            variance = np.var(relative_positions, axis=0)
            
            if all(variance < self.cfg.env_control.stuck_threshold):
                print("Hit stuck variance ", variance)
                return True
            else:
                return False

     
        return False



    def step(self, pos):
        self.local_x = pos[0] - self.spawn_position[0]
        self.local_z = pos[2] - self.spawn_position[1]


        if self._check_stuck(pos):
            self.reset()
            return "STUCK"
        
        # if the goal is not reached within 1 minute * 60 fps (3600 steps) then its also stuck  
        #if self.iter > 3600:
        #    self.reset()
        #    return "TIMEOUT"
        

        if self._is_goal_reached(pos):
            self.goals_reached += 1
            self.iter = 0
            if self.goals_reached > self.cfg.env_control.max_goals:
                self.reset()
                return "MESSI"

            self.goals.pop(0)  # Remove reached goal
            self.goals.append(self._generate_next_goal())  # Add new goal to existing list
            self.previous_positions = []
            return "GOAL"
        
        self.iter += 1
        return "NOMINAL"

    def reset(self):
        self.goals_reached = 0
        self.spawn_position = self._generate_random_coords()
        self.goals = self._generate_goals_list()
        self.previous_positions = []
        self.env_iter += 1
        self.iter = 0

    def generate_tp_command(self):
        # Setting the parameters for the command
        spreadDistance = 0
        maxRange = 5
        respectTeams = "false"
        player = "@p"  # The closest player
        # Structuring the command
        command = f"/spreadplayers {self.spawn_position[0]} {self.spawn_position[1]} {spreadDistance} {maxRange} {respectTeams} {player}"
        return command