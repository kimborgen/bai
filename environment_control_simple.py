import numpy as np
import random
import time
import minedojo

class EnvironmentControl():
    def __init__(self, cfg):
        self.cfg = cfg
        self.spawn_position = self._generate_random_coords()
        self.previous_positions = []
        self.action_rnd = 15
        
        self.env = minedojo.make(task_id=cfg.minedojo.task_id, image_size=tuple(cfg.minedojo.minecraft_rgb_shape), world_seed=cfg.minedojo.world_seed, generate_world_type=cfg.minedojo.generate_world_type, specified_biome=cfg.minedojo.specified_biome, allow_mob_spawn=False, allow_time_passage=False, initial_weather="clear")
        self.env.reset()
        

    def _generate_random_coords(self):
        x = random.randint(-self.cfg.minedojo.minecraft_world_radius, self.cfg.minedojo.minecraft_world_radius)
        z = random.randint(-self.cfg.minedojo.minecraft_world_radius, self.cfg.minedojo.minecraft_world_radius)
        
        return (x, z)

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

    def step(self):
        action = self.decideAction()
        obs, _, done, _ = self.env.step(action)

        pos = obs["location_stats"]["pos"]

        doEnvReset = False

        if done:
            doEnvReset = True
        elif obs["life_stats"]["life"] == 0:
            print("DEAD")
            doEnvReset = True
        elif self._check_stuck(pos):
            doEnvReset = True
        elif self.iter >= 1200: # we want 1000 but the stuck function works over the last 180 frames, so this ensures that we are not stuck.
            doEnvReset = True  
        if doEnvReset:
            obs = self.reset()
        else:
            self.iter += 1


        if self.iter % 100 == 0 and self.iter != 0:
            self.env.set_weather("clear")

        return obs, doEnvReset, done
    
    def decideAction(self):
        action = self.env.action_space.no_op()
        action[0] = 1
        action[2] = 1
        if self.iter % self.action_rnd == 0 and self.iter != 0:
            rn = random.randint(0,10)
            if rn == 9:
                return action
            elif rn >= 3:
                action[4] = 13
            else: 
                action[4] = 11
        return action

    def reset(self):
        self.spawn_position = self._generate_random_coords()
        self.previous_positions = []
        self.iter = 0
        self.action_rnd = random.randint(15,30)

        action = self.env.action_space.no_op()
        self.env.kill_agent()
        self.env.step(action)
        time.sleep(1)
        mc_cmd = self.generate_tp_command()
        self.env.execute_cmd(mc_cmd)
        time.sleep(1)
        self.env.set_weather("clear")
        # step 10 times to load world
        for i in range(20):
            self.env.step(action)
            time.sleep(0.1)

        obs, _, _, _ = self.env.step(action)

        return obs

    def generate_tp_command(self):
        # Setting the parameters for the command
        spreadDistance = 0
        maxRange = 5
        respectTeams = "false"
        player = "@p"  # The closest player
        # Structuring the command
        command = f"/spreadplayers {self.spawn_position[0]} {self.spawn_position[1]} {spreadDistance} {maxRange} {respectTeams} {player}"
        return command