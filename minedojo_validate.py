import minedojo
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from config.config import load_config

cfg = load_config("config/default_config.yaml")
#env = minedojo.make(task_id="harvest_milk", image_size=(160, 256))
env = minedojo.make(task_id=cfg.minedojo.task_id, image_size=tuple(cfg.minedojo.minecraft_rgb_shape), world_seed=cfg.minedojo.world_seed, generate_world_type=cfg.minedojo.generate_world_type, specified_biome=cfg.minedojo.specified_biome)

#print("task prompt: ", env.task_prompt)
#print("task guidance: ", env.task_guidance)
fig, ax = plt.subplots()
canvas = fig.canvas  # Get the canvas
plt.show(block=False)  # Add this line to display the window
obs = env.reset()
img = ax.imshow(np.random.randint(0, 256, (160, 256, 3), dtype=np.uint8))
for i in range(100):
    obs_space = env.observation_space
    pos = obs["location_stats"]["pos"][:3]
    print(pos)

    #print("position", obs_space["location_stats"]["pos"])
    #print("img ", obs["rgb"])

    img.set_data(obs["rgb"].transpose(1, 2, 0))
    plt.draw()
    canvas.flush_events()  # Update the window

    action = env.action_space.no_op()

    action[0] = 1
    action[2] = 1
    action[4] = 13

    next_obs, reward, done, info = env.step(action)
    obs = next_obs

env.close()