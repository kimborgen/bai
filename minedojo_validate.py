import minedojo
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
env = minedojo.make(task_id="harvest_milk", image_size=(160, 256))

print("task prompt: ", env.task_prompt)
print("task guidance: ", env.task_guidance)
fig, ax = plt.subplots()
canvas = fig.canvas  # Get the canvas
plt.show(block=False)  # Add this line to display the window
obs = env.reset()
img = ax.imshow(np.random.randint(0, 256, (160, 256, 3), dtype=np.uint8))
for i in range(100):
    obs_space = env.observation_space
    #print("position", obs_space["location_stats"]["pos"])
    #print("img ", obs["rgb"])

    img.set_data(obs["rgb"].transpose(1, 2, 0))
    plt.draw()
    canvas.flush_events()  # Update the window

    action = env.action_space.no_op()

    action[0] = 1
    action[2] = 1

    next_obs, reward, done, info = env.step(action)
    obs = next_obs

env.close()