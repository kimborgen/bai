import cv2
import minedojo
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import time
import threading
import queue
from tqdm import tqdm
from environment_observation import EnvironmentObservation
from environment_control import EnvironmentControl
import torch
from model import TopologyNet
from config.config import load_config
import random

def rgb_to_gray(cfg, rgb_img):
    frame = cv2.cvtColor(rgb_img.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (frame.shape[1] // cfg.preprocess_vision.downscale_factor, frame.shape[0] // cfg.preprocess_vision.downscale_factor), interpolation=cv2.INTER_AREA)
    return frame

"""
def convert_gray_to_event(cfg, prev_frame, frame):
    difference = frame.astype(np.int16) - prev_frame.astype(np.int16)
    conditions = [difference > cfg.preprocess_vision.intensity_treshold, difference < -cfg.preprocess_vision.intensity_treshold]
    choices = [2, 0]
    combined_events = np.select(conditions, choices, default=1)
    return combined_events
"""

def convert_gray_to_event(cfg, prev_frame, frame):
    difference = frame.astype(np.int16) - prev_frame.astype(np.int16)
    # Get absolute difference
    abs_difference = np.abs(difference)
    # Check if absolute difference is greater than or equal to the threshold
    combined_events = (abs_difference >= cfg.preprocess_vision.intensity_treshold).astype(int)
    return combined_events

def plot_images(cfg, image_queue):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    canvas = fig.canvas
    plt.show(block=False)
    
    while True:
        while cfg.debug.show_images_in_real_time and image_queue.qsize() > 1:
            image_queue.get()

        gray_frame, event_frame = image_queue.get()
        
        if gray_frame is None or event_frame is None:
            break
        
        axs[0].imshow(gray_frame, cmap='gray')
        axs[0].set_title('Grayscale frame')
        
        axs[1].imshow(event_frame, cmap='gray', vmin=0, vmax=2)
        axs[1].set_title('DVS Frame')
        
        plt.draw()
        canvas.flush_events()

def extract_obs(obs):
    rgb_frame = obs["rgb"]
    pos = obs["location_stats"]["pos"]
    yaw = obs["location_stats"]["yaw"]
    pitch = obs["location_stats"]["pitch"]
    return rgb_frame, pos, yaw, pitch

def gameLoop(cfg, image_queue):
    cfg.print(cfg)
    env = minedojo.make(task_id=cfg.minedojo.task_id, image_size=tuple(cfg.minedojo.minecraft_rgb_shape), world_seed=cfg.minedojo.world_seed, generate_world_type=cfg.minedojo.generate_world_type, specified_biome=cfg.minedojo.specified_biome)
    envControl = EnvironmentControl(cfg)

    obs = env.reset()
    mc_cmd = envControl.generate_tp_command()
    obs, _, _, _ = env.execute_cmd(mc_cmd)
    
    #snet = SNet(cfg).to(cfg.model_params.device)

    prev_gray_frame = rgb_to_gray(cfg, np.zeros((3, 160, 256), dtype=np.uint8))

    max_loops = 10000
    pbar = tqdm(total=max_loops, desc="Processing", dynamic_ncols=True)
    for i in range(max_loops):
        start_time = time.time()

        # extract variables from enviroment
        rgb_frame, pos, yaw, pitch = extract_obs(obs)
        
        # generate the event_frame
        gray_frame = rgb_to_gray(cfg, rgb_frame)
        event_frame = convert_gray_to_event(cfg, prev_gray_frame, gray_frame)
        if cfg.debug.show_images:
            image_queue.put((gray_frame, event_frame))
        
        print(envControl.goals)
        modelInputs = EnvironmentObservation(cfg, event_frame, pos, yaw, pitch, envControl.goals)
        
        print(modelInputs)

        action = env.action_space.no_op()
        
        action[0] = 1
        action[2] = 1
        if i % 15 == 0:
            rn = random.randint(0,3)
            if rn < 2:
                action[4] = 13
            else: 
                action[4] = 11

        #next_obs, reward, done, info = env.step(action)
        #print(reward, done, info)
        obs, _, done, _ = env.step(action)
        # check if dead
        doEnvReset = False
        reward_str = ""
        if obs["life_stats"]["life"] == 0:
            print("DEAD")
            envControl.reset()
            doEnvReset = True
            reward_str = "DEAD"
        else:
            envControlRes = envControl.step(pos)
            match envControlRes:
                case "STUCK":
                    doEnvReset = True
                    print("STUCK")
                case "TIMEOUT":
                    doEnvReset = True
                    print("TIMEOUT")
                case "MESSI":
                    doEnvReset = True
                    print("MESSI")
                case "GOAL":
                    doEnvReset = True
                    print("GOAL")
                case "NOMINAL":
                    pass
            reward_str = envControlRes

        if doEnvReset:
            env.reset()
            mc_cmd = envControl.generate_tp_command()
            print(envControl.goals)
            obs, _, _, _ = env.execute_cmd(mc_cmd)

        prev_gray_frame = gray_frame
        
        elapsed_time = time.time() - start_time
        framerate = 1 / elapsed_time if elapsed_time > 0 else float('inf')
        pbar.set_description(f"FPS: {framerate:.2f}")
        pbar.update(1)

        if done:
            print("done!")
            break

    if cfg.debug.show_images:
        image_queue.put((None, None))  # Signal the plotting thread to terminate

    env.close()

def main():
    image_queue = queue.Queue()
    cfg = load_config("config/default_config.yaml")
    
    # Start the worker thread for gameLoop
    worker_thread = threading.Thread(target=gameLoop, args=(cfg, image_queue,))
    worker_thread.start()
    
    # The main thread handles GUI updates
    if cfg.debug.show_images:
        plot_images(cfg, image_queue)

    worker_thread.join()


if __name__ == "__main__":
    main()