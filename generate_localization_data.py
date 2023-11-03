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
from environment_control_simple import EnvironmentControl
import torch
import torch.nn.functional as torchF
from model import TopologyNet
from config.config import load_config
import random
import math
from matplotlib.animation import FuncAnimation
import os

from image_processing import rgb_to_gray, convert_gray_to_event
from coords_processing import coord_to_clifford, coords_to_rad, coords_to_rad_scaling_factor, rad_to_coords, spike_history_to_clifford, spikes_to_clifford

def extract_obs(obs):
    rgb_frame = obs["rgb"]
    pos = obs["location_stats"]["pos"]
    yaw = obs["location_stats"]["yaw"]
    pitch = obs["location_stats"]["pitch"]
    return rgb_frame, pos, yaw, pitch

def get_gray_frame(cfg, obs):

    # extract variables from enviroment
    rgb_frame, pos, yaw, pitch = extract_obs(obs)
    
    # generate the event_frame
    gray_frame = rgb_to_gray(cfg, rgb_frame)
    return gray_frame


def gameLoop(cfg, thread_name):
    envControl = EnvironmentControl(cfg)
    obs = envControl.reset()

    file_paths = [f for f in os.listdir('data/localization') if f.endswith('.pt')]
    file_iteration = len(file_paths)

    gray_frames = torch.empty((0,80,128), device=cfg.device)
    coords = torch.empty((0,2), device=cfg.device)

    max_loops = 1000000000
    pbar = tqdm(total=max_loops, desc="Processing", dynamic_ncols=True)
    for i in range(max_loops):
        start_time = time.time()

        extracted_obs = extract_obs(obs)

        # get gray frame
        gray_frame = torch.tensor(get_gray_frame(cfg, obs), dtype=torch.float32, device=cfg.device)
        gray_frames = torch.cat((gray_frames, gray_frame.unsqueeze(0)), dim=0)

        # get coords x, z from x,y,z
        coord = (obs["location_stats"]["pos"][0], obs["location_stats"]["pos"][2])
        coord = torch.tensor(coord, dtype=torch.float32, device=cfg.device)  # or any other desired dtype
        coords = torch.cat((coords, coord.unsqueeze(0)), dim=0)

        # Take a step for the next iteration:
        obs, didEnvReset, done = envControl.step()

        if didEnvReset:

            # normalize coords, so that they start on zero
            td_local_coords = coords - coords[0]

            # check that we collected enough iters
            if coords.shape[0] >= 1201:
                if coords.shape[0] != 1201:
                    raise Exception("yo")
                file_path = f"data/localization/data_{thread_name}_{file_iteration}.pt"
                torch.save({
                    'td_gray_frames': gray_frames[:1001],
                    'td_local_coords': td_local_coords[:1001],
                }, file_path)
                file_iteration += 1


            gray_frames = torch.empty((0,80,128), device=cfg.device)
            coords = torch.empty((0,2), device=cfg.device)


        elapsed_time = time.time() - start_time
        framerate = 1 / elapsed_time if elapsed_time > 0 else float('inf')
        pbar.set_description(f"FPS: {framerate:.2f}")
        pbar.update(1)

        if done:
            print("done!")
            break

    envControl.env.close()

def main():

    cfg = load_config("config/default_config.yaml")
    cfg.print(cfg) 

    threads = []
    for i in range(4):
        thread = threading.Thread(target=gameLoop, args=(cfg, i))
        threads.append(thread)
        thread.start()
        time.sleep(5)

    # Join threads to wait for all to complete
    for thread in threads:
        thread.join() 
 


if __name__ == "__main__":
    main()