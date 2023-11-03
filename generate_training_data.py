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
import torch.nn.functional as torchF
from model import TopologyNet
from config.config import load_config
import random
import math
from matplotlib.animation import FuncAnimation
import os

from image_processing import rgb_to_gray, convert_gray_to_event
from coords_processing import coords_to_clifford, coords_to_rad, coords_to_rad_scaling_factor, rad_to_coords, spike_history_to_clifford, spikes_to_clifford

def plot_images(cfg, plot_queue):
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    axs[0].set_title('Latest grayscale frame (for viz only)' if cfg.debug.show_images_in_real_time else "Grayscale frame")
    axs[1].set_title('DVS Frame')

    axs[2].set_title('Local Coordinates over Time')
    axs[2].set_xlabel('X coordinate')
    axs[2].set_ylabel('Z coordinate')
    axs[2].legend()
    axs[2].set_xlim(-200,200)
    axs[2].set_ylim(-200,200)

    axs[3].set_title('Previous loss')

    def update(frame):
        if not plot_queue.empty():
            plot_data = plot_queue.get()

            if plot_data == "poison_pill":
                return
            
            gray_frame, event_frame = plot_data['images']
            coordinates = plot_data['coordinates']
            doClear = plot_data['reset']
            loss = plot_data['loss']

            axs[0].imshow(gray_frame, cmap='gray')
            axs[1].imshow(event_frame, cmap='gray', vmin=0, vmax=2)

            if coordinates:
                axs[2].plot(coordinates[0][0], coordinates[0][1], marker="o", markersize=1, markerfacecolor='red', markeredgecolor='red', color='blue', label='Ground Truth')
                axs[2].plot(coordinates[1][0], coordinates[1][1], marker="o", markersize=1, markerfacecolor='blue', markeredgecolor='blue', color='blue', label='Predicted')

            if doClear:
                axs[2].clear()  # Clear previous plot to avoid overlaying plots
                axs[2].set_xlim(-200,200)
                axs[2].set_ylim(-200,200)

            if loss:
                axs[3].plot(loss.clone().detach().cpu().numpy())
                

    ani = FuncAnimation(fig, update, repeat=False)
    plt.show()

def extract_obs(obs):
    rgb_frame = obs["rgb"]
    pos = obs["location_stats"]["pos"]
    yaw = obs["location_stats"]["yaw"]
    pitch = obs["location_stats"]["pitch"]
    return rgb_frame, pos, yaw, pitch

def gameSetup(cfg):
    env = minedojo.make(task_id=cfg.minedojo.task_id, image_size=tuple(cfg.minedojo.minecraft_rgb_shape), world_seed=cfg.minedojo.world_seed, generate_world_type=cfg.minedojo.generate_world_type, specified_biome=cfg.minedojo.specified_biome)
    envControl = EnvironmentControl(cfg)

    env.reset()
    action = env.action_space.no_op()
    env.step(action)
    mc_cmd = envControl.generate_tp_command()
    env.execute_cmd(mc_cmd)
    obs, _, _, _ = env.step(action)

    prev_gray_frame = rgb_to_gray(cfg, np.zeros((3, 160, 256), dtype=np.uint8))

    return env, envControl, obs, prev_gray_frame

def observeAndPreproccess(cfg, obs, prev_gray_frame, goals):

    # extract variables from enviroment
    rgb_frame, pos, yaw, pitch = extract_obs(obs)
    
    # generate the event_frame
    gray_frame = rgb_to_gray(cfg, rgb_frame)
    event_frame = convert_gray_to_event(cfg, prev_gray_frame, gray_frame)
    modelInputs = EnvironmentObservation(cfg, event_frame, pos, yaw, pitch, goals)
    return modelInputs, gray_frame, event_frame

def decideAction(cfg, env, env_step):
    action = env.action_space.no_op()
    action[0] = 1
    action[2] = 1
    if env_step % 15 == 0:
        rn = random.randint(0,3)
        if rn < 2:
            action[4] = 13
        else: 
            action[4] = 11
    return action

def rl_env_step(cfg, env, envControl, obs, done):
    doEnvReset = False
    reward_str = ""
    newPos = obs["location_stats"]["pos"]
    if done:
        return
    elif obs["life_stats"]["life"] == 0:
        print("DEAD")
        envControl.reset()
        doEnvReset = True
        reward_str = "DEAD"
    else:
        envControlRes = envControl.step(newPos)
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
        action = env.action_space.no_op()
        env.kill_agent()
        env.step(action)
        mc_cmd = envControl.generate_tp_command()
        env.execute_cmd(mc_cmd)
        obs, _, _, _ = env.step(action)
    return obs, reward_str, doEnvReset

def topology_net_train(cfg, tNet, target, optimizer):
    params = cfg.topology_net.pos_xz
    #spks = torch.randint(0, 2, size=(target.shape[0], 200)).to(cfg.device)
    spks = tNet.out_spks

    clifford_coords = spikes_to_clifford(cfg, spks)
    loss = torchF.mse_loss(clifford_coords, target)
    print(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

    """
    # initialize the loss & sum over time
    loss_val = loss_fn(spk_rec, targets)

    # Gradient calculation + weight update
    optimizer.zero_grad()
    loss_val.backward()
    optimizer.step()

    # Store loss history for future plotting
    loss_hist.append(loss_val.item())
    """

def gameLoop(cfg, plot_queue):
    cfg.print(cfg)
    env, envControl, obs, prev_gray_frame = gameSetup(cfg)
    #snet = SNet(cfg).to(cfg.model_params.device)

    tNet = TopologyNet(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(tNet.parameters(), lr=cfg.topology_net.lr, betas=(0.9, 0.999))
    loss_hist = []
    test_acc_hist = []

    file_paths = [f for f in os.listdir('data/localization') if f.endswith('.pt')]

    env_iteration = len(file_paths)

    td_gray_frames = torch.empty((0,80,128), device=cfg.device)
    td_event_frames = torch.empty((0,80,128), device=cfg.device)
    td_local_coords = torch.empty((0,2), device=cfg.device)
    td_clifford_coords = torch.empty((0,4), device=cfg.device) 

    target_clifford = torch.empty((0,4), dtype=torch.float32).to(cfg.device)
    max_loops = 1000000000
    pbar = tqdm(total=max_loops, desc="Processing", dynamic_ncols=True)
    for i in range(max_loops):
        start_time = time.time()

        modelInputs, prev_gray_frame, event_frame = observeAndPreproccess(cfg, obs, prev_gray_frame, envControl.goals)
        #print(modelInputs)

        # model forward pass to determine action
        #if cfg.topology_net.train:
        #    tNet.train()
        #out_spk = tNet(modelInputs.event_frame)
        #out_spks = tNet.out_spks

        action = decideAction(cfg, env, i)

        obs, _, done, _ = env.step(action)
        obs, reward_str, didEnvReset = rl_env_step(cfg, env, envControl, obs, done)
        #if i % 1000 == 0 and i != 0:
        #    didEnvReset = True
        #    reward_str = "POSITIVE"

        # training_data_gen:)

        coords = torch.tensor([envControl.local_x, envControl.local_z], dtype=torch.float32, device=cfg.device)

        cliff = coords_to_clifford(cfg, coords)

        # training data gen
        td_gray_frames = torch.cat((td_gray_frames, torch.from_numpy(prev_gray_frame).unsqueeze(0).to(cfg.device)), dim=0)
        td_event_frames = torch.cat((td_event_frames, torch.from_numpy(event_frame).unsqueeze(0).to(cfg.device)), dim=0)
        td_local_coords = torch.cat((td_local_coords, torch.tensor((envControl.local_x, envControl.local_z)).unsqueeze(0).to(cfg.device)), dim=0)
        td_clifford_coords = torch.cat((td_clifford_coords, cliff.unsqueeze(0)), dim=0)

        # clean up
        # remember to reset model memories on reset

        loss = None
        #if didEnvReset:
        #     # model training
        #    if cfg.topology_net.train and target_clifford.shape[0] > 1:
        #        loss = topology_net_train(cfg, tNet, target_clifford, optimizer)

        #        target_clifford = torch.empty((0,4), dtype=torch.float32).to(cfg.device)
        #        tNet.reset()

        if cfg.debug.show_images and (didEnvReset or plot_queue.empty()):
            plot_data = {
                "images": (prev_gray_frame, event_frame),
                "coordinates":((envControl.local_x, envControl.local_z), (0,0)),
                "reset": didEnvReset,
                "loss": loss
            }
            plot_queue.put(plot_data)

        if didEnvReset:
            # Example of writing each epoch to disk
            file_path = f"data/localization/data_d_{env_iteration}.pt"
            torch.save({
                'td_gray_frames': td_gray_frames,
                'td_event_frames': td_event_frames,
                'td_local_coords': td_local_coords,
                'td_clifford_coords': td_clifford_coords
            }, file_path)

            td_gray_frames = torch.empty((0,80,128), device=cfg.device)
            td_event_frames = torch.empty((0,80,128), device=cfg.device)
            td_local_coords = torch.empty((0,2), device=cfg.device)
            td_clifford_coords = torch.empty((0,4), device=cfg.device) 

            env_iteration += 1

            envControl.reset()


        elapsed_time = time.time() - start_time
        framerate = 1 / elapsed_time if elapsed_time > 0 else float('inf')
        pbar.set_description(f"FPS: {framerate:.2f}")
        pbar.update(1)

        if done:
            print("done!")
            break

    plot_data = "poison_pill"
    plot_queue.put(plot_data)
    env.close()

def main():
    plot_queue = queue.Queue()
    cfg = load_config("config/default_config.yaml")
    assert cfg.debug.show_images_in_real_time == True
    # Start the worker thread for gameLoop
    worker_thread = threading.Thread(target=gameLoop, args=(cfg, plot_queue))
    worker_thread.start()
    
    # The main thread handles GUI updates
    if cfg.debug.show_images:
        plot_images(cfg, plot_queue)

    worker_thread.join()


if __name__ == "__main__":
    main()