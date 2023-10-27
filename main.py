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

from matplotlib.animation import FuncAnimation

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

def spike_history_to_clifford(cfg, spks):
    # Create a geometric sequence of weights
    log_start = torch.tensor([1e-3], device=cfg.device).log()
    log_end = torch.tensor([1.0], device=cfg.device).log()
    log_steps = torch.linspace(0, 1, steps=cfg.topology_net.pos_xz.sliding_window, device=cfg.device)
    log_space = log_start * (1 - log_steps) + log_end * log_steps
    weights = torch.exp(log_space)
    weights /= weights.sum()  # Normalize to sum to 1

    # ... rest of your code ...

    # Compute the weighted average along the 0-th dimension (over the iterations)
    # Expand dimensions of weights to match the dimensions of window_spks
    weights_expanded = weights.unsqueeze(1)
    averaged_spks = torch.sum(spks * weights_expanded, dim=0)

    # Scale averaged_spks from [0, 1] to [-1, 1]
    scaled_spks = averaged_spks * 2 - 1
    return scaled_spks

def xz_to_clifford(cfg, x,z):
    x = torch.tensor([x], dtype=torch.float32)
    z = torch.tensor([z], dtype=torch.float32)

    cos_theta = torch.cos(x)
    sin_theta = torch.sin(x)
    cos_phi = torch.cos(z)
    sin_phi = torch.sin(z)

    result = torch.tensor([cos_theta, sin_theta, cos_phi, sin_phi], device=cfg.device)
    return result


def spikes_to_clifford(cfg, spikes_tensor):
    # Assume spikes_tensor is of shape (iter, 200)
    # Reshape the tensor to shape (iter, 4, 50)
    reshaped_tensor = spikes_tensor.view(-1, 4, cfg.topology_net.pos_xz.pop_code)

    # Generate neuron values tensor of shape (50)
    neuron_values = torch.linspace(-1, 1, steps=cfg.topology_net.pos_xz.pop_code, device=cfg.device)

    # Expand the dimensions of neuron_values to match the dimensions of reshaped_tensor
    # New shape of neuron_values: (1, 1, 50)
    neuron_values_expanded = neuron_values.unsqueeze(0).unsqueeze(0)

    # Compute the weighted sum along the last dimension
    # Shape of weighted_sum: (iter, 4)
    weighted_sum = torch.sum(reshaped_tensor * neuron_values_expanded, dim=2)

    # Compute the number of active neurons along the last dimension
    # Shape of num_active_neurons: (iter, 4)
    num_active_neurons = torch.sum(reshaped_tensor, dim=2) + 1e-10

    # Compute the decoded values
    decoded_values_tensor = weighted_sum / num_active_neurons

    return decoded_values_tensor


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

    target_clifford = torch.empty((0,4), dtype=torch.float32).to(cfg.device)
    max_loops = 100000
    pbar = tqdm(total=max_loops, desc="Processing", dynamic_ncols=True)
    for i in range(max_loops):
        start_time = time.time()

        modelInputs, prev_gray_frame, event_frame = observeAndPreproccess(cfg, obs, prev_gray_frame, envControl.goals)
        #print(modelInputs)

        # model forward pass to determine action
        if cfg.topology_net.train:
            tNet.train()
        out_spk = tNet(modelInputs.event_frame)
        out_spks = tNet.out_spks

        action = decideAction(cfg, env, i)

        obs, _, done, _ = env.step(action)
        obs, reward_str, didEnvReset = rl_env_step(cfg, env, envControl, obs, done)
        if i % 1000 == 0 and i != 0:
            didEnvReset = True
            reward_str = "POSITIVE"

        cliff = xz_to_clifford(cfg, envControl.local_x, envControl.local_z) 
        target_clifford = torch.cat((target_clifford, cliff.unsqueeze(0)), dim=0)

        # clean up
        # remember to reset model memories on reset

        loss = None
        if didEnvReset:
             # model training
            if cfg.topology_net.train and target_clifford.shape[0] > 1:
                loss = topology_net_train(cfg, tNet, target_clifford, optimizer)

                target_clifford = torch.empty((0,4), dtype=torch.float32).to(cfg.device)
                tNet.reset()

        if didEnvReset or plot_queue.empty():
            plot_data = {
                "images": (prev_gray_frame, event_frame),
                "coordinates":((envControl.local_x, envControl.local_z), (0,0)),
                "reset": didEnvReset,
                "loss": loss
            }
            plot_queue.put(plot_data)

        if didEnvReset:
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