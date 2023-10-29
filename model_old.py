# imports
import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools
from torchsummary import summary
import os
from tqdm import tqdm
import torch.nn.functional as torchF

def calculate_output_shape(cfg):
    # Unpack input shape
    input_height, input_width = cfg.preprocess_vision.event_frame_shape
    params = cfg.topology_net
    
    conv_stride = params.conv_stride
    pool_stride = params.maxpool2d_kernel
    conv_kernel_size = params.conv_kernel
    pool_kernel_size = params.maxpool2d_kernel

    # First Convolutional Layer
    conv1_out_height = (input_height - conv_kernel_size) // conv_stride + 1
    conv1_out_width = (input_width - conv_kernel_size) // conv_stride + 1
    
    # First Max Pooling Layer
    pool1_out_height = (conv1_out_height - pool_kernel_size) // pool_stride + 1
    pool1_out_width = (conv1_out_width - pool_kernel_size) // pool_stride + 1
    
    # Second Convolutional Layer
    conv2_out_height = (pool1_out_height - conv_kernel_size) // conv_stride + 1
    conv2_out_width = (pool1_out_width - conv_kernel_size) // conv_stride + 1
    
    # Second Max Pooling Layer
    pool2_out_height = (conv2_out_height - pool_kernel_size) // pool_stride + 1
    pool2_out_width = (conv2_out_width - pool_kernel_size) // pool_stride + 1
    
    return cfg.topology_net.conv2_output_dim * pool2_out_height * pool2_out_width

class TopologyNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg # if needed, use self.params for simplicity
        params = cfg.topology_net
        self.params = params
        assert self.params.spike_grad == "fast_sigmoid"
        spike_grad = surrogate.fast_sigmoid(slope=params.slope)

        # Initialize layers
        self.conv1 = nn.Conv2d(1, params.conv1_output_dim, params.conv_kernel, stride=params.conv_stride)
        self.lif_conv1 = snn.Leaky(beta=params.beta, spike_grad=spike_grad)
        self.maxpool = nn.MaxPool2d(params.maxpool2d_kernel)
        self.conv2 = nn.Conv2d(params.conv1_output_dim, params.conv2_output_dim, params.conv_kernel, stride=params.conv_stride)
        self.lif_conv2 = snn.Leaky(beta=params.beta, spike_grad=spike_grad)
        num_conv_outputs = calculate_output_shape(cfg)
        num_hidden = num_conv_outputs * params.hidden_scale
        num_outputs = params.pos_xz.num_outputs * params.pos_xz.pop_code + params.orientation.num_outputs + params.pos_y.num_outputs
        self.fc1 = nn.Linear(num_conv_outputs, num_hidden)
        self.lif_fc1 = snn.Leaky(beta=params.beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.lif_fc2 = snn.Leaky(beta=params.beta, spike_grad=spike_grad)
        self.fc3 = nn.Linear(num_hidden, num_hidden)
        self.lif_fc3 = snn.Leaky(beta=params.beta, spike_grad=spike_grad)
        self.fc4 = nn.Linear(num_hidden, num_outputs)
        self.lif_fc4 = snn.Leaky(beta=params.beta, spike_grad=spike_grad, output=True)

        self.out_spks = torch.empty(0, device=self.cfg.device)

        self.reset()

    def reset(self):
        # Initialize hidden states and outputs at t=0
        self.mem_conv1 = self.lif_conv1.init_leaky()
        self.mem_conv2 = self.lif_conv2.init_leaky()
        self.mem_fc1 = self.lif_fc1.init_leaky()
        self.mem_fc2 = self.lif_fc2.init_leaky()
        self.mem_fc3 = self.lif_fc3.init_leaky()
        self.mem_fc4 = self.lif_fc4.init_leaky()
        #self.out_spks = torch.empty(0, device=self.cfg.device)

    def forward(self, x):
        conv1 = self.conv1(x)
        cur_conv1 = self.maxpool(conv1)
        spk_conv1, self.mem_conv1 = self.lif_conv1(cur_conv1, self.mem_conv1)
        #print(spk_conv1.sum() >= 1)
        conv2 = self.conv2(spk_conv1)
        cur_conv2 = self.maxpool(conv2)
        spk_conv2, self.mem_conv2 = self.lif_conv2(cur_conv2, self.mem_conv2)
        #print(spk_conv2.sum() >= 1)
        reshaped = spk_conv2.view(self.params.batch_size, -1)
        cur_fc1 = self.fc1(reshaped)
        spk_fc1, self.mem_fc1 = self.lif_fc1(cur_fc1, self.mem_fc1)
        #print(spk_fc1.sum() >= 1)
        cur_fc2 = self.fc2(spk_fc1)
        spk_fc2, self.mem_fc2 = self.lif_fc2(cur_fc2, self.mem_fc2)
        #print(spk_fc2.sum() >= 1)

        #cur_fc3 = self.fc3(spk_fc2.view(self.params.batch_size, -1))
        cur_fc3 = self.fc3(spk_fc2)
        spk_fc3, self.mem_fc3 = self.lif_fc3(cur_fc3, self.mem_fc3)
        #print(spk_fc3.sum() >= 1)

        #respahed = spk_fc3.view(self.params.batch_size, -1)
        cur_fc4 = self.fc4(spk_fc3)
        spk_fc4, self.mem_fc4 = self.lif_fc4(cur_fc4, self.mem_fc4)
        #print(spk_fc4.sum() >= 1)
        #self.out_spks = torch.cat((self.out_spks, spk_fc4), 0) 

        return spk_fc4, self.mem_fc4

def debug_net():
    from config.config import load_config
    cfg = load_config("config/default_config.yaml")
    print(cfg.device)
    #cfg.print(cfg)
    # Extracting dimensions from cfg object
    dimensions = cfg.preprocess_vision.event_frame_shape
   
    tNet = TopologyNet(cfg).to(cfg.device)
    #summary(tNet, input_size=(1, 80, 128))  #
    print(sum(p.numel() for p in tNet.parameters() if p.requires_grad))

    for i in range(1000):
         # Generating a random 2D array with values between 0 and 2 inclusive
        event_frame_pos = np.random.randint(0,2, size=dimensions, dtype=int)
        e = torch.from_numpy(event_frame_pos).float().unsqueeze(0).to(cfg.device)
        spks, mems = tNet(e)
        print("Outspks: ", spks.sum() >= 1)
        print("\n\n")

def topology_net_train(cfg, tNet, target, optimizer):
    params = cfg.topology_net.pos_xz
    #spks = torch.randint(0, 2, size=(target.shape[0], 200)).to(cfg.device)
    spks = tNet.out_spks

    

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

def train():
    from config.config import load_config
    cfg = load_config("config/default_config.yaml")
    
    tNet = TopologyNet(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(tNet.parameters(), lr=cfg.topology_net.lr, betas=(0.9, 0.999))
    loss_hist = []
    test_acc_hist = []

    file_paths = [f for f in os.listdir('data/localization') if f.endswith('.pt')]
    torch.autograd.set_detect_anomaly(True)
    for file_path in tqdm(file_paths):
        # Load the tensor dictionary from the current file
        tensor_dict = torch.load("data/localization/" + file_path)
        td_gray_frames = tensor_dict["td_gray_frames"].to(cfg.device)
        td_event_frames = tensor_dict['td_event_frames'].to(cfg.device)
        td_local_coords = tensor_dict['td_local_coords'].to(cfg.device)
        td_clifford_coords = tensor_dict['td_clifford_coords'].to(cfg.device)
        optimizer.zero_grad()

        for frame in range(td_event_frames.shape[0]): 
            event_frame = td_event_frames[frame].unsqueeze(0)
            target_coords = td_clifford_coords[frame].unsqueeze(0)
            spks, mems = tNet(event_frame)

            out_clifford_coords = spikes_to_clifford(cfg, spks)
            loss = torchF.mse_loss(out_clifford_coords, target_coords)
            print(loss)

            loss.backward(retain_graph=True)
            optimizer.step()
        tNet.reset()

  
if __name__ == "__main__":
    #debug_net()
    train()