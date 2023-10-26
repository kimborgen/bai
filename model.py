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
        num_outputs = params.pos_xz.num_outputs + params.orientation.num_outputs + params.pos_y.num_outputs
        self.fc1 = nn.Linear(num_conv_outputs, num_hidden)
        self.lif_fc1 = snn.Leaky(beta=params.beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.lif_fc2 = snn.Leaky(beta=params.beta, spike_grad=spike_grad)
        self.fc3 = nn.Linear(num_hidden, num_hidden)
        self.lif_fc3 = snn.Leaky(beta=params.beta, spike_grad=spike_grad)
        self.fc4 = nn.Linear(num_hidden, num_outputs)
        self.lif_fc4 = snn.Leaky(beta=params.beta, spike_grad=spike_grad)

        self.reset()

    def reset(self):
        # Initialize hidden states and outputs at t=0
        self.mem_conv1 = self.lif_conv1.init_leaky()
        self.mem_conv2 = self.lif_conv2.init_leaky()
        self.mem_fc1 = self.lif_fc1.init_leaky()
        self.mem_fc2 = self.lif_fc2.init_leaky()
        self.mem_fc3 = self.lif_fc3.init_leaky()
        self.mem_fc4 = self.lif_fc4.init_leaky()

    def forward(self, x):
        conv1 = self.conv1(x)
        cur_conv1 = self.maxpool(conv1)
        spk_conv1, self.mem_conv1 = self.lif_conv1(cur_conv1, self.mem_conv1)

        conv2 = self.conv2(spk_conv1)
        cur_conv2 = self.maxpool(conv2)
        spk_conv2, self.mem_conv2 = self.lif_conv2(cur_conv2, self.mem_conv2)

        cur_fc1 = self.fc1(spk_conv2.view(self.params.batch_size, -1))
        spk_fc1, self.mem_fc1 = self.lif_fc1(cur_fc1, self.mem_fc1)

        cur_fc2 = self.fc2(spk_fc1.view(self.params.batch_size, -1))
        spk_fc2, self.mem_fc2 = self.lif_fc2(cur_fc2, self.mem_fc2)


        cur_fc3 = self.fc3(spk_fc2.view(self.params.batch_size, -1))
        spk_fc3, self.mem_fc3 = self.lif_fc3(cur_fc3, self.mem_fc3)


        cur_fc4 = self.fc4(spk_fc3.view(self.params.batch_size, -1))
        spk_fc4, self.mem_fc4 = self.lif_fc4(cur_fc4, self.mem_fc4)

        return spk_fc4, self.mem_fc4
    


def debug_net():
    from config.config import load_config
    cfg = load_config("config/default_config.yaml")
    print(cfg.device)
    #cfg.print(cfg)
    # Extracting dimensions from cfg object
    dimensions = cfg.preprocess_vision.event_frame_shape
    # Generating a random 2D array with values between 0 and 2 inclusive
    event_frame_pos = np.random.randint(0,2, size=dimensions, dtype=int)

    print(event_frame_pos)
    print(event_frame_pos.shape)

    print(cfg.device)

    tNet = TopologyNet(cfg).to(cfg.device)
    #summary(tNet, input_size=(1, 80, 128))  #
    print(sum(p.numel() for p in tNet.parameters() if p.requires_grad))

    e = torch.from_numpy(event_frame_pos).float().unsqueeze(0).to(cfg.device)

    print(tNet(e))



  
if __name__ == "__main__":
    debug_net()