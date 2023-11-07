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
import matplotlib.pyplot as plt
import random
import math

from coords_processing import rate_code_to_cliff, coords_to_clifford, spikes_to_clifford, spikes_to_rate_code, coords_to_rad, coords_to_rad_scaling_factor
from image_processing import convert_gray_to_event_with_polarity

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
    
    first_output = cfg.topology_net.conv1_output_dim * pool1_out_height * pool1_out_width

    # Second Convolutional Layer
    conv2_out_height = (pool1_out_height - conv_kernel_size) // conv_stride + 1
    conv2_out_width = (pool1_out_width - conv_kernel_size) // conv_stride + 1
    
    # Second Max Pooling Layer
    pool2_out_height = (conv2_out_height - pool_kernel_size) // pool_stride + 1
    pool2_out_width = (conv2_out_width - pool_kernel_size) // pool_stride + 1

    second_output = cfg.topology_net.conv2_output_dim * pool2_out_height * pool2_out_width
 
    return first_output, pool1_out_width, second_output, pool2_out_width 

class TopologyNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg # if needed, use self.params for simplicity
        params = cfg.topology_net
        self.params = params

        assert self.params.spike_grad == "fast_sigmoid"
        spike_grad = surrogate.fast_sigmoid(slope=params.slope)

        # For convolutional layers, you might need to determine the number of output features
        # based on the input size, stride, padding, and kernel size. 
        # Here I'm assuming some arbitrary numbers as placeholders:

        num_firstconv, num_firstconv_width, num_secondconv, num_secondconv_width = calculate_output_shape(cfg)
        num_hidden = num_secondconv * params.hidden_scale
        num_outputs = params.pos_xz.num_outputs * params.pos_xz.pop_code * params.pos_xz.rate_pop_code
        self.num_outputs = num_outputs

        # Initialize learnable beta and threshold parameters for each layer
        beta_conv1 = torch.rand(num_firstconv_width, device=cfg.device, requires_grad=True)
        #threshold_conv1 = torch.rand(num_firstconv_width, device=cfg.device, requires_grad=True)

        beta_conv2 = torch.rand(num_secondconv_width, device=cfg.device, requires_grad=True)
        #threshold_conv2 = torch.rand(num_secondconv_width, device=cfg.device, requires_grad=True)

        beta_fc1 = torch.rand(num_outputs, device=cfg.device, requires_grad=True)
        #threshold_fc1 = torch.rand(num_outputs, device=cfg.device, requires_grad=True)

        beta_fc2 = torch.rand(num_outputs, device=cfg.device, requires_grad=True)
        #threshold_fc4 = torch.rand(num_outputs, device=cfg.device, requires_grad=True)


        # Initialize layers
        # shared
        self.maxpool = nn.MaxPool2d(params.maxpool2d_kernel)

        #self.dropout1 = nn.Dropout(p=cfg.topology_net.dropout_rate)
        #self.mem_dropout1 = nn.Dropout(p=cfg.topology_net.mem_dropout_rate)
        #self.dropout2 = nn.Dropout(p=cfg.topology_net.dropout_rate)
        #self.mem_dropout2 = nn.Dropout(p=cfg.topology_net.mem_dropout_rate)
        #self.dropout_fc1 = nn.Dropout(p=cfg.topology_net.dropout_rate)
        #self.mem_dropoutfc1 = nn.Dropout(p=cfg.topology_net.mem_dropout_rate)

        # first conv
        self.conv1 = nn.Conv2d(2, params.conv1_output_dim, params.conv_kernel, stride=params.conv_stride)
        self.lif_conv1 = snn.Leaky(beta=beta_conv1, learn_beta=True, spike_grad=spike_grad, reset_mechanism="zero")

        # second conv
        self.conv2 = nn.Conv2d(params.conv1_output_dim, params.conv2_output_dim, params.conv_kernel, stride=params.conv_stride)
        self.lif_conv2 = snn.Leaky(beta=beta_conv2, learn_beta=True, spike_grad=spike_grad, reset_mechanism="zero")
        self.flatten = nn.Flatten()

        
        self.fc1 = nn.Linear(num_secondconv, num_outputs)
        self.lif_fc1 = snn.Leaky(beta=beta_fc1, learn_beta=True, spike_grad=spike_grad, reset_mechanism="zero")
        self.fc2 = nn.Linear(num_outputs, num_outputs)
        self.lif_fc2 = snn.Leaky(beta=beta_fc2, learn_beta=True, spike_grad=spike_grad, output=True, reset_mechanism="zero")

        #self.out_spks = torch.empty(0, device=self.cfg.device)
        num_ann_inp = num_outputs // cfg.topology_net.pos_xz.rate_pop_code
        self.ann_seq = nn.Sequential(
            nn.Linear(num_ann_inp, num_ann_inp * 2),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(num_ann_inp * 2, num_ann_inp),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(num_ann_inp, 2),
            nn.Dropout(),
            nn.Tanh()
        )
        self.reset()

    def reset(self):
        self.spks = torch.empty((0, self.params.batch_size, self.num_outputs), device=self.cfg.device, requires_grad=True, dtype=torch.float32)
        self.mem_conv1 = self.lif_conv1.init_leaky()
        self.mem_conv2 = self.lif_conv2.init_leaky()
        self.mem_fc1 = self.lif_fc1.init_leaky()
        #mem_fc2 = self.lif_fc2.init_leaky()
        #mem_fc3 = self.lif_fc3.init_leaky()
        self.mem_fc2 = self.lif_fc2.init_leaky()
    def forward(self, x):


        conv1 = self.conv1(x)
        cur_conv1 = self.maxpool(conv1)
        spk_conv1, self.mem_conv1 = self.lif_conv1(cur_conv1, self.mem_conv1)
        #spk_conv1 = self.dropout1(spk_conv1)
        #self.mem_conv1 = self.mem_dropout1(self.mem_conv1)
        spk_conv1_sum = spk_conv1.sum()
        #print()
        conv2 = self.conv2(spk_conv1)
        cur_conv2 = self.maxpool(conv2)
        spk_conv2, self.mem_conv2 = self.lif_conv2(cur_conv2, self.mem_conv2)
        #spk_conv2 = self.dropout2(spk_conv2)
        #self.mem_conv2 = self.mem_dropout2(self.mem_conv2)
        spk_conv2_sum = spk_conv2.sum()
        #print(spk_conv2.sum())
        #reshaped = spk_conv2.view(self.params.batch_size, -1)

    
        
        flat = self.flatten(spk_conv2)
        cur_fc1 = self.fc1(flat)
        spk_fc1, self.mem_fc1 = self.lif_fc1(cur_fc1, self.mem_fc1)
        #spk_fc1 = self.dropout_fc1(spk_fc1)
        #self.mem_fc1 = self.mem_dropoutfc1(self.mem_fc1)
        #print(spk_fc1.sum())
        spk_fc1_sum = spk_fc1.sum()
        #cur_fc2 = self.fc2(spk_fc1)
        #spk_fc2, mem_fc2 = self.lif_fc2(cur_fc2, mem_fc2)
        #print(spk_fc2.sum())
        #spk_fc2_sum = spk_fc2.sum()

        #cur_fc3 = self.fc3(spk_fc2.view(self.params.batch_size, -1))
        #cur_fc3 = self.fc3(spk_fc2)
        #spk_fc3, mem_fc3 = self.lif_fc3(cur_fc3, mem_fc3)
        #spk_fc3_sum = spk_fc3.sum()

        #print(spk_fc3.sum())

        #respahed = spk_fc3.view(self.params.batch_size, -1)
        cur_fc2 = self.fc2(spk_fc1)
        spk_fc2, self.mem_fc2 = self.lif_fc2(cur_fc2, self.mem_fc2)
        spk_fc2_sum = spk_fc2.sum()

        #print(spk_fc4.sum())
        #self.out_spks = torch.cat((self.out_spks, spk_fc4), 0) 

        self.spks = torch.cat((self.spks, spk_fc2.unsqueeze(0)), dim=0)

        # allright, ANN time
        rate_code = spikes_to_rate_code(self.cfg, self.spks)

        out = self.ann_seq(rate_code)
    
        return out

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

def custom_loss(cfg,out, target):
    # Map values from range [-1, 1] to [0, 2*pi]
    mapped_output = out * torch.tensor(np.pi, device=cfg.device)
    mapped_target = target * torch.tensor(np.pi, device=cfg.device)

    # Compute the angular distances
    abs_diff = torch.abs(mapped_output - mapped_target)
    distances = torch.min(abs_diff, 2*torch.tensor(np.pi) - abs_diff)

    # You might want to take the mean or sum of the distances,
    # depending on your use case
    loss = torch.mean(distances ** 2)

    return loss

def wrap_around_zero(coords):
    # Wrap around -1 and 1
    return coords - 2 * torch.floor((coords + 1) / 2)

def rad_to_coords(cfg, coords):
    scaling_factor = coords_to_rad_scaling_factor(cfg)
    return coords / scaling_factor

def normalize_coords(cfg, rad_coords):
    # Normalize the radian values to be within the range (-1, 1)
    return rad_coords / math.pi - 1

def preprocess_training_data(cfg, tensor_dict, cap):
    # Extract saved tensors and cap them
    td_gray_frames = tensor_dict["td_gray_frames"][:cap].to(cfg.device)
    td_local_coords = tensor_dict['td_local_coords'][1:cap].to(cfg.device)

    # get event frames
    event_frames = convert_gray_to_event_with_polarity(cfg, td_gray_frames)
    
    # get target
    #targets = coords_to_clifford(cfg, td_local_coords)
    
     # Convert coordinates to radians
    rad_coords = coords_to_rad(cfg, td_local_coords)

    # Normalize coordinates to be within the range (-1, 1)
    normalized_coords = normalize_coords(cfg, rad_coords)

    # Wrap around zero
    wrapped_coords = wrap_around_zero(normalized_coords)

    return event_frames, wrapped_coords



def train():
    from config.config import load_config
    cfg = load_config("config/default_config.yaml")
    
    tNet = TopologyNet(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(tNet.parameters(), lr=cfg.topology_net.lr, betas=(0.9, 0.999))
    loss_fn = nn.MSELoss()
    loss_hist = []
    test_acc_hist = []

    print(tNet)

    file_paths = [f for f in os.listdir('data/localization') if f.endswith('.pt')]
    torch.autograd.set_detect_anomaly(True)

       
    plt.ion()  # Turn interactive mode on
    fig, ax = plt.subplots()  # Create a figure and axis object for updating

    file_paths.sort()
    random_seed = 42
    random.seed(random_seed)
    random.shuffle(file_paths)
   
    total_files = len(file_paths)
    train_size = int(total_files * 0.8)
    test_size = validation_size = int(total_files * 0.1)

    train_files = file_paths[:train_size]
    validation_files = file_paths[train_size:train_size + validation_size]
    test_files = file_paths[train_size + validation_size:]

    iter_steps = 30

    cap = 1001
    batch_size = cfg.topology_net.batch_size
    num_out_spks = cfg.topology_net.pos_xz.pop_code * cfg.topology_net.pos_xz.rate_pop_code * cfg.topology_net.pos_xz.num_outputs

    for epoch in tqdm(range(cfg.topology_net.ephocs)):
        batch_event_frames = torch.empty((0,cap-1,2,80,128), device=cfg.device)
        batch_targets = torch.empty((0,cap-1,2), device=cfg.device)

        for i, file_path in enumerate(tqdm(train_files)):
            # Load the tensor dictionary from the current file
            tensor_dict = torch.load("data/localization/" + file_path)
            
            event_frames, targets = preprocess_training_data(cfg, tensor_dict, cap)
            batch_event_frames = torch.cat((batch_event_frames, event_frames.unsqueeze(0)), dim=0)
            batch_targets = torch.cat((batch_targets, targets.unsqueeze(0)), dim=0)
            
            if batch_targets.shape[0] < batch_size:
                continue

            out_clifford_coords = torch.empty((0,batch_size,4), device=cfg.device)

            all_out = torch.empty((0,batch_size, 2), device=cfg.device, dtype=torch.float32)

            # reshape event frames so that cap is first
            # # batch_event_frames is shape (batch_size, cap-1, 2, 80, 128)
            batch_event_frames = batch_event_frames.permute((1,0,2,3,4))
            # Batch targets is shape (batch_size, cap-1, 4)
            batch_targets = batch_targets.permute((1,0,2))

            for j in range(batch_targets.shape[0]):
                ef = batch_event_frames[j]
                out = tNet(ef) # out shape [batch, 2?]
                all_out = torch.cat((all_out, out.unsqueeze(0)), dim=0)
            
            # repermute the cap and batch dimensions
            #batch_event_frames = batch_event_frames.permute((1,0,2,3,4))
            batch_targets = batch_targets.permute((1,0,2))
            all_out = all_out.permute((1,0,2))

            loss = loss_fn(all_out, batch_targets)
            loss_hist.append(loss.item())  # append loss of current iteration to loss_hist
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tNet.reset()      

            del event_frames
            del targets
            del loss
            del batch_targets
            del batch_event_frames

            torch.cuda.empty_cache()  # This will release the GPU memory back to the system

            batch_event_frames = torch.empty((0,cap-1,2,80,128), device=cfg.device)
            batch_targets = torch.empty((0,cap-1,2), device=cfg.device)

            ax.clear()  # clear previous plot
            ax.plot(loss_hist)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            ax.set_title('Loss Curve')
            #ax.set_ylim([0, 7])
            fig.canvas.draw()
            fig.canvas.flush_events()

        """
        ax.clear()  # clear previous plot
        ax.plot(loss_hist)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curve')
        ax.set_ylim([0, 1])
        fig.canvas.draw()
        fig.canvas.flush_events()
        """
        print("epoch ", epoch, " finished ")
        print("average loss", np.mean(loss_hist))
        print("Avergae loss last epoch", np.mean(loss_hist[-len(file_paths):]))

        torch.save(tNet.state_dict(), f'model_dict/model.pt')

  
if __name__ == "__main__":
    #debug_net()
    train()