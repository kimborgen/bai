import yaml
from pprint import pformat, pprint
from torch import device
from box import Box
import json

def load_config(filename):
    with open(filename, 'r') as file:
            config_data = yaml.safe_load(file)
    cfg = Box(config_data)

    # Calculating event_frame_shape
    rgb_shape = cfg.minedojo.minecraft_rgb_shape
    downscale_factor = cfg.preprocess_vision.downscale_factor
    cfg.preprocess_vision.event_frame_shape = (
        rgb_shape[0] // downscale_factor,
        rgb_shape[1] // downscale_factor
    )

    # set torch.device
    cfg.device = device(cfg.device)

    cfg.print = print_cfg
    return cfg

def print_cfg(cfg):
    # Temporarily convert torch.device to string for serialization
    original_device = cfg.device
    cfg.device = str(original_device)
    del cfg.print
    
    # Serialize the configuration
    serialized_config = json.dumps(cfg.to_dict(), indent=4)
    print(serialized_config)

    # Revert torch.device back to original form
    cfg.device = original_device
    cfg.print = print_cfg


def debug_runConfig():
    cfg = load_config('config/default_config.yaml')
    cfg.print(cfg)

if __name__ == "__main__":
    debug_runConfig()