import os
import torch
from tqdm import tqdm
from config.config import load_config

def remove_and_cap_to_1000():
    cfg = load_config("config/default_config.yaml")
    file_paths = [f for f in os.listdir('data/localization') if f.endswith('.pt')]
    file_paths.sort()

    for file_path in tqdm(file_paths):
        full_path = os.path.join("data/localization", file_path)
        tensor_dict = torch.load(full_path)

        td_event_frames = tensor_dict['td_event_frames']

        if len(td_event_frames) < 1000:
            os.remove(full_path)  # Remove the file if event frames are less than 1000
            continue

        # If the event frames are 1000 or more, cap all tensors at 1000 and prepare new dict
        new_tensor_dict = {}
        for key, tensor in tensor_dict.items():
            if len(tensor) >= 1000:
                new_tensor_dict[key] = tensor[:1000]
            else:
                new_tensor_dict[key] = tensor

        # remove old file
        os.remove(full_path)
        # Save the new tensor dictionary
        
        torch.save(new_tensor_dict, full_path)

def remove_and_normalize():
    cfg = load_config("config/default_config.yaml")
    file_paths = [f for f in os.listdir('data/localization') if f.endswith('.pt')]
    file_paths.sort()
    print(f"Processing {len(file_paths)} files...")

    for file_path in tqdm(file_paths):
        # Load the tensor dictionary from the file
        path = os.path.join("data/localization", file_path)
        tensor_dict = torch.load(path)

        # Normalize local_coords by subtracting the first local_coords from all entries
        first_local_coord = tensor_dict['td_local_coords'][0]
        tensor_dict['td_local_coords'] = tensor_dict['td_local_coords'] - first_local_coord

        # Drop 'td_event_frames' and 'td_clifford_coords' from the dictionary
        if "td_event_frames" in tensor_dict:
            del tensor_dict['td_event_frames']
        if "td_clifford_coords" in tensor_dict:
            del tensor_dict['td_clifford_coords']

        torch.save(tensor_dict, path)

        # For demonstration purposes, we'll print out the first local_coords to show normalization
        #print(f"First local_coords of {file_path} normalized to:", tensor_dict['td_local_coords'][0].tolist())

def generate_training_set():
    cfg = load_config("config/default_config.yaml")
    file_paths = [f for f in os.listdir('data/localization') if f.endswith('.pt')]
    file_paths.sort()

if __name__ == "__main__":
    #remove_and_cap_to_1000()
    remove_and_normalize()