import cv2
import numpy as np
import torch

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

def convert_gray_to_event_with_polarity(cfg, gray_frames):
    # Calculate the difference along the time axis (dim=0)
    differences = gray_frames[1:] - gray_frames[:-1]

    # Apply the thresholds and convert to float
    high_threshold_events = (differences >= cfg.preprocess_vision.intensity_treshold).float()
    low_threshold_events = (-differences >= cfg.preprocess_vision.intensity_treshold).float()

    # We want to exclude the first frame since it has no previous frame to compare with
    # If we still want to keep a zero tensor for the first frame, uncomment below:
    # Initialize an empty tensor for the first frame with the same shape but with an extra dimension for the channels
    #initial_frame = torch.zeros((1, 2, *gray_frames.shape[1:]), dtype=torch.float, device=cfg.device)

    # Stack the high and low threshold tensors along a new dimension
    combined_events = torch.stack((high_threshold_events, low_threshold_events), dim=2)

    # Now transpose to get the shape (number of frames - 1, 2, height, width)
    combined_events = combined_events.transpose(1, 2)

    # If we still want to keep a zero tensor for the first frame, uncomment below:
    # Concatenate with the initial frame tensor to get the correct number of frames
    #event_frames = torch.cat((initial_frame, combined_events), dim=0)

    return combined_events
