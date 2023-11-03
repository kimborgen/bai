import cv2
import numpy as np

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
