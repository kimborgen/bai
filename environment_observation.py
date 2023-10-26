import torch
import numpy as np

class EnvironmentObservation():
    def __init__(self, cfg, event_frame, pos, yaw, pitch, goals):
        self.cfg = cfg
        device = self.cfg.device
        self.event_frame = torch.from_numpy(event_frame).float().to(device)
        self.pos = torch.from_numpy(np.array(pos)).float().to(device)  # Assuming 'pos' is a list or array-like
        tmp = np.array([yaw, pitch])
        self.orientation = torch.from_numpy(tmp).float().to(device)
        self.cfg = cfg
        self.goals = goals

    def get_pos(self):
        return self.pos[:3]

    def get_orientation(self):
        return self.orientation[:2]

    def __str__(self):
        x, y, z = self.get_pos()
        yaw, pitch = self.get_orientation()
        
        return (f"x: {x.item()}\n"
                f"y: {y.item()}\n"
                f"z: {z.item()}\n"
                f"yaw: {yaw.item()}\n"
                f"pitch: {pitch.item()}")
