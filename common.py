"""Configuration and input loading."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from policy import Policy
from environment import MicrogridEnv2EV_Baseline

ROOT = Path(__file__).resolve().parent

def setup(config_path, device):
    config = json.loads(Path(config_path).read_text())
    torch.set_num_threads(config['cpu_threads'])
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable')
    return config

def load_data(path):
    frame = pd.read_csv(path)
    columns = ['hvac_power', 'solar_power', 'alpha']
    if not set(columns).issubset(frame.columns):
        raise ValueError(f'CSV requires columns {columns}')
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or len(frame) % 24:
        raise ValueError('Data must contain finite nonnegative values and complete 24-hour days')
    return (frame.hvac_power.to_numpy() / 1000,
            frame.solar_power.to_numpy() / 1000,
            frame.alpha.to_numpy(dtype=np.float32),
            frame.alpha.to_numpy(dtype=np.float32) * 0.06)

def make_env(data, day):
    if day < 0 or (day + 1) * 24 > len(data[0]):
        raise ValueError('Day index outside dataset')
    return MicrogridEnv2EV_Baseline(*(x[day*24:(day+1)*24] for x in data))

def make_agent(config, device):
    return Policy(device)
