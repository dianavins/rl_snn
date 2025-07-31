import torch
import torch.nn as nn
import gymnasium as gym
import snntorch as snn
from snntorch import functional as SF
from snntorch import spikeplot as splt
import torchvision.transforms as T
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv, VecFrameStack

from spikingjelly.clock_driven import ann2snn, functional, neuron
from torch.utils.data import DataLoader, TensorDataset

from spikingjelly.clock_driven.neuron import IFNode as ClockDrivenIFNode                                                                           
from spikingjelly.activation_based.neuron import IFNode as ActivationBasedIFNode 

snn_model_path = "/home/dvins/rl_snn/fused_snn_pong.pth"
# Load the model (includes architecture and weights)
snn_model = torch.load(snn_model_path, weights_only=False, map_location=torch.device('cpu'))

def convert_module(module):
    for name, child in module.named_children():
        print(f"Converting {name}: {type(child)}")
        if isinstance(child, ClockDrivenIFNode):
            # Create new activation_based IFNode with same parameters                                                                               
            new_neuron = ActivationBasedIFNode(                                                                                                     
                v_threshold=child.v_threshold,                                                                                                      
                v_reset=child.v_reset,                                                                                                              
                surrogate_function=child.surrogate_function,                                                                                        
                detach_reset=child.detach_reset,                                                                                                    
                # step_mode=child.step_mode                                                                                                           
            )                                                                                                                                       
            setattr(module, name, new_neuron)                                                                                                       
            print(f"Converted {name}: clock_driven.IFNode -> activation_based.IFNode")
            print(f"Proof: {type(new_neuron)}")                                                                               
    return module

snn_model = convert_module(snn_model.features_extractor.cnn)
print(snn_model)