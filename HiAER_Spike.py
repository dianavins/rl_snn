import torch
# import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv, VecFrameStack

from spikingjelly.clock_driven import ann2snn, functional
from torch.utils.data import DataLoader, TensorDataset
import hs_api
from hs_api.converter import *


# upload snn model

snn_model_path = "/home/dvins/rl_snn/snn_pong_q_net_full.pt"
# Load the model (includes architecture and weights)
snn_model = torch.load("snn_pong_q_net_full.pt", weights_only=False, map_location=torch.device('cpu'))
snn_model.eval()
print("SNN model loaded successfully.")
print("Network pre quantization:")
for name, param in snn_model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.data.shape}, {param.data.min()}, {param.data.max()}, {param.data.mean()}")
        
print(list(snn_model._modules))

# quantize

from spikingjelly.activation_based import neuron
from spikingjelly.activation_based.ann2snn.modules import VoltageScaler
from torch.quantization import quantize_dynamic
quantized_model = quantize_dynamic(
    snn_model,                    # the original model
    {torch.nn.Linear, torch.nn.Conv2d, neuron.IFNode, VoltageScaler},            # a set of layer classes to quantize
    dtype=torch.qint8             # quantize to 8-bit integers
)
print("dynamically quantized")

alpha = 4
qn = Quantize_Network(w_alpha=alpha)
net_quan = qn.quantize(snn_model)
# print changes made to the network
print("Post quantization:")
for name, param in net_quan.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.data.shape}, {param.data.min()}, {param.data.max()}, {param.data.mean()}")
print("Network quantized successfully.")

# unrolling
num_steps   = 1              # or however many timesteps you’re using
input_layer = 0              # start converting at the very first Conv2d
output_layer = 4             # stop at the final Linear head
snn_layers   = 5             # total number of synapse layers in the model
input_shape = (4, 84, 84)    # your network’s input tensor shape
v_threshold = 1.0  # from your quantized SNN (e.g. int(IFNode.v_threshold/Δ))
embed_dim   = None           # only used for spikformer, can leave None here

cn = CRI_Converter(
    num_steps=num_steps,
    input_layer=input_layer,
    output_layer=output_layer,
    snn_layers=snn_layers,
    input_shape=input_shape,
    v_threshold=int(v_threshold),
    embed_dim=embed_dim,
    backend="spikingjelly",
)

# TODO: update v_thresholds since eahc IFNode has a VoltageScaler layer right before it


print("Converting network to CRI format aka unrolling the network...")
cn.layer_converter(net_quan)
print("Network converted to CRI format successfully.")

# initiate the model

config = {}
config['neuron_type'] = "I&F"
config['global_neuron_params'] = {}
config['global_neuron_params']['v_thr'] = int(qn.v_threshold)

softwareNetwork = CRI_network(dict(cn.axon_dict),
                              connections=dict(cn.neuron_dict),
                              config=config,target='simpleSim', 
                              outputs = cn.output_neurons,
                              coreID=1)

print("CRI network initiated successfully.")

# run the model

print("Running the model...")

inputs = ['alpha','beta']
spikes = network.step(inputs)
#Alternative
potentials, spikes = network.step(inputs, membranePotential=True)
print("Potentials and spikes obtained successfully.")
print("Potentials:", potentials)
print("Spikes:", spikes)