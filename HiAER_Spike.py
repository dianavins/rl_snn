import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv, VecFrameStack

# Register ALE environments for ROM support
try:
    import ale_py
    # Try different registration methods for different ale_py versions
    if hasattr(ale_py, 'register_all'):
        ale_py.register_all()
    else:
        # Fallback for older versions
        gym.register_envs(ale_py)
    print("SUCCESS: ALE environments registered")
except Exception as e:
    print(f"WARNING: ALE registration issue: {e}")
    print("Proceeding anyway - environment may still work")

from spikingjelly.clock_driven import ann2snn, functional
from torch.utils.data import DataLoader, TensorDataset
import hs_api
from hs_api.converter import *

from spikingjelly.clock_driven.neuron import IFNode as ClockDrivenIFNode                                                                           
from spikingjelly.activation_based.neuron import IFNode as ActivationBasedIFNode 



# ann setup
ann_model_path = "/home/dvins/rl_snn/PongNoFrameskip-v4.zip"
env = make_atari_env("PongNoFrameskip-v4", n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)
ann_model = DQN.load(ann_model_path, env=env)
ann_model = nn.Sequential(
    ann_model.policy.q_net.features_extractor.cnn[0],
    ann_model.policy.q_net.features_extractor.cnn[2],
    ann_model.policy.q_net.features_extractor.cnn[4],
    ann_model.policy.q_net.features_extractor.linear[0],
    ann_model.policy.q_net.q_net[0],
)
print(ann_model)
# add IFNode layers after each layer except last
def add_ifnode_layers(module: nn.Sequential):
    """
    Returns a new nn.Sequential where each Conv2d and Linear layer is followed by an IFNode.
    The IFNode has a v_threshold of 1.0 and a v_reset of 0.0.
    """
    new_layers = OrderedDict()
    idx = 0
    for name, layer in module.named_children():
        new_layers[str(idx)] = layer
        idx += 1

        # Check for Conv2d or Linear
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            # Add IFNode after Conv2d or Linear
            new_layers[str(idx)] = ActivationBasedIFNode(v_threshold=1.0, v_reset=0.0)
            idx += 1

    return nn.Sequential(new_layers)
snn_model = add_ifnode_layers(ann_model)
print(snn_model)







# upload snn model
snn_model_path = "/home/dvins/rl_snn/fused_snn_pong.pth"
# Load the model (includes architecture and weights)
snn_model = torch.load(snn_model_path, weights_only=False, map_location=torch.device('cpu'))
print(snn_model)
import torch.nn as nn
from collections import OrderedDict

def insert_identity_after_convs(module: nn.Sequential, linear_out_features: int):
    """
    Returns a new nn.Sequential where each Conv2d is followed by a Linear layer.
    The Linear layer maps flattened Conv2d outputs to `linear_out_features`.
    """
    new_layers = OrderedDict()
    idx = 0
    for name, layer in module.named_children():
        new_layers[str(idx)] = layer
        idx += 1

        # Check for Conv2d
        if isinstance(layer, nn.Conv2d):
            # Compute output features: must know feature map size later
            new_layers[str(idx)] = nn.Identity()
            idx += 1

    return nn.Sequential(new_layers)


def convert_module(module):
    for name, child in module.named_children():
        print(f"Checking {name}: {type(child)}")
        # print(f"isinstance(child, ClockDrivenIFNode): {isinstance(child, ClockDrivenIFNode)}")
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

snn_model.features_extractor.cnn = convert_module(snn_model.features_extractor.cnn)
snn_model.features_extractor.linear = convert_module(snn_model.features_extractor.linear)

# add identity layers before each IFNode layer
snn_model.features_extractor.cnn = insert_identity_after_convs(
    snn_model.features_extractor.cnn,
    linear_out_features=128   # choose whatever hidden size you want
)
print("Identity layers added successfully.")

# print each layer type
for name, layer in snn_model.named_modules():
    print(f"{name}: {type(layer)}")

snn_model.eval()
print("SNN model loaded successfully.")
# print("Network pre quantization:")
# for name, param in snn_model.named_parameters():
#     if param.requires_grad:
#         print(f"{name}: {param.data.shape}, {param.data.min()}, {param.data.max()}, {param.data.mean()}")
        
# print(list(snn_model._modules))

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
# print("Post quantization:")
# for name, param in net_quan.named_parameters():
#     if param.requires_grad:
#         print(f"{name}: {param.data.shape}, {param.data.min()}, {param.data.max()}, {param.data.mean()}")
print("Network quantized successfully.")

# unrolling
num_steps   = 1              # or however many timesteps you’re using
input_layer = 0              # start converting at the very first Conv2d
output_layer = 12             # stop at the final Linear head
snn_layers   = 4             # total number of synapse layers in the model
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
print(net_quan)
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