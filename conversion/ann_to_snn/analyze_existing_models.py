#!/usr/bin/env python3
"""
Analyze existing PyTorch models and show how to convert them to HiAER format
Works with the models we already have without requiring SB3 dependencies
"""

import torch
import torch.nn as nn
import numpy as np
import os
from collections import OrderedDict, defaultdict
import json

def find_available_models():
    """Find all available PyTorch model files"""
    models = []
    for root, dirs, files in os.walk("models"):
        for file in files:
            if file.endswith(('.pt', '.pth')):
                models.append(os.path.join(root, file))
    return models

def analyze_model_structure(model_path):
    """Analyze the structure of a PyTorch model"""
    print(f"\nAnalyzing model: {model_path}")
    print("=" * 60)
    
    try:
        # Try to load the model
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        print("Checkpoint keys:")
        if isinstance(checkpoint, dict):
            for key in checkpoint.keys():
                print(f"  {key}: {type(checkpoint[key])}")
            
            # Look for model state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"\nModel state dict has {len(state_dict)} parameters:")
                
                # Group parameters by layer
                layers = defaultdict(list)
                for param_name in state_dict.keys():
                    if '.' in param_name:
                        layer_name = '.'.join(param_name.split('.')[:-1])
                        param_type = param_name.split('.')[-1]
                        layers[layer_name].append(param_type)
                
                print("\nLayers found:")
                for layer_name, params in layers.items():
                    param_shapes = []
                    for param in params:
                        full_param_name = f"{layer_name}.{param}"
                        if full_param_name in state_dict:
                            shape = state_dict[full_param_name].shape
                            param_shapes.append(f"{param}: {shape}")
                    print(f"  {layer_name}: {', '.join(param_shapes)}")
                
                return state_dict, layers
            
            elif 'policy' in checkpoint:
                print("\nFound policy in checkpoint - likely SB3 model")
                return None, None
            else:
                # Direct model save
                print("Direct model checkpoint")
                return checkpoint, None
        
        else:
            print("Direct model (not a checkpoint dict)")
            return checkpoint, None
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def create_hiaer_structure_from_state_dict(state_dict, layers, alpha=4, v_threshold=2**19):
    """Create HiAER data structures from PyTorch state dict"""
    print("\nCreating HiAER data structures from state dict...")
    print("=" * 60)
    
    axon_dict = {}
    connections = {}
    
    # Assume standard CNN + FC architecture for Pong
    input_shape = (4, 84, 84)
    input_size = np.prod(input_shape)
    
    print(f"Input layer: {input_size} axons ({input_shape})")
    
    # Create input axons
    for i in range(input_size):
        axon_dict[f"input_{i}"] = []
    
    # Process layers in order
    layer_names = sorted(layers.keys())
    current_shape = input_shape
    neuron_counter = 0
    
    for layer_idx, layer_name in enumerate(layer_names):
        params = layers[layer_name]
        print(f"\nLayer {layer_idx}: {layer_name}")
        
        # Get weight and bias tensors
        weight_key = f"{layer_name}.weight"
        bias_key = f"{layer_name}.bias"
        
        if weight_key not in state_dict:
            print(f"  No weight found for {layer_name}, skipping...")
            continue
        
        weight = state_dict[weight_key]
        bias = state_dict[bias_key] if bias_key in state_dict else None
        
        print(f"  Weight shape: {weight.shape}")
        if bias is not None:
            print(f"  Bias shape: {bias.shape}")
        
        # Determine layer type from weight shape
        if len(weight.shape) == 4:  # Conv2d: [out_channels, in_channels, kernel_h, kernel_w]
            out_c, in_c, k_h, k_w = weight.shape
            
            # Assume stride=1, padding=0 for simplicity (adjust as needed)
            stride = 1
            padding = 0
            
            if len(current_shape) == 3:
                _, in_h, in_w = current_shape
            else:
                print(f"  Warning: Unexpected input shape {current_shape} for conv layer")
                continue
            
            out_h = (in_h + 2 * padding - k_h) // stride + 1
            out_w = (in_w + 2 * padding - k_w) // stride + 1
            
            print(f"  Conv2d: {current_shape} -> ({out_c}, {out_h}, {out_w})")
            print(f"  Creating {out_c * out_h * out_w} neurons")
            
            # Create neurons and connections
            for out_ch in range(out_c):
                for out_y in range(out_h):
                    for out_x in range(out_w):
                        neuron_id = f"L{layer_idx}_{layer_name.replace('.', '_')}_C{out_ch}_Y{out_y}_X{out_x}"
                        
                        neuron_connections = []
                        
                        # Connect to input region
                        for in_ch in range(in_c):
                            for k_y in range(k_h):
                                for k_x in range(k_w):
                                    in_y = out_y * stride - padding + k_y
                                    in_x = out_x * stride - padding + k_x
                                    
                                    if 0 <= in_y < in_h and 0 <= in_x < in_w:
                                        if layer_idx == 0:
                                            input_idx = in_ch * in_h * in_w + in_y * in_w + in_x
                                            axon_id = f"input_{input_idx}"
                                        else:
                                            # Connect to previous layer (simplified)
                                            axon_id = f"L{layer_idx-1}_C{in_ch}_Y{in_y}_X{in_x}"
                                        
                                        w = weight[out_ch, in_ch, k_y, k_x].item()
                                        w_quantized = int(w * alpha)
                                        
                                        neuron_connections.append((axon_id, w_quantized))
                                        
                                        if axon_id not in axon_dict:
                                            axon_dict[axon_id] = []
                                        axon_dict[axon_id].append((neuron_id, w_quantized))
                        
                        # Add bias
                        if bias is not None:
                            bias_val = int(bias[out_ch].item() * alpha)
                            bias_axon = f"bias_{layer_name.replace('.', '_')}_C{out_ch}"
                            neuron_connections.append((bias_axon, bias_val))
                            
                            if bias_axon not in axon_dict:
                                axon_dict[bias_axon] = []
                            axon_dict[bias_axon].append((neuron_id, bias_val))
                        
                        connections[neuron_id] = (neuron_connections, "LIF_neuron")
                        neuron_counter += 1
            
            current_shape = (out_c, out_h, out_w)
            
        elif len(weight.shape) == 2:  # Linear: [out_features, in_features]
            out_features, in_features = weight.shape
            
            # Flatten input if needed
            if len(current_shape) > 1:
                input_size = np.prod(current_shape)
                print(f"  Flattening: {current_shape} -> {input_size}")
            else:
                input_size = current_shape[0]
            
            print(f"  Linear: {input_size} -> {out_features}")
            print(f"  Creating {out_features} neurons")
            
            for out_idx in range(out_features):
                neuron_id = f"L{layer_idx}_{layer_name.replace('.', '_')}_N{out_idx}"
                
                neuron_connections = []
                
                for in_idx in range(in_features):
                    if layer_idx == 0:
                        axon_id = f"input_{in_idx}"
                    else:
                        # Previous layer connections (simplified)
                        axon_id = f"L{layer_idx-1}_N{in_idx}"
                    
                    w = weight[out_idx, in_idx].item()
                    w_quantized = int(w * alpha)
                    
                    neuron_connections.append((axon_id, w_quantized))
                    
                    if axon_id not in axon_dict:
                        axon_dict[axon_id] = []
                    axon_dict[axon_id].append((neuron_id, w_quantized))
                
                # Add bias
                if bias is not None:
                    bias_val = int(bias[out_idx].item() * alpha)
                    bias_axon = f"bias_{layer_name.replace('.', '_')}_N{out_idx}"
                    neuron_connections.append((bias_axon, bias_val))
                    
                    if bias_axon not in axon_dict:
                        axon_dict[bias_axon] = []
                    axon_dict[bias_axon].append((neuron_id, bias_val))
                
                # Use ANN neuron for output layer
                neuron_type = "ANN_neuron" if layer_idx == len(layer_names) - 1 else "LIF_neuron"
                connections[neuron_id] = (neuron_connections, neuron_type)
                neuron_counter += 1
            
            current_shape = (out_features,)
    
    config = {
        'neuron_type': "LI&F",
        'global_neuron_params': {
            'v_thr': v_threshold
        }
    }
    
    print(f"\nConversion complete:")
    print(f"  Total neurons: {neuron_counter}")
    print(f"  Total axons: {len(axon_dict)}")
    print(f"  Output shape: {current_shape}")
    
    return axon_dict, connections, config

def save_hiaer_format(axon_dict, connections, config, filename):
    """Save HiAER data structures"""
    hiaer_data = {
        'axon_dict': {k: v for k, v in axon_dict.items()},
        'connections': {k: {'connections': v[0], 'neuron_type': v[1]} for k, v in connections.items()},
        'config': config
    }
    
    with open(filename, 'w') as f:
        json.dump(hiaer_data, f, indent=2)
    
    print(f"HiAER data saved to: {filename}")

def main():
    """Main analysis function"""
    print("=" * 80)
    print("EXISTING MODEL ANALYSIS FOR HIAER CONVERSION")
    print("=" * 80)
    
    # Find available models
    models = find_available_models()
    print(f"Found {len(models)} model files:")
    for model in models:
        print(f"  {model}")
    
    if not models:
        print("No model files found!")
        return
    
    # Analyze each model
    for model_path in models:
        state_dict, layers = analyze_model_structure(model_path)
        
        if state_dict is not None and layers is not None:
            # Try to create HiAER structure
            try:
                axon_dict, connections, config = create_hiaer_structure_from_state_dict(state_dict, layers)
                
                # Save the conversion
                base_name = os.path.splitext(os.path.basename(model_path))[0]
                output_file = f"hiaer_conversion_{base_name}.json"
                save_hiaer_format(axon_dict, connections, config, output_file)
                
                print(f"\n✅ Successfully converted {model_path}")
                
            except Exception as e:
                print(f"\n❌ Failed to convert {model_path}: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "-" * 60)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey Findings:")
    print("1. Direct conversion from PyTorch models to HiAER is possible")
    print("2. The sequential intermediate step can be bypassed")
    print("3. HiAER expects specific data structures:")
    print("   - axon_dict: maps axon_id -> [(neuron_id, weight), ...]")
    print("   - connections: maps neuron_id -> ([(axon_id, weight), ...], neuron_type)")
    print("   - config: neuron parameters and thresholds")
    print("\n4. For a working implementation, ensure:")
    print("   - Proper layer dimension calculations")
    print("   - Correct weight quantization (alpha scaling)")
    print("   - Appropriate neuron types (LIF vs ANN)")
    print("   - Valid axon-neuron connectivity")

if __name__ == "__main__":
    main()