import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from collections import defaultdict, deque
import copy
from typing import Dict, List, Tuple, Any
import cv2


class PongPreprocessor:
    """Preprocess Pong frames to match training preprocessing"""
    
    def __init__(self, frame_size: Tuple[int, int] = (84, 84)):
        self.frame_size = frame_size
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to grayscale and resize"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        binary = np.where(gray > 87, 255, 0).astype(np.uint8)
        resized = cv2.resize(binary, self.frame_size, interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0


def load_dqn_model(model_path: str = "PongNoFrameskip-v4.zip") -> DQN:
    """Load the DQN model from zip file"""
    print(f"Loading DQN model from {model_path}...")
    
    # Try loading with custom parameters to handle the ReplayBuffer issue
    try:
        model = DQN.load(model_path)
    except ValueError as e:
        if "ReplayBuffer does not support optimize_memory_usage = True and handle_timeout_termination = True" in str(e):
            print("Handling ReplayBuffer compatibility issue...")
            # Load with custom parameters that avoid the conflict
            custom_objects = {
                "optimize_memory_usage": False,
                "handle_timeout_termination": False
            }
            model = DQN.load(model_path, custom_objects=custom_objects)
        else:
            raise e
    
    model.policy.eval()
    print("Model loaded successfully")
    return model


def get_layer_by_name(model, layer_name: str):
    """Navigate to a layer by its dotted name"""
    parts = layer_name.split('.')
    current = model
    for part in parts:
        current = getattr(current, part)
    return current


def set_layer_by_name(model, layer_name: str, new_layer):
    """Replace a layer by its dotted name"""
    parts = layer_name.split('.')
    current = model
    for part in parts[:-1]:
        current = getattr(current, part)
    setattr(current, parts[-1], new_layer)


def collect_activations(model: DQN, num_episodes: int = 10, max_steps: int = 1000) -> Dict[str, List[torch.Tensor]]:
    """
    Run the model on Pong episodes and collect activations from all layers
    
    Args:
        model: The DQN model
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        
    Returns:
        Dictionary mapping layer names to lists of activation tensors
    """
    print(f"Collecting activations over {num_episodes} episodes...")
    
    # Create environment
    env = gym.make('PongNoFrameskip-v4')
    preprocessor = PongPreprocessor()
    
    # Storage for activations
    activations = defaultdict(list)
    
    # Hook function to capture activations
    def get_activation_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                # Store detached copy to avoid memory issues
                activations[name].append(output.detach().cpu())
        return hook
    
    # Register hooks on all layers
    hooks = []
    q_net = model.policy.q_net
    
    # Register hooks for conv layers and linear layers
    layer_names = {}
    for name, module in q_net.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layer_names[name] = module
            hook = module.register_forward_hook(get_activation_hook(name))
            hooks.append(hook)
    
    print(f"Registered hooks for layers: {list(layer_names.keys())}")
    
    model.policy.eval()
    
    try:
        for episode in range(num_episodes):
            obs, _ = env.reset()
            obs = preprocessor.preprocess_frame(obs)
            
            # Stack 4 frames as expected by the model
            stacked_frames = deque([obs] * 4, maxlen=4)
            obs = np.stack(stacked_frames, axis=0)
            
            episode_steps = 0
            
            while episode_steps < max_steps:
                # Convert to tensor and add batch dimension
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                
                # Forward pass through the model (this triggers the hooks)
                with torch.no_grad():
                    action, _ = model.predict(obs, deterministic=True)
                
                # Take step in environment
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                if done:
                    break
                
                # Preprocess next observation
                next_obs = preprocessor.preprocess_frame(next_obs)
                stacked_frames.append(next_obs)
                obs = np.stack(stacked_frames, axis=0)
                
                episode_steps += 1
            
            if (episode + 1) % 2 == 0:
                print(f"Completed episode {episode + 1}/{num_episodes}")
    
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()
        env.close()
    
    # Convert lists to tensors for easier processing
    processed_activations = {}
    for layer_name, activation_list in activations.items():
        if activation_list:
            # Concatenate all activations along batch dimension
            processed_activations[layer_name] = torch.cat(activation_list, dim=0)
            print(f"Layer {layer_name}: collected {processed_activations[layer_name].shape[0]} samples")
    
    return processed_activations


def identify_least_active_nodes(activations: Dict[str, torch.Tensor], pruning_ratio: float = 0.05) -> Dict[str, List[int]]:
    """
    Identify the least active nodes in each layer
    
    Args:
        activations: Dictionary mapping layer names to activation tensors
        pruning_ratio: Fraction of nodes to prune (default 5%)
        
    Returns:
        Dictionary mapping layer names to lists of node indices to remove
    """
    print(f"Identifying least active nodes (bottom {pruning_ratio*100}%)...")
    
    nodes_to_remove = {}
    
    for layer_name, layer_activations in activations.items():
        print(f"\nAnalyzing layer: {layer_name}")
        print(f"Activation shape: {layer_activations.shape}")
        
        # Calculate average activation per node across all samples and spatial dimensions
        if len(layer_activations.shape) == 4:  # Conv2d: (batch, channels, height, width)
            # Average over batch, height, width dimensions
            avg_activations = layer_activations.mean(dim=(0, 2, 3))
            num_nodes = layer_activations.shape[1]
        elif len(layer_activations.shape) == 2:  # Linear: (batch, features)
            # Average over batch dimension
            avg_activations = layer_activations.mean(dim=0)
            num_nodes = layer_activations.shape[1]
        else:
            print(f"Skipping layer {layer_name} with unexpected shape: {layer_activations.shape}")
            continue
        
        # Don't prune the final output layer (q_values)
        if 'q_net.0' in layer_name:
            print(f"Skipping final output layer {layer_name}")
            continue
        
        # Number of nodes to remove
        num_to_remove = max(1, int(num_nodes * pruning_ratio))
        
        # Find indices of least active nodes
        _, least_active_indices = torch.topk(avg_activations, num_to_remove, largest=False)
        least_active_indices = least_active_indices.tolist()
        
        nodes_to_remove[layer_name] = least_active_indices
        
        print(f"Nodes to remove: {num_to_remove}/{num_nodes}")
        print(f"Average activations range: {avg_activations.min():.6f} to {avg_activations.max():.6f}")
        print(f"Least active nodes: {least_active_indices}")
        print(f"Their average activations: {avg_activations[least_active_indices].tolist()}")
    
    return nodes_to_remove


def copy_conv_weights(old_layer, new_layer, nodes_to_remove, input_mask=None):
    """Copy conv weights excluding pruned channels"""
    old_out_channels = old_layer.weight.shape[0]
    old_in_channels = old_layer.weight.shape[1]
    
    # Create mask for keeping output channels
    out_keep_mask = torch.ones(old_out_channels, dtype=torch.bool)
    if nodes_to_remove:
        out_keep_mask[nodes_to_remove] = False
    
    # Handle input channel masking (for layers that depend on pruned previous layers)
    if input_mask is not None:
        # Select both input and output channels
        new_layer.weight.data = old_layer.weight.data[out_keep_mask][:, input_mask]
    else:
        # Only select output channels
        new_layer.weight.data = old_layer.weight.data[out_keep_mask]
    
    # Copy biases
    if old_layer.bias is not None:
        new_layer.bias.data = old_layer.bias.data[out_keep_mask]


def copy_linear_weights(old_layer, new_layer, nodes_to_remove, input_mask=None):
    """Copy linear weights excluding pruned neurons"""
    old_out_features = old_layer.weight.shape[0]
    old_in_features = old_layer.weight.shape[1]
    
    # Create mask for keeping output features
    out_keep_mask = torch.ones(old_out_features, dtype=torch.bool)
    if nodes_to_remove:
        out_keep_mask[nodes_to_remove] = False
    
    # Handle input feature masking
    if input_mask is not None:
        # Select both input and output features
        new_layer.weight.data = old_layer.weight.data[out_keep_mask][:, input_mask]
    else:
        # Only select output features
        new_layer.weight.data = old_layer.weight.data[out_keep_mask]
    
    # Copy biases
    if old_layer.bias is not None:
        new_layer.bias.data = old_layer.bias.data[out_keep_mask]


def create_pruned_network(original_model: DQN, nodes_to_remove: Dict[str, List[int]]) -> DQN:
    """
    Create a new network with the specified nodes removed using proper dependency tracking
    
    Args:
        original_model: Original DQN model
        nodes_to_remove: Dictionary mapping layer names to node indices to remove
        
    Returns:
        New DQN model with pruned architecture
    """
    print("Creating pruned network with proper dependency tracking...")
    
    # Define the dependency map for the DQN architecture
    dependency_map = {
        'features_extractor.cnn.0': ['features_extractor.cnn.2'],      # First conv affects second conv
        'features_extractor.cnn.2': ['features_extractor.cnn.4'],      # Second conv affects third conv  
        'features_extractor.cnn.4': ['features_extractor.linear.0'],   # Third conv affects first linear
        'features_extractor.linear.0': ['q_net.0']                     # First linear affects final linear
    }
    
    # Layer processing order (topological)
    layer_order = [
        'features_extractor.cnn.0',
        'features_extractor.cnn.2', 
        'features_extractor.cnn.4',
        'features_extractor.linear.0',
        'q_net.0'
    ]
    
    # Calculate size changes and create masks
    size_changes = {}
    channel_masks = {}  # Keep track of which channels/neurons to keep
    
    for layer_name in layer_order:
        old_layer = get_layer_by_name(original_model.policy.q_net, layer_name)
        
        if layer_name in nodes_to_remove:
            pruned_indices = nodes_to_remove[layer_name]
            
            if isinstance(old_layer, nn.Conv2d):
                old_out_channels = old_layer.out_channels
                keep_mask = torch.ones(old_out_channels, dtype=torch.bool)
                keep_mask[pruned_indices] = False
                channel_masks[layer_name] = keep_mask
                
                new_out_channels = old_out_channels - len(pruned_indices)
                size_changes[layer_name] = {
                    'out_channels': new_out_channels,
                    'original_out': old_out_channels
                }
                
                print(f"Layer {layer_name}: {old_out_channels} -> {new_out_channels} output channels")
                
            elif isinstance(old_layer, nn.Linear):
                old_out_features = old_layer.out_features
                keep_mask = torch.ones(old_out_features, dtype=torch.bool)
                keep_mask[pruned_indices] = False
                channel_masks[layer_name] = keep_mask
                
                new_out_features = old_out_features - len(pruned_indices)
                size_changes[layer_name] = {
                    'out_features': new_out_features,
                    'original_out': old_out_features
                }
                
                print(f"Layer {layer_name}: {old_out_features} -> {new_out_features} output features")
        else:
            # Layer not being pruned, but may need input size adjustment
            if isinstance(old_layer, nn.Conv2d):
                channel_masks[layer_name] = torch.ones(old_layer.out_channels, dtype=torch.bool)
            elif isinstance(old_layer, nn.Linear):
                channel_masks[layer_name] = torch.ones(old_layer.out_features, dtype=torch.bool)
    
    # Calculate input size changes for dependent layers
    for layer_name in layer_order:
        if layer_name in dependency_map:
            dependent_layers = dependency_map[layer_name]
            
            for dependent_layer in dependent_layers:
                dependent_old = get_layer_by_name(original_model.policy.q_net, dependent_layer)
                
                if layer_name in size_changes:
                    # This layer was pruned, so dependent layer needs input adjustment
                    if isinstance(dependent_old, nn.Conv2d):
                        new_in_channels = size_changes[layer_name]['out_channels']
                        size_changes.setdefault(dependent_layer, {})
                        size_changes[dependent_layer]['in_channels'] = new_in_channels
                        
                    elif isinstance(dependent_old, nn.Linear):
                        if layer_name == 'features_extractor.cnn.4':
                            # Special case: conv to linear transition
                            new_channels = size_changes[layer_name]['out_channels']
                            # Spatial dimensions are 7x7 after the conv layers
                            new_in_features = new_channels * 7 * 7
                            size_changes.setdefault(dependent_layer, {})
                            size_changes[dependent_layer]['in_features'] = new_in_features
                            print(f"Conv-to-Linear: {dependent_layer} input size {dependent_old.in_features} -> {new_in_features}")
                        else:
                            # Linear to linear
                            new_in_features = size_changes[layer_name]['out_features']
                            size_changes.setdefault(dependent_layer, {})
                            size_changes[dependent_layer]['in_features'] = new_in_features
    
    # Create the pruned model
    pruned_model = copy.deepcopy(original_model)
    
    # Replace layers with new architectures
    for layer_name in layer_order:
        old_layer = get_layer_by_name(original_model.policy.q_net, layer_name)
        changes = size_changes.get(layer_name, {})
        
        if isinstance(old_layer, nn.Conv2d):
            new_layer = nn.Conv2d(
                in_channels=changes.get('in_channels', old_layer.in_channels),
                out_channels=changes.get('out_channels', old_layer.out_channels),
                kernel_size=old_layer.kernel_size,
                stride=old_layer.stride,
                padding=old_layer.padding,
                bias=old_layer.bias is not None
            )
            
            # Determine input mask from previous layer
            input_mask = None
            for prev_layer, deps in dependency_map.items():
                if layer_name in deps and prev_layer in channel_masks:
                    input_mask = channel_masks[prev_layer]
                    break
            
            # Copy weights
            copy_conv_weights(
                old_layer, 
                new_layer, 
                nodes_to_remove.get(layer_name, []),
                input_mask
            )
            
        elif isinstance(old_layer, nn.Linear):
            new_layer = nn.Linear(
                in_features=changes.get('in_features', old_layer.in_features),
                out_features=changes.get('out_features', old_layer.out_features),
                bias=old_layer.bias is not None
            )
            
            # Determine input mask from previous layer
            input_mask = None
            for prev_layer, deps in dependency_map.items():
                if layer_name in deps and prev_layer in channel_masks:
                    if prev_layer == 'features_extractor.cnn.4':
                        # Special handling for conv-to-linear: expand mask for flattened tensor
                        conv_mask = channel_masks[prev_layer]
                        # Create mask for flattened tensor (channels * 7 * 7)
                        expanded_mask = conv_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, 7, 7).flatten()
                        input_mask = expanded_mask
                    else:
                        input_mask = channel_masks[prev_layer]
                    break
            
            # Copy weights
            copy_linear_weights(
                old_layer,
                new_layer,
                nodes_to_remove.get(layer_name, []),
                input_mask
            )
        
        # Replace the layer in the pruned model
        set_layer_by_name(pruned_model.policy.q_net, layer_name, new_layer)
        print(f"Replaced layer {layer_name}")
    
    print("Pruned network created successfully with proper dependency tracking")
    return pruned_model


def evaluate_model_performance(model: DQN, num_episodes: int = 5) -> float:
    """Evaluate model performance on Pong"""
    print(f"Evaluating model performance over {num_episodes} episodes...")
    
    env = gym.make('PongNoFrameskip-v4')
    preprocessor = PongPreprocessor()
    
    total_reward = 0
    
    try:
        for episode in range(num_episodes):
            obs, _ = env.reset()
            obs = preprocessor.preprocess_frame(obs)
            stacked_frames = deque([obs] * 4, maxlen=4)
            obs = np.stack(stacked_frames, axis=0)
            
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                next_obs = preprocessor.preprocess_frame(next_obs)
                stacked_frames.append(next_obs)
                obs = np.stack(stacked_frames, axis=0)
                
                episode_reward += reward
            
            total_reward += episode_reward
            print(f"Episode {episode + 1}: {episode_reward}")
        
    finally:
        env.close()
    
    avg_reward = total_reward / num_episodes
    print(f"Average reward: {avg_reward:.2f}")
    return avg_reward


def main():
    """Main function to run the network pruning pipeline"""
    print("Starting network pruning pipeline...")
    
    # Load the original model
    original_model = load_dqn_model("PongNoFrameskip-v4.zip")
    
    # Print model architecture
    print("\n" + "="*50)
    print("ORIGINAL MODEL ARCHITECTURE")
    print("="*50)
    for name, module in original_model.policy.q_net.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            print(f"{name}: {module}")
    
    # Evaluate original model performance
    print("\n" + "="*50)
    print("EVALUATING ORIGINAL MODEL")
    print("="*50)
    original_performance = evaluate_model_performance(original_model, num_episodes=3)
    
    # Collect activations
    print("\n" + "="*50)
    print("COLLECTING ACTIVATIONS")
    print("="*50)
    activations = collect_activations(original_model, num_episodes=5, max_steps=500)
    
    # Identify least active nodes
    print("\n" + "="*50)
    print("IDENTIFYING LEAST ACTIVE NODES")
    print("="*50)
    nodes_to_remove = identify_least_active_nodes(activations, pruning_ratio=0.05)
    
    # Create pruned network
    print("\n" + "="*50)
    print("CREATING PRUNED NETWORK")
    print("="*50)
    try:
        pruned_model = create_pruned_network(original_model, nodes_to_remove)
        
        # Print pruned model architecture
        print("\n" + "="*30)
        print("PRUNED MODEL ARCHITECTURE")
        print("="*30)
        for name, module in pruned_model.policy.q_net.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                print(f"{name}: {module}")
        
        # Test that the pruned model can run inference
        print("\n" + "="*30)
        print("TESTING PRUNED MODEL INFERENCE")
        print("="*30)
        env = gym.make('PongNoFrameskip-v4') 
        obs, _ = env.reset()
        preprocessor = PongPreprocessor()
        obs = preprocessor.preprocess_frame(obs)
        stacked_frames = deque([obs] * 4, maxlen=4)
        obs = np.stack(stacked_frames, axis=0)
        
        try:
            action, _ = pruned_model.predict(obs, deterministic=True)
            print(f"✓ Pruned model inference successful! Action: {action}")
        except Exception as e:
            print(f"✗ Pruned model inference failed: {e}")
            env.close()
            return original_model, nodes_to_remove, activations
        
        env.close()
        
        # Evaluate pruned model performance
        print("\n" + "="*50)
        print("EVALUATING PRUNED MODEL")
        print("="*50)
        pruned_performance = evaluate_model_performance(pruned_model, num_episodes=3)
        
        # Compare performance
        print("\n" + "="*50)
        print("PERFORMANCE COMPARISON")
        print("="*50)
        print(f"Original model average reward: {original_performance:.2f}")
        print(f"Pruned model average reward: {pruned_performance:.2f}")
        print(f"Performance change: {pruned_performance - original_performance:.2f}")
        if original_performance != 0:
            print(f"Performance retention: {(pruned_performance/original_performance)*100:.1f}%")
        
        # Calculate parameter reduction
        original_params = sum(p.numel() for p in original_model.policy.q_net.parameters())
        pruned_params = sum(p.numel() for p in pruned_model.policy.q_net.parameters())
        reduction = (original_params - pruned_params) / original_params * 100
        
        print(f"\nParameter count:")
        print(f"Original: {original_params:,}")
        print(f"Pruned: {pruned_params:,}")
        print(f"Reduction: {reduction:.2f}%")
        
        # Save pruned model
        pruned_model.save("pruned_pong_model")
        print("\nPruned model saved as 'pruned_pong_model.zip'")
        
        return pruned_model, nodes_to_remove, activations
        
    except Exception as e:
        print(f"Error creating pruned network: {e}")
        import traceback
        traceback.print_exc()
        return original_model, nodes_to_remove, activations


if __name__ == "__main__":
    model, nodes_to_remove, activations = main()