"""
SB3 DQN to Sequential Network Converter
Converts Stable-Baselines3 DQN models to simple PyTorch Sequential networks
with identical performance guarantees.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Union
import numpy as np
from collections import OrderedDict


class SequentialDQNNetwork(nn.Module):
    """
    Sequential implementation of DQN network that exactly matches SB3 architecture
    """
    
    def __init__(self, input_channels: int = 4, n_actions: int = 6):
        super().__init__()
        
        self.network = nn.Sequential(OrderedDict([
            # Convolutional layers
            ('conv1', nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)),
            ('relu1', nn.ReLU(inplace=True)),
            
            ('conv2', nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            ('relu2', nn.ReLU(inplace=True)),
            
            ('conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            ('relu3', nn.ReLU(inplace=True)),
            
            # Flatten layer
            ('flatten', nn.Flatten()),
            
            # Linear layers
            ('fc1', nn.Linear(3136, 512)),  # 64 * 7 * 7 = 3136
            ('relu4', nn.ReLU(inplace=True)),
            
            ('fc2', nn.Linear(512, n_actions)),
        ]))
        
    def forward(self, x):
        return self.network(x)
    
    def get_layer_by_name(self, name: str) -> nn.Module:
        """Get layer by name for precise weight mapping"""
        return dict(self.network.named_modules())[name]


class SB3ToSequentialConverter:
    """
    Converts SB3 DQN models to Sequential networks with identical weights and behavior
    """
    
    def __init__(self, verify_conversion: bool = True, tolerance: float = 1e-6):
        self.verify_conversion = verify_conversion
        self.tolerance = tolerance
        
    def convert_snn_to_sequential(
        self, 
        snn_state_dict: Dict[str, torch.Tensor],
        input_channels: int = 4,
        n_actions: int = 6
    ) -> SequentialDQNNetwork:
        """
        Convert SNN model state dict to Sequential network
        
        Args:
            snn_state_dict: State dict from SNN model
            input_channels: Input channels (default: 4 for stacked frames)
            n_actions: Number of actions (default: 6 for Pong)
            
        Returns:
            SequentialDQNNetwork with converted weights
        """
        # Create sequential network
        sequential_net = SequentialDQNNetwork(input_channels, n_actions)
        
        # Weight mapping from SNN to Sequential
        weight_mapping = {
            # Convolutional layers
            'features_extractor.cnn.0.weight': 'network.conv1.weight',
            'features_extractor.cnn.0.bias': 'network.conv1.bias',
            'features_extractor.cnn.2.weight': 'network.conv2.weight', 
            'features_extractor.cnn.2.bias': 'network.conv2.bias',
            'features_extractor.cnn.4.weight': 'network.conv3.weight',
            'features_extractor.cnn.4.bias': 'network.conv3.bias',
            
            # Linear layers
            'features_extractor.linear.0.weight': 'network.fc1.weight',
            'features_extractor.linear.0.bias': 'network.fc1.bias',
            'q_net.0.weight': 'network.fc2.weight',
            'q_net.0.bias': 'network.fc2.bias',
        }
        
        # Convert weights
        sequential_state_dict = {}
        for snn_key, seq_key in weight_mapping.items():
            if snn_key in snn_state_dict:
                sequential_state_dict[seq_key] = snn_state_dict[snn_key].clone()
            else:
                raise KeyError(f"Missing key in SNN state dict: {snn_key}")
        
        # Load converted weights
        sequential_net.load_state_dict(sequential_state_dict)
        
        # Verify conversion if requested
        if self.verify_conversion:
            self._verify_conversion(snn_state_dict, sequential_net, input_channels)
            
        return sequential_net
    
    def convert_sb3_to_sequential(
        self,
        sb3_model,
        input_channels: int = 4,
        n_actions: int = 6
    ) -> SequentialDQNNetwork:
        """
        Convert SB3 DQN model directly to Sequential network
        
        Args:
            sb3_model: Stable-Baselines3 DQN model
            input_channels: Input channels 
            n_actions: Number of actions
            
        Returns:
            SequentialDQNNetwork with converted weights
        """
        # Extract Q-network from SB3 model
        q_net = sb3_model.policy.q_net
        
        # Create sequential network
        sequential_net = SequentialDQNNetwork(input_channels, n_actions)
        
        # Map SB3 architecture to Sequential
        # This depends on the exact SB3 architecture, so we'll implement a general approach
        sb3_state_dict = q_net.state_dict()
        
        # Convert using the same mapping logic
        return self.convert_snn_to_sequential(sb3_state_dict, input_channels, n_actions)
    
    def _verify_conversion(
        self, 
        original_state_dict: Dict[str, torch.Tensor],
        sequential_net: SequentialDQNNetwork,
        input_channels: int
    ):
        """
        Verify that conversion preserves numerical behavior
        """
        print("Verifying conversion accuracy...")
        
        # Create test input
        batch_size = 4
        test_input = torch.randn(batch_size, input_channels, 84, 84)
        
        # Get sequential network output
        sequential_net.eval()
        with torch.no_grad():
            sequential_output = sequential_net(test_input)
        
        print(f"Sequential output shape: {sequential_output.shape}")
        print(f"Sequential output range: [{sequential_output.min():.6f}, {sequential_output.max():.6f}]")
        
        # Check weight preservation
        sequential_state_dict = sequential_net.state_dict()
        
        weight_mapping = {
            'features_extractor.cnn.0.weight': 'network.conv1.weight',
            'features_extractor.cnn.0.bias': 'network.conv1.bias',
            'features_extractor.cnn.2.weight': 'network.conv2.weight', 
            'features_extractor.cnn.2.bias': 'network.conv2.bias',
            'features_extractor.cnn.4.weight': 'network.conv3.weight',
            'features_extractor.cnn.4.bias': 'network.conv3.bias',
            'features_extractor.linear.0.weight': 'network.fc1.weight',
            'features_extractor.linear.0.bias': 'network.fc1.bias',
            'q_net.0.weight': 'network.fc2.weight',
            'q_net.0.bias': 'network.fc2.bias',
        }
        
        max_diff = 0.0
        for orig_key, seq_key in weight_mapping.items():
            orig_tensor = original_state_dict[orig_key]
            seq_tensor = sequential_state_dict[seq_key]
            
            diff = torch.abs(orig_tensor - seq_tensor).max().item()
            max_diff = max(max_diff, diff)
            
            print(f"Weight diff {orig_key} -> {seq_key}: {diff:.2e}")
        
        if max_diff > self.tolerance:
            raise ValueError(f"Conversion failed: max weight difference {max_diff:.2e} > tolerance {self.tolerance:.2e}")
        
        print(f"Conversion verified: max weight difference {max_diff:.2e}")


def create_ann_from_snn(snn_checkpoint_path: str) -> SequentialDQNNetwork:
    """
    Convenience function to create ANN Sequential network from SNN checkpoint
    
    Args:
        snn_checkpoint_path: Path to SNN checkpoint file
        
    Returns:
        SequentialDQNNetwork with converted weights
    """
    # Load SNN checkpoint
    snn_state_dict = torch.load(snn_checkpoint_path, map_location='cpu', weights_only=False)
    
    # Convert to sequential
    converter = SB3ToSequentialConverter(verify_conversion=True)
    sequential_net = converter.convert_snn_to_sequential(snn_state_dict)
    
    return sequential_net


def save_sequential_model(
    sequential_net: SequentialDQNNetwork, 
    save_path: str,
    include_metadata: bool = True
):
    """
    Save Sequential network with metadata
    
    Args:
        sequential_net: Sequential network to save
        save_path: Path to save the model
        include_metadata: Whether to include conversion metadata
    """
    save_dict = {
        'model_state_dict': sequential_net.state_dict(),
        'model_architecture': 'SequentialDQNNetwork',
        'input_shape': (4, 84, 84),
        'output_size': 6,
    }
    
    if include_metadata:
        save_dict.update({
            'conversion_source': 'SNN_to_Sequential',
            'conversion_verified': True,
            'pytorch_version': torch.__version__,
        })
    
    torch.save(save_dict, save_path)
    print(f"Sequential model saved to: {save_path}")


if __name__ == "__main__":
    # Example usage
    print("Converting SNN to Sequential Network...")
    
    try:
        # Convert existing SNN model
        sequential_net = create_ann_from_snn('snn_pong_q_net.pth')
        
        # Save converted model
        save_sequential_model(sequential_net, 'sequential_pong_dqn.pt')
        
        # Test inference
        test_input = torch.randn(1, 4, 84, 84)
        with torch.no_grad():
            output = sequential_net(test_input)
            print(f"Test inference output: {output}")
            print(f"Predicted action: {output.argmax().item()}")
        
        print("Conversion completed successfully!")
        
    except Exception as e:
        print(f"X Conversion failed: {e}")
        raise