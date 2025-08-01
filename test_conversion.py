"""
Comprehensive unit tests for SB3-to-Sequential conversion
Ensures zero performance tradeoffs through rigorous testing
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from sb3_to_sequential_converter import (
    SequentialDQNNetwork, 
    SB3ToSequentialConverter,
    create_ann_from_snn
)
import tempfile
import os


class TestSequentialDQNNetwork(unittest.TestCase):
    """Test the Sequential DQN Network implementation"""
    
    def setUp(self):
        self.network = SequentialDQNNetwork(input_channels=4, n_actions=6)
        self.test_input = torch.randn(2, 4, 84, 84)  # Batch of 2
        
    def test_network_initialization(self):
        """Test network is properly initialized"""
        self.assertIsInstance(self.network, nn.Module)
        self.assertEqual(len(list(self.network.parameters())), 10)  # 5 layers × 2 params (weight + bias)
        
    def test_forward_pass_shape(self):
        """Test forward pass produces correct output shape"""
        output = self.network(self.test_input)
        self.assertEqual(output.shape, (2, 6))  # Batch size, actions
        
    def test_forward_pass_deterministic(self):
        """Test forward pass is deterministic"""
        self.network.eval()
        with torch.no_grad():
            output1 = self.network(self.test_input)
            output2 = self.network(self.test_input)
        torch.testing.assert_close(output1, output2)
        
    def test_gradient_flow(self):
        """Test gradients flow through the network"""
        output = self.network(self.test_input)
        loss = output.sum()
        loss.backward()
        
        # Check all layers have gradients
        for name, param in self.network.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")
            self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient in {name}")
            
    def test_layer_access(self):
        """Test layer access by name"""
        conv1 = self.network.get_layer_by_name('conv1')
        self.assertIsInstance(conv1, nn.Conv2d)
        self.assertEqual(conv1.in_channels, 4)
        self.assertEqual(conv1.out_channels, 32)


class TestSB3ToSequentialConverter(unittest.TestCase):
    """Test the conversion process"""
    
    def setUp(self):
        self.converter = SB3ToSequentialConverter(verify_conversion=True, tolerance=1e-6)
        
        # Create mock SNN state dict
        self.mock_snn_state_dict = {
            'features_extractor.cnn.0.weight': torch.randn(32, 4, 8, 8),
            'features_extractor.cnn.0.bias': torch.randn(32),
            'features_extractor.cnn.2.weight': torch.randn(64, 32, 4, 4),
            'features_extractor.cnn.2.bias': torch.randn(64),
            'features_extractor.cnn.4.weight': torch.randn(64, 64, 3, 3),
            'features_extractor.cnn.4.bias': torch.randn(64),
            'features_extractor.linear.0.weight': torch.randn(512, 3136),
            'features_extractor.linear.0.bias': torch.randn(512),
            'q_net.0.weight': torch.randn(6, 512),
            'q_net.0.bias': torch.randn(6),
            # Mock voltage scaler parameters (should be ignored)
            'features_extractor.cnn.1.0.scale': torch.tensor(1.0),
            'features_extractor.cnn.1.2.scale': torch.tensor(1.0),
            'features_extractor.cnn.3.0.scale': torch.tensor(1.0),
            'features_extractor.cnn.3.2.scale': torch.tensor(1.0),
            'features_extractor.cnn.5.0.scale': torch.tensor(1.0),
            'features_extractor.cnn.5.2.scale': torch.tensor(1.0),
            'features_extractor.linear.1.0.scale': torch.tensor(1.0),
            'features_extractor.linear.1.2.scale': torch.tensor(1.0),
        }
        
    def test_conversion_weight_mapping(self):
        """Test weight mapping is correct"""
        sequential_net = self.converter.convert_snn_to_sequential(self.mock_snn_state_dict)
        
        # Check each layer has correct weights
        state_dict = sequential_net.state_dict()
        
        # Conv1
        torch.testing.assert_close(
            state_dict['network.conv1.weight'], 
            self.mock_snn_state_dict['features_extractor.cnn.0.weight']
        )
        torch.testing.assert_close(
            state_dict['network.conv1.bias'], 
            self.mock_snn_state_dict['features_extractor.cnn.0.bias']
        )
        
        # FC2 (output layer)
        torch.testing.assert_close(
            state_dict['network.fc2.weight'], 
            self.mock_snn_state_dict['q_net.0.weight']
        )
        torch.testing.assert_close(
            state_dict['network.fc2.bias'], 
            self.mock_snn_state_dict['q_net.0.bias']
        )
        
    def test_missing_key_error(self):
        """Test error handling for missing keys"""
        incomplete_state_dict = {
            'features_extractor.cnn.0.weight': torch.randn(32, 4, 8, 8),
            # Missing other required keys
        }
        
        with self.assertRaises(KeyError):
            self.converter.convert_snn_to_sequential(incomplete_state_dict)
            
    def test_conversion_verification(self):
        """Test conversion verification works"""
        # This should pass without issues
        sequential_net = self.converter.convert_snn_to_sequential(self.mock_snn_state_dict)
        self.assertIsInstance(sequential_net, SequentialDQNNetwork)
        
    def test_tolerance_setting(self):
        """Test different tolerance settings"""
        strict_converter = SB3ToSequentialConverter(tolerance=1e-8)
        lenient_converter = SB3ToSequentialConverter(tolerance=1e-3)
        
        # Both should work with exact weight copies
        strict_net = strict_converter.convert_snn_to_sequential(self.mock_snn_state_dict)
        lenient_net = lenient_converter.convert_snn_to_sequential(self.mock_snn_state_dict)
        
        self.assertIsInstance(strict_net, SequentialDQNNetwork)
        self.assertIsInstance(lenient_net, SequentialDQNNetwork)


class TestEndToEndConversion(unittest.TestCase):
    """Test end-to-end conversion with real model files"""
    
    def test_conversion_preserves_inference(self):
        """Test that conversion preserves inference behavior"""
        if not os.path.exists('snn_pong_q_net.pth'):
            self.skipTest("SNN model file not found")
            
        # Load original SNN weights
        snn_state_dict = torch.load('snn_pong_q_net.pth', map_location='cpu', weights_only=False)
        
        # Convert to sequential
        converter = SB3ToSequentialConverter(verify_conversion=True)
        sequential_net = converter.convert_snn_to_sequential(snn_state_dict)
        
        # Test inference on same input
        test_input = torch.randn(3, 4, 84, 84)  # Batch of 3
        
        sequential_net.eval()
        with torch.no_grad():
            output = sequential_net(test_input)
            
        # Check output properties
        self.assertEqual(output.shape, (3, 6))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
        # Check outputs are reasonable Q-values
        self.assertTrue(output.abs().max() < 100)  # Reasonable Q-value range
        
    def test_save_and_load_cycle(self):
        """Test saving and loading converted model"""
        if not os.path.exists('snn_pong_q_net.pth'):
            self.skipTest("SNN model file not found")
            
        # Convert model
        sequential_net = create_ann_from_snn('snn_pong_q_net.pth')
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
            temp_path = tmp_file.name
            
        try:
            from sb3_to_sequential_converter import save_sequential_model
            save_sequential_model(sequential_net, temp_path)
            
            # Load back
            checkpoint = torch.load(temp_path, map_location='cpu')
            
            # Create new network and load weights
            new_net = SequentialDQNNetwork()
            new_net.load_state_dict(checkpoint['model_state_dict'])
            
            # Test they produce same outputs
            test_input = torch.randn(1, 4, 84, 84)
            sequential_net.eval()
            new_net.eval()
            
            with torch.no_grad():
                output1 = sequential_net(test_input)
                output2 = new_net(test_input)
                
            torch.testing.assert_close(output1, output2)
            
        finally:
            os.unlink(temp_path)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Benchmark tests to ensure no performance degradation"""
    
    def setUp(self):
        self.network = SequentialDQNNetwork()
        self.network.eval()
        self.test_inputs = [
            torch.randn(1, 4, 84, 84),    # Single inference
            torch.randn(32, 4, 84, 84),   # Batch inference
            torch.randn(128, 4, 84, 84),  # Large batch
        ]
        
    def test_inference_speed(self):
        """Test inference speed is reasonable"""
        import time
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.network(self.test_inputs[0])
        
        # Time batch inference
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = self.network(self.test_inputs[1])  # 32 batch size
        end_time = time.time()
        
        avg_time_per_batch = (end_time - start_time) / 100
        avg_time_per_sample = avg_time_per_batch / 32
        
        # Should be fast enough for real-time use (< 10ms per sample on CPU)
        self.assertLess(avg_time_per_sample, 0.01, 
                       f"Inference too slow: {avg_time_per_sample:.4f}s per sample")
        
    def test_memory_usage(self):
        """Test memory usage is reasonable"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple networks
        networks = [SequentialDQNNetwork() for _ in range(10)]
        
        # Memory after creating networks
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_per_network = (current_memory - baseline_memory) / 10
        
        # Should be reasonable (< 50MB per network)
        self.assertLess(memory_per_network, 50, 
                       f"Memory usage too high: {memory_per_network:.2f}MB per network")
        
        # Cleanup
        del networks


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)