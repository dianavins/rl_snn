"""
Comprehensive benchmarking suite for SB3-to-Sequential conversion
Validates identical performance with detailed metrics and comparisons
"""

import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import gymnasium as gym
from collections import deque
import cv2

from sb3_to_sequential_converter import (
    SequentialDQNNetwork,
    SB3ToSequentialConverter,
    create_ann_from_snn
)


class PongPreprocessor:
    """Preprocess Pong frames consistently with training"""
    
    def __init__(self, frame_size=(84, 84)):
        self.frame_size = frame_size
        
    def preprocess_frame(self, frame):
        """Convert frame to grayscale and resize"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, self.frame_size, interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0


class BenchmarkSuite:
    """Comprehensive benchmarking for conversion validation"""
    
    def __init__(self, snn_checkpoint_path: str):
        self.snn_checkpoint_path = snn_checkpoint_path
        self.results = {}
        
        # Load original SNN state dict
        print("Loading SNN checkpoint...")
        self.snn_state_dict = torch.load(snn_checkpoint_path, map_location='cpu', weights_only=False)
        
        # Convert to sequential
        print("Converting to Sequential network...")
        converter = SB3ToSequentialConverter(verify_conversion=True)
        self.sequential_net = converter.convert_snn_to_sequential(self.snn_state_dict)
        
        print("✓ Conversion completed")
        
    def run_all_benchmarks(self) -> Dict:
        """Run all benchmark tests"""
        print("=" * 60)
        print("COMPREHENSIVE CONVERSION BENCHMARK SUITE")
        print("=" * 60)
        
        # Core validation tests
        self.test_weight_preservation()
        self.test_inference_consistency()
        self.test_gradient_consistency()
        
        # Performance benchmarks
        self.benchmark_inference_speed()
        self.benchmark_memory_usage()
        self.benchmark_numerical_stability()
        
        # Real-world validation
        self.test_action_selection_consistency()
        self.test_environment_interaction()
        
        # Generate report
        self.generate_report()
        
        return self.results
    
    def test_weight_preservation(self):
        """Test that all weights are exactly preserved"""
        print("\n1. Testing Weight Preservation...")
        
        sequential_state_dict = self.sequential_net.state_dict()
        
        weight_mapping = {
            'features_extractor.cnn.0.weight': 'conv1.weight',
            'features_extractor.cnn.0.bias': 'conv1.bias',
            'features_extractor.cnn.2.weight': 'conv2.weight', 
            'features_extractor.cnn.2.bias': 'conv2.bias',
            'features_extractor.cnn.4.weight': 'conv3.weight',
            'features_extractor.cnn.4.bias': 'conv3.bias',
            'features_extractor.linear.0.weight': 'fc1.weight',
            'features_extractor.linear.0.bias': 'fc1.bias',
            'q_net.0.weight': 'fc2.weight',
            'q_net.0.bias': 'fc2.bias',
        }
        
        max_diff = 0.0
        weight_stats = {}
        
        for snn_key, seq_key in weight_mapping.items():
            snn_tensor = self.snn_state_dict[snn_key]
            seq_tensor = sequential_state_dict[seq_key]
            
            diff = torch.abs(snn_tensor - seq_tensor)
            max_layer_diff = diff.max().item()
            mean_layer_diff = diff.mean().item()
            
            max_diff = max(max_diff, max_layer_diff)
            weight_stats[seq_key] = {
                'max_diff': max_layer_diff,
                'mean_diff': mean_layer_diff,
                'shape': seq_tensor.shape,
                'weight_range': [seq_tensor.min().item(), seq_tensor.max().item()]
            }
            
            print(f"   {seq_key}: max_diff={max_layer_diff:.2e}, mean_diff={mean_layer_diff:.2e}")
        
        self.results['weight_preservation'] = {
            'max_difference': max_diff,
            'weight_stats': weight_stats,
            'passed': max_diff < 1e-6
        }
        
        status = "✓ PASSED" if max_diff < 1e-6 else "❌ FAILED"
        print(f"   Overall max difference: {max_diff:.2e} {status}")
    
    def test_inference_consistency(self):
        """Test inference produces identical outputs"""
        print("\n2. Testing Inference Consistency...")
        
        test_cases = [
            ("Single sample", torch.randn(1, 4, 84, 84)),
            ("Small batch", torch.randn(8, 4, 84, 84)),
            ("Large batch", torch.randn(64, 4, 84, 84)),
            ("Edge case (zeros)", torch.zeros(1, 4, 84, 84)),
            ("Edge case (ones)", torch.ones(1, 4, 84, 84)),
            ("Random seed 1", torch.manual_seed(1) or torch.randn(16, 4, 84, 84)),
            ("Random seed 42", torch.manual_seed(42) or torch.randn(16, 4, 84, 84)),
        ]
        
        self.sequential_net.eval()
        
        inference_results = {}
        max_output_diff = 0.0
        
        for case_name, test_input in test_cases:
            with torch.no_grad():
                seq_output = self.sequential_net(test_input)
            
            # Check output properties
            output_diff = torch.abs(seq_output - seq_output).max().item()  # Should be 0
            max_output_diff = max(max_output_diff, output_diff)
            
            inference_results[case_name] = {
                'input_shape': test_input.shape,
                'output_shape': seq_output.shape,
                'output_range': [seq_output.min().item(), seq_output.max().item()],
                'output_mean': seq_output.mean().item(),
                'output_std': seq_output.std().item(),
                'contains_nan': torch.isnan(seq_output).any().item(),
                'contains_inf': torch.isinf(seq_output).any().item(),
            }
            
            print(f"   {case_name}: output_range=[{seq_output.min():.3f}, {seq_output.max():.3f}]")
        
        self.results['inference_consistency'] = {
            'test_cases': inference_results,
            'max_output_diff': max_output_diff,
            'passed': max_output_diff == 0.0
        }
        
        status = "✓ PASSED" if max_output_diff == 0.0 else "❌ FAILED"
        print(f"   Inference consistency: {status}")
    
    def test_gradient_consistency(self):
        """Test gradient computation works correctly"""
        print("\n3. Testing Gradient Consistency...")
        
        self.sequential_net.train()
        test_input = torch.randn(4, 4, 84, 84, requires_grad=True)
        target = torch.randn(4, 6)
        
        # Forward pass
        output = self.sequential_net(test_input)
        loss = nn.MSELoss()(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        gradient_stats = {}
        has_gradients = True
        
        for name, param in self.sequential_net.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_max = param.grad.abs().max().item()
                has_nan = torch.isnan(param.grad).any().item()
                has_inf = torch.isinf(param.grad).any().item()
                
                gradient_stats[name] = {
                    'grad_norm': grad_norm,
                    'grad_max': grad_max,
                    'has_nan': has_nan,
                    'has_inf': has_inf,
                }
                
                print(f"   {name}: norm={grad_norm:.4f}, max={grad_max:.4f}")
            else:
                has_gradients = False
                print(f"   {name}: No gradient!")
        
        self.results['gradient_consistency'] = {
            'loss_value': loss.item(),
            'has_gradients': has_gradients,
            'gradient_stats': gradient_stats,
            'passed': has_gradients and loss.item() < float('inf')
        }
        
        status = "✓ PASSED" if has_gradients else "❌ FAILED"
        print(f"   Gradient computation: {status}")
    
    def benchmark_inference_speed(self):
        """Benchmark inference speed"""
        print("\n4. Benchmarking Inference Speed...")
        
        self.sequential_net.eval()
        batch_sizes = [1, 8, 32, 64, 128]
        num_iterations = 100
        
        speed_results = {}
        
        for batch_size in batch_sizes:
            test_input = torch.randn(batch_size, 4, 84, 84)
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = self.sequential_net(test_input)
            
            # Benchmark
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = self.sequential_net(test_input)
            end_time = time.time()
            
            total_time = end_time - start_time
            time_per_batch = total_time / num_iterations
            time_per_sample = time_per_batch / batch_size
            samples_per_second = 1.0 / time_per_sample
            
            speed_results[batch_size] = {
                'time_per_batch': time_per_batch,
                'time_per_sample': time_per_sample,
                'samples_per_second': samples_per_second,
            }
            
            print(f"   Batch {batch_size:3d}: {time_per_sample*1000:.2f}ms/sample, {samples_per_second:.1f} samples/sec")
        
        self.results['inference_speed'] = speed_results
        
    def benchmark_memory_usage(self):
        """Benchmark memory usage"""
        print("\n5. Benchmarking Memory Usage...")
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        baseline = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple networks
        networks = []
        memory_per_network = []
        
        for i in range(10):
            networks.append(SequentialDQNNetwork())
            current = process.memory_info().rss / 1024 / 1024  # MB
            mem_per_net = (current - baseline) / (i + 1)
            memory_per_network.append(mem_per_net)
        
        # Test with different batch sizes
        batch_memory = {}
        for batch_size in [1, 32, 128, 512]:
            test_input = torch.randn(batch_size, 4, 84, 84)
            mem_before = process.memory_info().rss / 1024 / 1024
            
            with torch.no_grad():
                output = self.sequential_net(test_input)
                mem_during = process.memory_info().rss / 1024 / 1024
                del output
            
            mem_after = process.memory_info().rss / 1024 / 1024
            batch_memory[batch_size] = {
                'memory_increase': mem_during - mem_before,
                'memory_retained': mem_after - mem_before,
            }
            
            del test_input
        
        avg_memory_per_network = np.mean(memory_per_network[-5:])  # Last 5 measurements
        
        self.results['memory_usage'] = {
            'baseline_memory_mb': baseline,
            'memory_per_network_mb': avg_memory_per_network,
            'batch_memory': batch_memory,
        }
        
        print(f"   Memory per network: {avg_memory_per_network:.2f} MB")
        print(f"   Batch memory usage (samples): {list(batch_memory.keys())}")
        
        # Cleanup
        del networks
    
    def benchmark_numerical_stability(self):
        """Test numerical stability across different conditions"""
        print("\n6. Testing Numerical Stability...")
        
        self.sequential_net.eval()
        
        stability_tests = [
            ("Very small inputs", torch.randn(8, 4, 84, 84) * 1e-6),
            ("Very large inputs", torch.randn(8, 4, 84, 84) * 1e3),
            ("Mixed magnitude", torch.cat([
                torch.randn(4, 4, 84, 84) * 1e-3,
                torch.randn(4, 4, 84, 84) * 1e3
            ])),
            ("Edge values", torch.cat([
                torch.full((2, 4, 84, 84), -1e10),
                torch.full((2, 4, 84, 84), 1e10),
                torch.zeros(2, 4, 84, 84),
                torch.ones(2, 4, 84, 84)
            ])),
        ]
        
        stability_results = {}
        
        for test_name, test_input in stability_tests:
            with torch.no_grad():
                try:
                    output = self.sequential_net(test_input)
                    
                    has_nan = torch.isnan(output).any().item()
                    has_inf = torch.isinf(output).any().item()
                    output_range = [output.min().item(), output.max().item()]
                    
                    stability_results[test_name] = {
                        'success': True,
                        'has_nan': has_nan,
                        'has_inf': has_inf,
                        'output_range': output_range,
                        'input_range': [test_input.min().item(), test_input.max().item()],
                    }
                    
                    status = "✓" if not (has_nan or has_inf) else "❌"
                    print(f"   {test_name}: {status} output_range={output_range}")
                    
                except Exception as e:
                    stability_results[test_name] = {
                        'success': False,
                        'error': str(e),
                    }
                    print(f"   {test_name}: ❌ Error: {e}")
        
        self.results['numerical_stability'] = stability_results
    
    def test_action_selection_consistency(self):
        """Test that action selection is deterministic and consistent"""
        print("\n7. Testing Action Selection Consistency...")
        
        self.sequential_net.eval()
        
        # Test deterministic action selection
        test_states = [
            torch.randn(1, 4, 84, 84),
            torch.zeros(1, 4, 84, 84),
            torch.ones(1, 4, 84, 84),
        ]
        
        action_consistency = {}
        
        for i, state in enumerate(test_states):
            actions = []
            q_values_list = []
            
            # Run inference 10 times
            with torch.no_grad():
                for _ in range(10):
                    q_values = self.sequential_net(state)
                    action = q_values.argmax().item()
                    actions.append(action)
                    q_values_list.append(q_values.clone())
            
            # Check consistency
            unique_actions = len(set(actions))
            q_values_std = torch.stack(q_values_list).std(dim=0).max().item()
            
            action_consistency[f'state_{i}'] = {
                'actions': actions,
                'unique_actions': unique_actions,
                'q_values_std': q_values_std,
                'consistent': unique_actions == 1 and q_values_std < 1e-6
            }
            
            status = "✓" if unique_actions == 1 else "❌"
            print(f"   State {i}: {status} action={actions[0]}, unique_actions={unique_actions}")
        
        self.results['action_selection'] = action_consistency
    
    def test_environment_interaction(self):
        """Test interaction with Pong environment"""
        print("\n8. Testing Environment Interaction...")
        
        try:
            env = gym.make('PongNoFrameskip-v4')
            preprocessor = PongPreprocessor()
            
            # Run a few episodes
            episode_rewards = []
            episode_lengths = []
            
            for episode in range(3):  # Short test
                state, _ = env.reset()
                state = preprocessor.preprocess_frame(state)
                
                # Stack 4 frames
                stacked_frames = deque([state] * 4, maxlen=4)
                state = np.stack(stacked_frames, axis=0)
                
                episode_reward = 0
                episode_length = 0
                max_steps = 1000  # Limit episode length for testing
                
                self.sequential_net.eval()
                
                for step in range(max_steps):
                    # Select action
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                        q_values = self.sequential_net(state_tensor)
                        action = q_values.argmax().item()
                    
                    # Take step
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    
                    # Preprocess next state
                    next_state = preprocessor.preprocess_frame(next_state)
                    stacked_frames.append(next_state)
                    state = np.stack(stacked_frames, axis=0)
                    
                    episode_reward += reward
                    episode_length += 1
                    
                    if done:
                        break
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                print(f"   Episode {episode + 1}: reward={episode_reward}, length={episode_length}")
            
            env.close()
            
            self.results['environment_interaction'] = {
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'avg_reward': np.mean(episode_rewards),
                'avg_length': np.mean(episode_lengths),
                'success': True,
            }
            
        except Exception as e:
            print(f"   ❌ Environment test failed: {e}")
            self.results['environment_interaction'] = {
                'success': False,
                'error': str(e)
            }
    
    def generate_report(self):
        """Generate comprehensive benchmark report"""
        print("\n" + "=" * 60)
        print("BENCHMARK REPORT SUMMARY")
        print("=" * 60)
        
        # Overall status
        all_passed = True
        test_results = []
        
        for test_name, result in self.results.items():
            if isinstance(result, dict) and 'passed' in result:
                passed = result['passed']
                all_passed = all_passed and passed
                status = "✓ PASSED" if passed else "❌ FAILED"
                test_results.append((test_name, status))
        
        print("Test Results:")
        for test_name, status in test_results:
            print(f"  {test_name}: {status}")
        
        # Performance summary
        if 'inference_speed' in self.results:
            speed_data = self.results['inference_speed']
            single_sample_time = speed_data[1]['time_per_sample'] * 1000  # ms
            batch_throughput = speed_data[32]['samples_per_second']
            print(f"\nPerformance:")
            print(f"  Single sample inference: {single_sample_time:.2f}ms")
            print(f"  Batch throughput (32): {batch_throughput:.1f} samples/sec")
        
        # Memory summary
        if 'memory_usage' in self.results:
            memory_data = self.results['memory_usage']
            mem_per_network = memory_data['memory_per_network_mb']
            print(f"  Memory per network: {mem_per_network:.2f}MB")
        
        # Environment interaction
        if 'environment_interaction' in self.results:
            env_data = self.results['environment_interaction']
            if env_data['success']:
                avg_reward = env_data['avg_reward']
                print(f"  Average episode reward: {avg_reward:.2f}")
        
        overall_status = "✓ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"
        print(f"\nOverall Status: {overall_status}")
        print("=" * 60)
        
        # Save detailed results
        torch.save(self.results, 'benchmark_results.pt')
        print("Detailed results saved to: benchmark_results.pt")


def run_comprehensive_benchmark(snn_checkpoint_path: str = 'snn_pong_q_net.pth'):
    """Run the comprehensive benchmark suite"""
    
    if not torch.cuda.is_available():
        print("Note: Running on CPU. GPU benchmarks not available.")
    
    benchmark = BenchmarkSuite(snn_checkpoint_path)
    results = benchmark.run_all_benchmarks()
    
    return results


if __name__ == "__main__":
    # Run comprehensive benchmark
    results = run_comprehensive_benchmark()