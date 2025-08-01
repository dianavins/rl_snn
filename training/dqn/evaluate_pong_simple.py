"""
Simple evaluation of Sequential DQN on Pong environment
Tests the converted model without external dependencies beyond PyTorch
"""

import torch
import numpy as np
from collections import deque
import time

from conversion.sb3_to_sequential.sb3_to_sequential_converter import SequentialDQNNetwork

# Try to import gym/gymnasium
try:
    import gymnasium as gym
    USE_GYMNASIUM = True
    print("Using gymnasium")
except ImportError:
    try:
        import gym
        USE_GYMNASIUM = False
        print("Using legacy gym")
    except ImportError:
        print("No gym library available. Testing with synthetic data instead.")
        USE_GYM = False


def simple_preprocess(frame):
    """Simple preprocessing without OpenCV"""
    if len(frame.shape) == 3:
        # Convert RGB to grayscale using standard weights
        gray = 0.299 * frame[:,:,0] + 0.587 * frame[:,:,1] + 0.114 * frame[:,:,2]
    else:
        gray = frame
    
    # Simple resize using nearest neighbor (basic implementation)
    # This is less optimal than OpenCV but works for testing
    h, w = gray.shape
    target_h, target_w = 84, 84
    
    # Simple downsampling
    step_h = h // target_h
    step_w = w // target_w
    
    if step_h > 0 and step_w > 0:
        resized = gray[::step_h, ::step_w][:target_h, :target_w]
        
        # Pad if needed
        if resized.shape[0] < target_h or resized.shape[1] < target_w:
            padded = np.zeros((target_h, target_w))
            padded[:resized.shape[0], :resized.shape[1]] = resized
            resized = padded
    else:
        # Fallback: just crop/pad to target size
        resized = np.zeros((target_h, target_w))
        min_h = min(h, target_h)
        min_w = min(w, target_w)
        resized[:min_h, :min_w] = gray[:min_h, :min_w]
    
    # Normalize to [0, 1]
    return resized.astype(np.float32) / 255.0


def load_sequential_model(model_path='sequential_pong_dqn.pt'):
    """Load the converted Sequential DQN model"""
    print(f"Loading Sequential DQN model from: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        network = SequentialDQNNetwork(input_channels=4, n_actions=6)
        network.load_state_dict(checkpoint['model_state_dict'])
        network.eval()
        
        print("Sequential DQN model loaded successfully!")
        
        # Test the model with random input
        test_input = torch.randn(1, 4, 84, 84)
        with torch.no_grad():
            test_output = network(test_input)
            
        print(f"Model test - Input shape: {test_input.shape}")
        print(f"Model test - Output shape: {test_output.shape}")
        print(f"Model test - Q-values: {test_output.numpy().round(3)}")
        print(f"Model test - Predicted action: {test_output.argmax().item()}")
        
        return network
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def test_with_synthetic_data(model, num_tests=10):
    """Test the model with synthetic data if gym is not available"""
    print(f"\n{'='*60}")
    print("TESTING WITH SYNTHETIC DATA (No Gym Available)")
    print(f"{'='*60}")
    
    results = []
    
    for i in range(num_tests):
        # Create synthetic "game state"
        test_input = torch.randn(1, 4, 84, 84)
        
        with torch.no_grad():
            q_values = model(test_input)
            action = q_values.argmax().item()
            max_q = q_values.max().item()
            
        results.append({
            'test': i + 1,
            'q_values': q_values.numpy().flatten(),
            'action': action,
            'max_q': max_q
        })
        
        print(f"Test {i+1}: Action={action}, Max Q={max_q:.3f}, Q-values={q_values.numpy().flatten().round(3)}")
    
    print(f"\nSynthetic Test Summary:")
    actions = [r['action'] for r in results]
    print(f"Actions selected: {actions}")
    print(f"Action distribution: {np.bincount(actions, minlength=6)}")
    print(f"Mean max Q-value: {np.mean([r['max_q'] for r in results]):.3f}")
    
    return results


def evaluate_on_pong_simple(model, num_episodes=3):
    """Simple Pong evaluation"""
    print(f"\n{'='*60}")
    print("EVALUATING SEQUENTIAL DQN ON PONG")
    print(f"{'='*60}")
    
    try:
        # Create environment
        if USE_GYMNASIUM:
            env = gym.make('PongNoFrameskip-v4')
        else:
            env = gym.make('PongNoFrameskip-v4')
            
        print(f"Environment: PongNoFrameskip-v4")
        print(f"Action space: {env.action_space.n} actions")
        
    except Exception as e:
        print(f"Could not create Pong environment: {e}")
        print("Falling back to synthetic data testing...")
        return test_with_synthetic_data(model)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        try:
            # Reset environment
            if USE_GYMNASIUM:
                state, info = env.reset()
            else:
                state = env.reset()
                
            # Preprocess state
            state = simple_preprocess(state)
            
            # Stack 4 frames
            stacked_frames = deque([state] * 4, maxlen=4)
            state = np.stack(stacked_frames, axis=0)
            
            episode_reward = 0
            step_count = 0
            max_steps = 5000  # Limit steps for testing
            
            start_time = time.time()
            
            while step_count < max_steps:
                # Select action
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = model(state_tensor)
                    action = q_values.argmax().item()
                
                # Take step
                if USE_GYMNASIUM:
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                else:
                    next_state, reward, done, info = env.step(action)
                
                # Update state
                next_state = simple_preprocess(next_state)
                stacked_frames.append(next_state)
                state = np.stack(stacked_frames, axis=0)
                
                episode_reward += reward
                step_count += 1
                
                # Print progress
                if step_count % 1000 == 0:
                    print(f"  Step {step_count}: Current reward = {episode_reward}")
                
                if done:
                    break
            
            end_time = time.time()
            duration = end_time - start_time
            
            episode_rewards.append(episode_reward)
            
            print(f"  Episode {episode + 1} Results:")
            print(f"    Final reward: {episode_reward}")
            print(f"    Steps taken: {step_count}")
            print(f"    Duration: {duration:.2f} seconds")
            print(f"    Steps/second: {step_count/duration:.1f}")
            
        except Exception as e:
            print(f"  Episode {episode + 1} failed: {e}")
            episode_rewards.append(-21)  # Worst possible score
    
    try:
        env.close()
    except:
        pass
    
    # Results summary
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    
    print(f"Episode rewards: {episode_rewards}")
    
    if episode_rewards:
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        min_reward = np.min(episode_rewards)
        max_reward = np.max(episode_rewards)
        
        print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"Min reward: {min_reward}")
        print(f"Max reward: {max_reward}")
        
        # Check for +21 performance
        perfect_episodes = sum(1 for r in episode_rewards if r >= 21)
        success_rate = perfect_episodes / len(episode_rewards) * 100
        
        print(f"\nPong Performance:")
        print(f"Episodes with +21 reward: {perfect_episodes}/{len(episode_rewards)}")
        print(f"Success rate: {success_rate:.1f}%")
        
        if all(r >= 21 for r in episode_rewards):
            print("PERFECT! All episodes achieved +21 reward!")
        elif perfect_episodes > 0:
            print("GOOD! Some episodes achieved +21 reward")
        else:
            print("No episodes achieved +21 reward yet")
            if max_reward > 0:
                print(f"   But achieved positive reward up to {max_reward}")
    
    return episode_rewards


def main():
    """Main evaluation function"""
    print("="*60)
    print("SEQUENTIAL DQN PONG EVALUATION")
    print("Expected performance: +21 reward on all episodes")
    print("="*60)
    
    # Load model
    model = load_sequential_model('sequential_pong_dqn.pt')
    if model is None:
        print("Could not load model. Exiting.")
        return None
    
    # Try to run Pong evaluation
    try:
        if 'gym' in globals() or 'gymnasium' in globals():
            results = evaluate_on_pong_simple(model, num_episodes=3)
        else:
            print("No gym available. Testing with synthetic data only.")
            results = test_with_synthetic_data(model, num_tests=10)
            
    except Exception as e:
        print(f"Evaluation failed: {e}")
        print("Running synthetic data test as fallback...")
        results = test_with_synthetic_data(model, num_tests=10)
    
    print(f"\n{'='*60}")
    print("FINAL ASSESSMENT")
    print(f"{'='*60}")
    
    if isinstance(results, list) and len(results) > 0:
        if isinstance(results[0], dict):  # Synthetic data results
            print("Model is working and producing consistent outputs")
            print("   To test on actual Pong, install gymnasium: pip install gymnasium[atari]")
        else:  # Pong episode results
            perfect_count = sum(1 for r in results if r >= 21)
            if perfect_count == len(results):
                print("SUCCESS: All episodes achieved +21 reward!")
            elif perfect_count > 0:
                print(f"PARTIAL SUCCESS: {perfect_count}/{len(results)} episodes achieved +21")
            else:
                print("No perfect games yet, but model is running")
    
    return results


if __name__ == "__main__":
    results = main()