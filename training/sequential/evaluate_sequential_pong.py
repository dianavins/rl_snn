"""
Evaluate the converted Sequential DQN network on Pong environment
Should achieve +21 reward consistently (perfect Pong score)
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from collections import deque
import time

from conversion.sb3_to_sequential.sb3_to_sequential_converter import SequentialDQNNetwork

# Try to import gymnasium, fallback to gym if not available
try:
    import gymnasium as gym
    USE_GYMNASIUM = True
except ImportError:
    try:
        import gym
        USE_GYMNASIUM = False
        print("Using legacy gym instead of gymnasium")
    except ImportError:
        print("Neither gymnasium nor gym available. Please install one of them.")
        exit(1)


class PongPreprocessor:
    """Preprocess Pong frames consistently with training"""
    
    def __init__(self, frame_size=(84, 84)):
        self.frame_size = frame_size
        
    def preprocess_frame(self, frame):
        """Convert frame to grayscale and resize"""
        # Convert RGB to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
            
        # Resize to target size
        resized = cv2.resize(gray, self.frame_size, interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        return resized.astype(np.float32) / 255.0


def load_sequential_model(model_path='sequential_pong_dqn.pt'):
    """Load the converted Sequential DQN model"""
    print(f"Loading Sequential DQN model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Create network and load weights
    network = SequentialDQNNetwork(input_channels=4, n_actions=6)
    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval()
    
    print("Sequential DQN model loaded successfully!")
    print(f"Model architecture: {checkpoint.get('model_architecture', 'SequentialDQNNetwork')}")
    
    return network


def evaluate_on_pong(model, num_episodes=5, max_steps_per_episode=10000, render=False):
    """
    Evaluate the Sequential DQN model on Pong environment
    
    Args:
        model: Sequential DQN network
        num_episodes: Number of episodes to run
        max_steps_per_episode: Maximum steps per episode
        render: Whether to render the environment
        
    Returns:
        List of episode rewards
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING SEQUENTIAL DQN ON PONG")
    print(f"{'='*60}")
    
    # Create Pong environment
    if USE_GYMNASIUM:
        env = gym.make('PongNoFrameskip-v4', render_mode='human' if render else None)
    else:
        env = gym.make('PongNoFrameskip-v4')
        if render:
            env.render()
    
    preprocessor = PongPreprocessor()
    episode_rewards = []
    episode_lengths = []
    all_q_values = []
    
    print(f"Environment: PongNoFrameskip-v4")
    print(f"Action space: {env.action_space.n} actions")
    print(f"Episodes to run: {num_episodes}")
    print(f"Max steps per episode: {max_steps_per_episode}")
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        # Reset environment
        if USE_GYMNASIUM:
            state, info = env.reset()
        else:
            state = env.reset()
            
        state = preprocessor.preprocess_frame(state)
        
        # Initialize frame stack (4 frames)
        stacked_frames = deque([state] * 4, maxlen=4)
        state = np.stack(stacked_frames, axis=0)
        
        episode_reward = 0
        episode_length = 0
        episode_q_values = []
        
        start_time = time.time()
        
        for step in range(max_steps_per_episode):
            # Select action using the Sequential DQN
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
                q_values = model(state_tensor)
                action = q_values.argmax().item()
                
                # Store Q-values for analysis
                episode_q_values.append(q_values.cpu().numpy().flatten())
            
            # Take action in environment
            if USE_GYMNASIUM:
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            else:
                next_state, reward, done, info = env.step(action)
            
            # Preprocess next state
            next_state = preprocessor.preprocess_frame(next_state)
            stacked_frames.append(next_state)
            state = np.stack(stacked_frames, axis=0)
            
            episode_reward += reward
            episode_length += 1
            
            # Print progress every 1000 steps
            if step % 1000 == 0 and step > 0:
                print(f"  Step {step}: Current reward = {episode_reward}")
            
            if done:
                break
        
        end_time = time.time()
        episode_duration = end_time - start_time
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        all_q_values.extend(episode_q_values)
        
        # Episode summary
        print(f"  Final reward: {episode_reward}")
        print(f"  Episode length: {episode_length} steps")
        print(f"  Duration: {episode_duration:.2f} seconds")
        print(f"  Steps per second: {episode_length/episode_duration:.1f}")
        
        # Q-value statistics
        if episode_q_values:
            q_values_array = np.array(episode_q_values)
            mean_q_values = q_values_array.mean(axis=0)
            print(f"  Mean Q-values: {mean_q_values.round(3)}")
            print(f"  Most selected action: {q_values_array.argmax(axis=1).flatten()}")
    
    env.close()
    
    # Final statistics
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    
    print(f"Episode rewards: {episode_rewards}")
    print(f"Episode lengths: {episode_lengths}")
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)
    
    mean_length = np.mean(episode_lengths)
    
    print(f"\nPerformance Summary:")
    print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  Min reward: {min_reward}")
    print(f"  Max reward: {max_reward}")
    print(f"  Mean episode length: {mean_length:.1f} steps")
    
    # Check if achieving perfect Pong score (+21)
    perfect_episodes = sum(1 for r in episode_rewards if r >= 21)
    success_rate = perfect_episodes / num_episodes * 100
    
    print(f"\nPong Performance Analysis:")
    print(f"  Episodes with +21 reward: {perfect_episodes}/{num_episodes}")
    print(f"  Success rate: {success_rate:.1f}%")
    
    if all(r >= 21 for r in episode_rewards):
        print("  🎉 PERFECT! All episodes achieved +21 reward!")
    elif perfect_episodes > 0:
        print("  ✅ Good! Some episodes achieved +21 reward")
    else:
        print("  ⚠️  No episodes achieved +21 reward")
    
    # Q-value analysis
    if all_q_values:
        all_q_array = np.array(all_q_values)
        print(f"\nQ-Value Analysis:")
        print(f"  Mean Q-values across all steps: {all_q_array.mean(axis=0).round(3)}")
        print(f"  Q-value ranges: {all_q_array.min(axis=0).round(3)} to {all_q_array.max(axis=0).round(3)}")
    
    return episode_rewards, episode_lengths


def main():
    """Main evaluation function"""
    print("Sequential DQN Pong Evaluation")
    print("Expected performance: +21 reward on all episodes")
    
    try:
        # Load the Sequential DQN model
        model = load_sequential_model('sequential_pong_dqn.pt')
        
        # Evaluate on Pong
        rewards, lengths = evaluate_on_pong(
            model, 
            num_episodes=5,  # Test on 5 episodes
            max_steps_per_episode=10000,
            render=False  # Set to True to watch the games
        )
        
        # Final assessment
        print(f"\n{'='*60}")
        print("FINAL ASSESSMENT")
        print(f"{'='*60}")
        
        perfect_count = sum(1 for r in rewards if r >= 21)
        
        if perfect_count == len(rewards):
            print("🎉 SUCCESS: All episodes achieved +21 reward!")
            print("   The Sequential DQN conversion is working perfectly!")
        elif perfect_count > len(rewards) * 0.8:
            print("✅ VERY GOOD: Most episodes achieved +21 reward")
            print("   The Sequential DQN is performing well")
        elif perfect_count > 0:
            print("⚠️  PARTIAL: Some episodes achieved +21 reward")
            print("   The Sequential DQN may need further tuning")
        else:
            print("❌ ISSUE: No episodes achieved +21 reward")
            print("   There may be an issue with the conversion or evaluation")
        
        return rewards
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()