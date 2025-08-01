#!/usr/bin/env python3
"""Quick training of Sequential DQN to achieve decent performance"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from conversion.sb3_to_sequential.sb3_to_sequential_converter import SequentialDQNNetwork
import ale_py

class SimpleReplayBuffer:
    """Simple replay buffer for DQN training"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

def quick_train_sequential():
    """Quick training of Sequential DQN"""
    print("=== Quick Training Sequential DQN ===")
    
    # Register ALE
    try:
        ale_py.register_all()
    except AttributeError:
        pass
    
    # Create environment
    env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)
    print("SUCCESS: Training environment created")
    
    # Create model
    model = SequentialDQNNetwork()
    target_model = SequentialDQNNetwork()
    target_model.load_state_dict(model.state_dict())
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    replay_buffer = SimpleReplayBuffer(10000)
    
    print("SUCCESS: Model and training setup created")
    
    # Training parameters
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    batch_size = 32
    target_update_freq = 1000
    
    episode = 0
    step_count = 0
    best_avg_reward = -float('inf')
    
    print("\n=== Starting Quick Training ===")
    print("Training for fast convergence (this may take a few episodes)")
    
    # Training loop
    for episode in range(50):  # Quick training
        obs = env.reset()
        episode_reward = 0
        done = False
        episode_steps = 0
        
        while not done and episode_steps < 2000:
            # Preprocess observation
            if len(obs.shape) == 4 and obs.shape[-1] == 4:
                obs_tensor = torch.FloatTensor(obs).permute(0, 3, 1, 2) / 255.0
            else:
                obs_tensor = torch.FloatTensor(obs) / 255.0
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = model(obs_tensor)
                    action = q_values.argmax().item()
            
            # Take step
            next_obs, reward, done, info = env.step([action])
            
            # Store transition
            if len(obs.shape) == 4 and obs.shape[-1] == 4:
                obs_np = obs[0].transpose(2, 0, 1)  # HWC -> CHW
                next_obs_np = next_obs[0].transpose(2, 0, 1) if not done else np.zeros_like(obs_np)
            else:
                obs_np = obs[0]
                next_obs_np = next_obs[0] if not done else np.zeros_like(obs_np)
            
            replay_buffer.push(
                obs_np / 255.0,
                action,
                reward[0],
                next_obs_np / 255.0,
                done[0]
            )
            
            # Training step
            if len(replay_buffer) > batch_size and step_count % 4 == 0:
                # Sample batch
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.BoolTensor(dones)
                
                # Current Q values
                current_q_values = model(states).gather(1, actions.unsqueeze(1))
                
                # Next Q values from target network
                with torch.no_grad():
                    next_q_values = target_model(next_states).max(1)[0]
                    target_q_values = rewards + (0.99 * next_q_values * ~dones)
                
                # Compute loss and update
                loss = criterion(current_q_values.squeeze(), target_q_values)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Update target network
            if step_count % target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())
            
            obs = next_obs
            episode_reward += reward[0]
            episode_steps += 1
            step_count += 1
        
        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.1f}, Epsilon = {epsilon:.3f}, Steps = {episode_steps}")
        
        # Test every 10 episodes
        if (episode + 1) % 10 == 0:
            print(f"\nTesting after episode {episode + 1}...")
            test_rewards = []
            
            for test_ep in range(3):
                test_env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=test_ep + 300)
                test_env = VecFrameStack(test_env, n_stack=4)
                
                test_obs = test_env.reset()
                test_reward = 0
                test_done = False
                test_steps = 0
                
                model.eval()
                while not test_done and test_steps < 3000:
                    if len(test_obs.shape) == 4 and test_obs.shape[-1] == 4:
                        test_obs_tensor = torch.FloatTensor(test_obs).permute(0, 3, 1, 2) / 255.0
                    else:
                        test_obs_tensor = torch.FloatTensor(test_obs) / 255.0
                    
                    with torch.no_grad():
                        q_values = model(test_obs_tensor)
                        action = q_values.argmax().item()
                    
                    test_obs, reward, test_done, info = test_env.step([action])
                    test_reward += reward[0]
                    test_steps += 1
                
                test_env.close()
                test_rewards.append(test_reward)
                model.train()
            
            avg_test_reward = np.mean(test_rewards)
            print(f"Test episodes: {test_rewards}")
            print(f"Average test reward: {avg_test_reward:.2f}")
            
            if avg_test_reward > best_avg_reward:
                best_avg_reward = avg_test_reward
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'episode': episode + 1,
                    'best_avg_reward': best_avg_reward,
                    'test_rewards': test_rewards
                }, 'best_trained_sequential_dqn.pt')
                print(f"NEW BEST: Saved model with {avg_test_reward:.2f} average reward")
            
            # Early stopping if we achieve good performance
            if avg_test_reward >= 15:
                print(f"EXCELLENT: Achieved {avg_test_reward:.2f} average reward!")
                print("Stopping training early due to good performance")
                break
    
    env.close()
    
    # Final test
    print(f"\n=== Final Testing ===")
    print(f"Best average reward achieved: {best_avg_reward:.2f}")
    
    if best_avg_reward >= 15:
        print("SUCCESS: Model trained to good performance!")
        print("Model saved as 'best_trained_sequential_dqn.pt'")
        return model, best_avg_reward
    elif best_avg_reward >= 0:
        print("PARTIAL SUCCESS: Model shows positive performance")
        return model, best_avg_reward
    else:
        print("LIMITED SUCCESS: Model trained but needs more work")
        return model, best_avg_reward

if __name__ == "__main__":
    model, best_reward = quick_train_sequential()
    
    print(f"\nCONCLUSION: Quick training achieved {best_reward:.2f} best average reward")
    
    if best_reward >= 15:
        print("SUCCESS: Sequential DQN is working well!")
        print("Ready for further use or SNN conversion!")
    elif best_reward >= 0:
        print("PROGRESS: Sequential DQN shows promise with more training")
    else:
        print("CHALLENGE: Sequential DQN needs more training time")