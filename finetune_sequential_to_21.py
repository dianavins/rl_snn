#!/usr/bin/env python3
"""Effective fine-tuning script to get Sequential DQN to +21 performance"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from sb3_to_sequential_converter import SequentialDQNNetwork
import ale_py

class DQNTrainer:
    """Efficient DQN trainer focused on quick convergence to +21"""
    
    def __init__(self, model, target_model, lr=0.0001):
        self.model = model
        self.target_model = target_model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()  # Huber loss for stability
        self.replay_buffer = deque(maxlen=50000)
        
        # Training parameters optimized for quick convergence
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 0.1  # Start with low epsilon for exploitation
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update_freq = 1000
        self.steps = 0
        
    def preprocess_obs(self, obs):
        """Preprocess observation to match training format"""
        if len(obs.shape) == 4 and obs.shape[-1] == 4:
            obs_tensor = torch.FloatTensor(obs).permute(0, 3, 1, 2)
        else:
            obs_tensor = torch.FloatTensor(obs)
        return obs_tensor / 255.0
    
    def select_action(self, obs, training=True):
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randint(0, 5)
        
        obs_tensor = self.preprocess_obs(obs)
        with torch.no_grad():
            q_values = self.model(obs_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, obs, action, reward, next_obs, done):
        """Store transition in replay buffer"""
        obs_processed = self.preprocess_obs(obs)[0].numpy()
        next_obs_processed = self.preprocess_obs(next_obs)[0].numpy() if not done else np.zeros_like(obs_processed)
        
        self.replay_buffer.append((obs_processed, action, reward, next_obs_processed, done))
    
    def train_step(self):
        """Single training step"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)
        
        # Current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)  # Gradient clipping
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()

def finetune_sequential_to_21():
    """Fine-tune Sequential DQN to achieve +21 performance"""
    print("=== Fine-tuning Sequential DQN to +21 Performance ===")
    
    # Register ALE
    try:
        ale_py.register_all()
    except AttributeError:
        pass
    
    # Create environment
    env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)
    print("SUCCESS: Training environment created")
    
    # Load existing Sequential model as starting point
    print("\n1. Loading Sequential model...")
    try:
        checkpoint = torch.load('working_sequential_pong_dqn.pt', map_location='cpu', weights_only=False)
        model = SequentialDQNNetwork()
        model.load_state_dict(checkpoint['model_state_dict'])
        print("SUCCESS: Loaded existing Sequential model")
    except:
        print("Creating fresh Sequential model...")
        model = SequentialDQNNetwork()
        # Better initialization for Pong
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        model.apply(init_weights)
    
    # Create target model
    target_model = SequentialDQNNetwork()
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    
    # Create trainer
    trainer = DQNTrainer(model, target_model, lr=0.00025)  # Standard DQN learning rate
    
    print("SUCCESS: Trainer initialized")
    
    # Training loop
    print("\n2. Starting fine-tuning...")
    episode = 0
    best_avg_reward = -float('inf')
    recent_rewards = deque(maxlen=10)
    
    while episode < 200:  # Max episodes
        obs = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        episode_loss = 0
        training_steps = 0
        
        while not done and step_count < 10000:  # Max steps per episode
            # Select action
            action = trainer.select_action(obs, training=True)
            
            # Take step
            next_obs, reward, done, info = env.step([action])
            
            # Store transition
            trainer.store_transition(obs, action, reward[0], next_obs, done[0])
            
            # Train every 4 steps
            if step_count % 4 == 0:
                loss = trainer.train_step()
                if loss > 0:
                    episode_loss += loss
                    training_steps += 1
            
            obs = next_obs
            episode_reward += reward[0]
            step_count += 1
        
        recent_rewards.append(episode_reward)
        avg_recent_reward = np.mean(recent_rewards)
        
        # Print progress
        avg_loss = episode_loss / max(training_steps, 1)
        print(f"Episode {episode + 1:3d}: Reward = {episode_reward:6.1f}, "
              f"Avg10 = {avg_recent_reward:6.2f}, Loss = {avg_loss:.4f}, "
              f"Eps = {trainer.epsilon:.3f}, Steps = {step_count}")
        
        # Test every 20 episodes
        if (episode + 1) % 20 == 0:
            print(f"\n--- Testing after episode {episode + 1} ---")
            test_rewards = []
            
            model.eval()
            for test_ep in range(5):
                test_env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=test_ep + 1000)
                test_env = VecFrameStack(test_env, n_stack=4)
                
                test_obs = test_env.reset()
                test_reward = 0
                test_done = False
                test_steps = 0
                
                while not test_done and test_steps < 10000:
                    test_action = trainer.select_action(test_obs, training=False)
                    test_obs, reward, test_done, info = test_env.step([test_action])
                    test_reward += reward[0]
                    test_steps += 1
                
                test_env.close()
                test_rewards.append(test_reward)
            
            model.train()
            
            avg_test_reward = np.mean(test_rewards)
            print(f"Test rewards: {[int(r) for r in test_rewards]}")
            print(f"Average test reward: {avg_test_reward:.2f}")
            
            # Save if best
            if avg_test_reward > best_avg_reward:
                best_avg_reward = avg_test_reward
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'episode': episode + 1,
                    'best_avg_reward': best_avg_reward,
                    'test_rewards': test_rewards,
                    'training_steps': trainer.steps
                }, 'finetuned_sequential_pong_dqn.pt')
                print(f"*** NEW BEST: {avg_test_reward:.2f} - Model saved! ***")
            
            # Early stopping if we achieve +21
            if avg_test_reward >= 21:
                print(f"\n🎉 SUCCESS: Achieved +21 average reward ({avg_test_reward:.2f})!")
                print("Fine-tuning complete!")
                break
            elif avg_test_reward >= 15:
                print(f"🎯 EXCELLENT: High performance achieved ({avg_test_reward:.2f})")
            elif avg_test_reward >= 5:
                print(f"✅ GOOD: Positive performance ({avg_test_reward:.2f})")
            elif avg_test_reward > best_avg_reward - 5:
                print(f"📈 PROGRESS: Improving ({avg_test_reward:.2f})")
            
            print()
        
        episode += 1
    
    env.close()
    
    # Final results
    print(f"\n=== FINE-TUNING COMPLETE ===")
    print(f"Best average reward achieved: {best_avg_reward:.2f}")
    print(f"Total training episodes: {episode}")
    print(f"Total training steps: {trainer.steps}")
    
    if best_avg_reward >= 21:
        print("🎉 SUCCESS: Sequential DQN fine-tuned to +21 performance!")
        print("Model saved as 'finetuned_sequential_pong_dqn.pt'")
        print("Ready for SNN conversion!")
    elif best_avg_reward >= 15:
        print("🎯 VERY GOOD: High performance achieved!")
        print("Close to +21 target")
    elif best_avg_reward >= 0:
        print("✅ PROGRESS: Positive performance achieved")
        print("Continue training for better results")
    else:
        print("⚠️ CHALLENGE: More training needed")
        print("Try adjusting hyperparameters or training longer")
    
    return model, best_avg_reward

if __name__ == "__main__":
    print("Starting Sequential DQN fine-tuning...")
    print("Target: +21 average reward on Pong")
    print("This will take some time but should converge faster than training from scratch.\n")
    
    model, best_reward = finetune_sequential_to_21()
    
    print(f"\nFINAL RESULT: Sequential DQN achieved {best_reward:.2f} best average reward")
    
    if best_reward >= 21:
        print("🏆 MISSION ACCOMPLISHED: Sequential DQN ready for SNN conversion!")
    elif best_reward >= 15:
        print("🌟 EXCELLENT PROGRESS: Very close to target!")
    else:
        print("💪 GOOD FOUNDATION: Continue training for optimal results")