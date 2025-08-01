#!/usr/bin/env python3
"""Continue training Sequential DQN from -16.80 to +21 performance"""

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

class AdvancedDQNTrainer:
    """Advanced DQN trainer with improvements for breaking through -16 barrier"""
    
    def __init__(self, model, target_model, lr=0.0001):
        self.model = model
        self.target_model = target_model
        self.optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-4)
        self.criterion = nn.SmoothL1Loss()
        self.replay_buffer = deque(maxlen=100000)  # Larger buffer
        
        # Adjusted parameters for continued learning
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 0.05  # Lower epsilon for more exploitation
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995  # Slower decay
        self.target_update_freq = 2000  # Less frequent updates for stability
        self.steps = 0
        
        # Learning rate scheduling
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=20
        )
        
    def preprocess_obs(self, obs):
        """Preprocess observation"""
        if len(obs.shape) == 4 and obs.shape[-1] == 4:
            obs_tensor = torch.FloatTensor(obs).permute(0, 3, 1, 2)
        else:
            obs_tensor = torch.FloatTensor(obs)
        return obs_tensor / 255.0
    
    def select_action(self, obs, training=True):
        """Improved epsilon-greedy with exploration bonus"""
        if training and random.random() < self.epsilon:
            # Weighted random selection - avoid always choosing same random action
            weights = np.array([0.1, 0.2, 0.25, 0.25, 0.1, 0.1])  # Prefer movement actions
            return np.random.choice(6, p=weights)
        
        obs_tensor = self.preprocess_obs(obs)
        with torch.no_grad():
            q_values = self.model(obs_tensor)
            
            # Add small noise for exploration during exploitation
            if training:
                noise = torch.randn_like(q_values) * 0.01
                q_values += noise
                
            return q_values.argmax().item()
    
    def store_transition(self, obs, action, reward, next_obs, done):
        """Store transition with reward shaping for Pong"""
        obs_processed = self.preprocess_obs(obs)[0].numpy()
        next_obs_processed = self.preprocess_obs(next_obs)[0].numpy() if not done else np.zeros_like(obs_processed)
        
        # Reward shaping for Pong - encourage longer rallies
        shaped_reward = reward
        if reward == 0:  # Small positive reward for keeping ball in play
            shaped_reward = 0.01
        elif reward == 1:  # Bonus for scoring
            shaped_reward = 1.0
        elif reward == -1:  # Penalty for being scored on
            shaped_reward = -1.0
            
        self.replay_buffer.append((obs_processed, action, shaped_reward, next_obs_processed, done))
    
    def train_step(self):
        """Enhanced training step with Double DQN"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch with prioritization (simple version)
        if len(self.replay_buffer) > 10000:
            # Sample more recent experiences
            recent_indices = list(range(len(self.replay_buffer) - 5000, len(self.replay_buffer)))
            old_indices = list(range(len(self.replay_buffer) - 5000))
            
            recent_sample = random.sample(recent_indices, self.batch_size // 2)
            old_sample = random.sample(old_indices, self.batch_size // 2)
            
            batch_indices = recent_sample + old_sample
            batch = [self.replay_buffer[i] for i in batch_indices]
        else:
            batch = random.sample(self.replay_buffer, self.batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)
        
        # Current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use main network to select actions, target network to evaluate
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1)
            next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # Optimize with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            print(f"  Target network updated at step {self.steps}")
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()

def continue_training_to_21():
    """Continue training from -16.80 to +21"""
    print("=== Continuing Training Sequential DQN: -16.80 → +21 ===")
    
    # Register ALE
    try:
        ale_py.register_all()
    except AttributeError:
        pass
    
    # Create environment
    env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)
    print("SUCCESS: Training environment created")
    
    # Load the previous best model
    print("\n1. Loading previous best model...")
    try:
        checkpoint = torch.load('finetuned_sequential_pong_dqn.pt', map_location='cpu', weights_only=False)
        model = SequentialDQNNetwork()
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"SUCCESS: Loaded model from episode {checkpoint.get('episode', 'unknown')}")
        print(f"Previous best reward: {checkpoint.get('best_avg_reward', 'unknown')}")
        starting_episode = checkpoint.get('episode', 0)
    except Exception as e:
        print(f"Could not load previous model: {e}")
        print("Creating fresh model...")
        model = SequentialDQNNetwork()
        starting_episode = 0
    
    # Create target model
    target_model = SequentialDQNNetwork()
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    
    # Create advanced trainer with lower learning rate
    trainer = AdvancedDQNTrainer(model, target_model, lr=0.00005)  # Lower LR for fine-tuning
    
    print("SUCCESS: Advanced trainer initialized")
    print(f"Starting from episode {starting_episode}")
    
    # Training loop
    print("\n2. Continuing training with advanced techniques...")
    episode = 0
    best_avg_reward = -16.80  # Start from known best
    recent_rewards = deque(maxlen=20)  # Longer moving average
    plateau_count = 0
    
    while episode < 300:  # More episodes for breakthrough
        obs = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        episode_loss = 0
        training_steps = 0
        
        while not done and step_count < 15000:  # Longer episodes
            # Select action
            action = trainer.select_action(obs, training=True)
            
            # Take step
            next_obs, reward, done, info = env.step([action])
            
            # Store transition
            trainer.store_transition(obs, action, reward[0], next_obs, done[0])
            
            # Train more frequently as we get more data
            if step_count % 2 == 0 and len(trainer.replay_buffer) > 1000:
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
        print(f"Episode {episode + 1 + starting_episode:3d}: "
              f"Reward = {episode_reward:6.1f}, Avg20 = {avg_recent_reward:6.2f}, "
              f"Loss = {avg_loss:.4f}, Eps = {trainer.epsilon:.4f}, "
              f"Buffer = {len(trainer.replay_buffer)}")
        
        # Test every 15 episodes
        if (episode + 1) % 15 == 0:
            print(f"\n--- Testing after episode {episode + 1 + starting_episode} ---")
            test_rewards = []
            
            model.eval()
            for test_ep in range(7):  # More test episodes
                test_env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=test_ep + 2000 + episode)
                test_env = VecFrameStack(test_env, n_stack=4)
                
                test_obs = test_env.reset()
                test_reward = 0
                test_done = False
                test_steps = 0
                
                while not test_done and test_steps < 15000:
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
            
            # Update learning rate based on progress
            trainer.scheduler.step(avg_test_reward)
            
            # Save if best
            if avg_test_reward > best_avg_reward:
                improvement = avg_test_reward - best_avg_reward
                best_avg_reward = avg_test_reward
                plateau_count = 0
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'episode': episode + 1 + starting_episode,
                    'best_avg_reward': best_avg_reward,
                    'test_rewards': test_rewards,
                    'training_steps': trainer.steps,
                    'improvement': improvement
                }, 'continued_sequential_pong_dqn.pt')
                
                print(f"*** NEW BEST: {avg_test_reward:.2f} (+{improvement:.2f}) - Model saved! ***")
            else:
                plateau_count += 1
                
            # Success check
            if avg_test_reward >= 21:
                print(f"\n🎉 SUCCESS: Achieved +21 average reward ({avg_test_reward:.2f})!")
                print("Training complete!")
                break
            elif avg_test_reward >= 10:
                print(f"🚀 BREAKTHROUGH: Positive territory reached ({avg_test_reward:.2f})")
            elif avg_test_reward >= 0:
                print(f"🎯 MAJOR PROGRESS: Near positive ({avg_test_reward:.2f})")
            elif avg_test_reward > -10:
                print(f"📈 GOOD PROGRESS: Breaking through ({avg_test_reward:.2f})")
            
            # Adaptive training adjustments
            if plateau_count >= 3:
                print(f"  Plateau detected ({plateau_count} tests), adjusting strategy...")
                trainer.epsilon = min(0.1, trainer.epsilon * 1.5)  # Increase exploration
                plateau_count = 0
            
            print()
        
        episode += 1
    
    env.close()
    
    # Final results
    print(f"\n=== CONTINUED TRAINING COMPLETE ===")
    print(f"Best average reward achieved: {best_avg_reward:.2f}")
    print(f"Improvement from -16.80: +{best_avg_reward + 16.80:.2f}")
    print(f"Total training episodes: {episode + starting_episode}")
    
    if best_avg_reward >= 21:
        print("🏆 MISSION ACCOMPLISHED: Sequential DQN achieved +21!")
        print("Model saved as 'continued_sequential_pong_dqn.pt'")
        print("Ready for SNN conversion!")
    elif best_avg_reward >= 10:
        print("🚀 BREAKTHROUGH: Reached positive territory!")
        print("Continue training to reach +21 target")
    elif best_avg_reward >= 0:
        print("🎯 MAJOR BREAKTHROUGH: Near positive performance!")
    elif best_avg_reward > -10:
        print("📈 SIGNIFICANT PROGRESS: Substantial improvement!")
    else:
        print("💪 PROGRESS MADE: Continue with longer training")
    
    return model, best_avg_reward

if __name__ == "__main__":
    print("Continuing Sequential DQN training from -16.80...")
    print("Target: +21 average reward")
    print("Using advanced techniques: Double DQN, reward shaping, adaptive exploration\n")
    
    model, best_reward = continue_training_to_21()
    
    improvement = best_reward + 16.80
    print(f"\nFINAL RESULT: Improved from -16.80 to {best_reward:.2f} (+{improvement:.2f})")
    
    if best_reward >= 21:
        print("🏆 SUCCESS: Ready for SNN conversion!")
    elif improvement >= 10:
        print("🚀 EXCELLENT: Major breakthrough achieved!")
    else:
        print("📈 PROGRESS: Keep training for optimal results!")