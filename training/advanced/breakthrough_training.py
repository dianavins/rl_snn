#!/usr/bin/env python3
"""Breakthrough training script for Sequential DQN stuck at -15"""

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

class BreakthroughTrainer:
    """Advanced trainer designed to break through -15 plateau"""
    
    def __init__(self, model, target_model):
        self.model = model
        self.target_model = target_model
        
        # Multiple optimizers for different learning phases
        self.main_optimizer = optim.Adam(model.parameters(), lr=0.0003, eps=1e-4)  # Higher LR
        self.fine_optimizer = optim.RMSprop(model.parameters(), lr=0.0001)  # Alternative optimizer
        self.current_optimizer = self.main_optimizer
        
        self.criterion = nn.SmoothL1Loss()
        self.replay_buffer = deque(maxlen=200000)  # Much larger buffer
        
        # Aggressive exploration settings
        self.batch_size = 64
        self.gamma = 0.995  # Slightly higher gamma for longer-term rewards
        self.epsilon = 0.3   # High exploration to break plateau
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.target_update_freq = 1000
        self.steps = 0
        
        # Curriculum learning parameters
        self.difficulty_level = 0
        self.success_threshold = -10
        
        # Performance tracking
        self.recent_rewards = deque(maxlen=50)
        self.plateau_counter = 0
        self.best_reward = -21
        
    def adaptive_epsilon(self, avg_reward):
        """Adaptive exploration based on performance"""
        if avg_reward < -18:
            return 0.4  # Very high exploration if terrible
        elif avg_reward < -15:
            return 0.2  # High exploration if stuck
        elif avg_reward < -10:
            return 0.1  # Medium exploration if improving
        else:
            return 0.05  # Low exploration if good
    
    def preprocess_obs(self, obs):
        """Enhanced preprocessing with normalization"""
        if len(obs.shape) == 4 and obs.shape[-1] == 4:
            obs_tensor = torch.FloatTensor(obs).permute(0, 3, 1, 2)
        else:
            obs_tensor = torch.FloatTensor(obs)
        
        # Improved normalization
        obs_tensor = obs_tensor / 255.0
        # Add slight contrast enhancement
        obs_tensor = torch.clamp(obs_tensor * 1.1 - 0.05, 0, 1)
        return obs_tensor
    
    def select_action(self, obs, training=True):
        """Improved action selection with noise injection"""
        if training:
            # Dynamic epsilon based on recent performance
            current_epsilon = self.adaptive_epsilon(np.mean(self.recent_rewards) if self.recent_rewards else -21)
            
            if random.random() < current_epsilon:
                # Smarter exploration - avoid NOOP too much
                action_weights = [0.05, 0.25, 0.25, 0.25, 0.1, 0.1]  # Reduce NOOP, favor movement
                return np.random.choice(6, p=action_weights)
        
        obs_tensor = self.preprocess_obs(obs)
        with torch.no_grad():
            q_values = self.model(obs_tensor)
            
            # Add noise during training for exploration
            if training:
                noise = torch.randn_like(q_values) * 0.02
                q_values += noise
            
            return q_values.argmax().item()
    
    def store_transition(self, obs, action, reward, next_obs, done):
        """Enhanced transition storage with reward engineering"""
        obs_processed = self.preprocess_obs(obs)[0].numpy()
        next_obs_processed = self.preprocess_obs(next_obs)[0].numpy() if not done else np.zeros_like(obs_processed)
        
        # Reward engineering for Pong
        engineered_reward = reward
        
        # Survival bonus - reward staying alive
        if reward == 0:
            engineered_reward = 0.01
        
        # Big bonus for scoring
        if reward == 1:
            engineered_reward = 2.0
        
        # Big penalty for being scored on, but not too harsh
        if reward == -1:
            engineered_reward = -1.5
        
        self.replay_buffer.append((obs_processed, action, engineered_reward, next_obs_processed, done))
    
    def prioritized_sample(self, batch_size):
        """Prioritized experience replay"""
        if len(self.replay_buffer) < batch_size:
            return None
            
        # Simple prioritization: favor recent experiences and non-zero rewards
        buffer_list = list(self.replay_buffer)
        priorities = []
        
        for i, (_, _, reward, _, done) in enumerate(buffer_list):
            base_priority = 1.0
            
            # Higher priority for non-zero rewards
            if abs(reward) > 0.1:
                base_priority *= 3.0
            
            # Higher priority for recent experiences
            recency_factor = (i / len(buffer_list)) ** 0.5
            base_priority *= (0.5 + recency_factor)
            
            priorities.append(base_priority)
        
        # Convert to probabilities
        priorities = np.array(priorities)
        probabilities = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(len(buffer_list), batch_size, p=probabilities)
        batch = [buffer_list[i] for i in indices]
        
        return batch
    
    def train_step(self):
        """Enhanced training step with techniques to break plateau"""
        batch = self.prioritized_sample(self.batch_size)
        if batch is None:
            return 0.0
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)
        
        # Current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN with target network
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1)
            next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss with gradient clipping
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # Optimize with current optimizer
        self.current_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)  # Tighter clipping
        self.current_optimizer.step()
        
        # Update target network more frequently if plateau
        update_freq = 500 if np.mean(self.recent_rewards[-10:] if len(self.recent_rewards) >= 10 else [-21]) > -16 else 1000
        
        self.steps += 1
        if self.steps % update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        return loss.item()
    
    def check_plateau_and_adapt(self, avg_reward):
        """Check for plateau and adapt training strategy"""
        if len(self.recent_rewards) < 20:
            return
            
        recent_improvement = avg_reward - np.mean(list(self.recent_rewards)[-20:-10])
        
        if abs(recent_improvement) < 0.5:  # Less than 0.5 improvement
            self.plateau_counter += 1
        else:
            self.plateau_counter = 0
        
        # Adaptive strategies
        if self.plateau_counter >= 5:
            print(f"  PLATEAU DETECTED ({self.plateau_counter}) - Adapting strategy...")
            
            # Strategy 1: Switch optimizer
            if self.current_optimizer == self.main_optimizer:
                self.current_optimizer = self.fine_optimizer
                print("    Switched to RMSprop optimizer")
            else:
                self.current_optimizer = self.main_optimizer
                print("    Switched back to Adam optimizer")
            
            # Strategy 2: Increase exploration
            self.epsilon = min(0.5, self.epsilon * 1.5)
            print(f"    Increased exploration to {self.epsilon:.3f}")
            
            # Strategy 3: Learning rate adjustment
            for param_group in self.current_optimizer.param_groups:
                param_group['lr'] *= 1.2  # Increase LR
            print(f"    Increased learning rate to {param_group['lr']:.6f}")
            
            self.plateau_counter = 0

def breakthrough_training():
    """Main breakthrough training function"""
    print("=== BREAKTHROUGH TRAINING: Breaking Through -15 Plateau ===")
    
    # Register ALE
    try:
        ale_py.register_all()
    except:
        pass
    
    # Create environment
    env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)
    print("SUCCESS: Environment created")
    
    # Load stuck model
    print("\n1. Loading model stuck at -15...")
    try:
        checkpoint = torch.load('continued_sequential_pong_dqn.pt', map_location='cpu', weights_only=False)
        model = SequentialDQNNetwork()
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"SUCCESS: Loaded model with {checkpoint.get('best_avg_reward', 'unknown')} reward")
    except:
        print("Using finetuned model as fallback...")
        checkpoint = torch.load('finetuned_sequential_pong_dqn.pt', map_location='cpu', weights_only=False)
        model = SequentialDQNNetwork()
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create target model
    target_model = SequentialDQNNetwork()
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    
    # Create breakthrough trainer
    trainer = BreakthroughTrainer(model, target_model)
    
    print("SUCCESS: Breakthrough trainer initialized")
    print("Strategy: High exploration, reward engineering, prioritized replay, adaptive optimization")
    
    # Training loop with breakthrough focus
    episode = 0
    breakthrough_achieved = False
    
    while episode < 150 and not breakthrough_achieved:
        obs = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        episode_loss = 0
        training_steps = 0
        
        while not done and step_count < 20000:  # Longer episodes
            action = trainer.select_action(obs, training=True)
            next_obs, reward, done, info = env.step([action])
            
            trainer.store_transition(obs, action, reward[0], next_obs, done[0])
            
            # Train every step if enough data
            if len(trainer.replay_buffer) > 1000:
                loss = trainer.train_step()
                if loss > 0:
                    episode_loss += loss
                    training_steps += 1
            
            obs = next_obs
            episode_reward += reward[0]
            step_count += 1
        
        trainer.recent_rewards.append(episode_reward)
        avg_recent = np.mean(trainer.recent_rewards)
        
        # Check for breakthrough
        if avg_recent > -5:
            breakthrough_achieved = True
            print(f"\n🎉 BREAKTHROUGH ACHIEVED! Average: {avg_recent:.2f}")
        
        # Adaptive training
        trainer.check_plateau_and_adapt(avg_recent)
        
        # Progress report
        avg_loss = episode_loss / max(training_steps, 1)
        current_eps = trainer.adaptive_epsilon(avg_recent)
        
        print(f"Episode {episode + 1:3d}: Reward = {episode_reward:6.1f}, "
              f"Avg{len(trainer.recent_rewards)} = {avg_recent:6.2f}, "
              f"Loss = {avg_loss:.4f}, Eps = {current_eps:.3f}, "
              f"Buffer = {len(trainer.replay_buffer)}")
        
        # Test every 10 episodes
        if (episode + 1) % 10 == 0:
            print(f"\n--- Testing after episode {episode + 1} ---")
            test_rewards = []
            
            model.eval()
            for test_ep in range(5):
                test_env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=test_ep + 3000)
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
            
            avg_test = np.mean(test_rewards)
            print(f"Test rewards: {[int(r) for r in test_rewards]}")
            print(f"Average test: {avg_test:.2f}")
            
            # Save if improvement
            if avg_test > trainer.best_reward:
                improvement = avg_test - trainer.best_reward
                trainer.best_reward = avg_test
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'episode': episode + 1,
                    'best_avg_reward': trainer.best_reward,
                    'test_rewards': test_rewards,
                    'breakthrough_achieved': breakthrough_achieved
                }, 'breakthrough_sequential_pong_dqn.pt')
                
                print(f"*** NEW BEST: {avg_test:.2f} (+{improvement:.2f}) ***")
                
                if avg_test >= 0:
                    print("🚀 POSITIVE TERRITORY REACHED!")
                    breakthrough_achieved = True
            
            print()
        
        episode += 1
    
    env.close()
    
    print(f"\n=== BREAKTHROUGH TRAINING COMPLETE ===")
    print(f"Final best reward: {trainer.best_reward:.2f}")
    print(f"Episodes completed: {episode}")
    
    if trainer.best_reward >= 0:
        print("🏆 SUCCESS: Broke through to positive territory!")
    elif trainer.best_reward > -10:
        print("🎯 MAJOR PROGRESS: Significant improvement achieved!")
    elif trainer.best_reward > -15:
        print("📈 PROGRESS: Broke through -15 plateau!")
    else:
        print("💪 FOUNDATION: Continue with longer training")
    
    return model, trainer.best_reward

if __name__ == "__main__":
    print("BREAKTHROUGH TRAINING MODE")
    print("Designed to break through -15 plateau with:")
    print("- Aggressive exploration and reward engineering")
    print("- Prioritized experience replay") 
    print("- Adaptive optimization strategies")
    print("- Plateau detection and strategy switching\n")
    
    model, best_reward = breakthrough_training()
    
    improvement = best_reward + 15
    print(f"\nFINAL RESULT: Improved from -15 to {best_reward:.2f} (+{improvement:.2f})")
    
    if best_reward >= 0:
        print("🏆 MISSION ACCOMPLISHED: Ready for SNN conversion!")
    else:
        print("🚀 PROGRESS MADE: Continue breakthrough training!")