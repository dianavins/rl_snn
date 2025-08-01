#!/usr/bin/env python3
"""Advanced Dueling Double DQN to break through -15 plateau"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import ale_py

class DuelingDQNNetwork(nn.Module):
    """Dueling DQN architecture with separate value and advantage streams"""
    
    def __init__(self, input_channels=4, n_actions=6):
        super().__init__()
        
        # Shared convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate conv output size
        self.conv_output_size = 3136  # 64 * 7 * 7
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
    def forward(self, x):
        # Shared convolutional features
        conv_out = self.conv_layers(x)
        
        # Value and advantage streams
        value = self.value_stream(conv_out)
        advantage = self.advantage_stream(conv_out)
        
        # Combine using dueling architecture
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values

class AdvancedDQNTrainer:
    """Enhanced trainer with Double DQN, Dueling DQN, and advanced training techniques"""
    
    def __init__(self, model, target_model):
        self.model = model
        self.target_model = target_model
        
        # Advanced optimizers with learning rate scheduling
        self.optimizer = optim.Adam(model.parameters(), lr=0.00025, eps=1e-4, weight_decay=1e-6)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50000, gamma=0.9)
        self.criterion = nn.SmoothL1Loss()
        
        # Larger experience replay buffer for better stability
        self.replay_buffer = deque(maxlen=200000)
        self.priorities = deque(maxlen=200000)
        
        # Enhanced training parameters
        self.batch_size = 64  # Larger batch for more stable learning
        self.gamma = 0.995    # Higher discount for longer-term thinking
        self.epsilon = 0.9    # Start with higher exploration
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.9995  # Slower decay for better exploration
        self.target_update_freq = 8000  # Less frequent updates for stability
        self.steps = 0
        
        # Performance tracking and adaptive parameters
        self.recent_rewards = deque(maxlen=100)
        self.best_reward = -21
        self.performance_history = deque(maxlen=50)
        self.stuck_counter = 0
        self.last_avg_reward = -21
        
        # Enhanced reward shaping parameters
        self.ball_y_history = deque(maxlen=5)
        self.paddle_y_history = deque(maxlen=5)
        self.consecutive_hits = 0
        
    def preprocess_obs(self, obs):
        """Preprocess observation"""
        if len(obs.shape) == 4 and obs.shape[-1] == 4:
            obs_tensor = torch.FloatTensor(obs).permute(0, 3, 1, 2)
        else:
            obs_tensor = torch.FloatTensor(obs)
        return obs_tensor / 255.0
    
    def select_action(self, obs, training=True):
        """Advanced action selection with adaptive exploration"""
        obs_tensor = self.preprocess_obs(obs)
        
        if training:
            # Adaptive epsilon based on recent performance
            current_epsilon = self.epsilon
            if len(self.performance_history) > 10:
                recent_improvement = np.mean(list(self.performance_history)[-5:]) - np.mean(list(self.performance_history)[-10:-5])
                if recent_improvement < 0.1:  # Stuck, increase exploration
                    current_epsilon = min(0.5, self.epsilon * 1.5)
            
            if random.random() < current_epsilon:
                # Intelligent exploration based on game state
                with torch.no_grad():
                    q_values = self.model(obs_tensor)
                    q_probs = F.softmax(q_values / 0.1, dim=1).squeeze().numpy()
                
                # Bias towards actions with higher Q-values but still explore
                action_weights = q_probs * 0.7 + 0.3 / 6  # Mix Q-values with uniform
                # Reduce NOOP probability
                action_weights[0] *= 0.3
                action_weights = action_weights / action_weights.sum()
                
                return np.random.choice(6, p=action_weights)
        
        with torch.no_grad():
            q_values = self.model(obs_tensor)
            return q_values.argmax().item()
    
    def extract_game_features(self, obs):
        """Extract game features for enhanced reward shaping"""
        try:
            # Convert to numpy if tensor
            if torch.is_tensor(obs):
                frame = obs[0, -1].numpy() * 255  # Last frame, denormalized
            else:
                frame = obs[0, :, :, -1] * 255  # Last frame
            
            frame = frame.astype(np.uint8)
            
            # Find ball and paddle positions (rough approximation)
            # Ball is typically small and moves, paddle is larger and at bottom
            ball_candidates = np.where(frame > 200)  # Bright pixels
            
            ball_y = np.mean(ball_candidates[0]) if len(ball_candidates[0]) > 0 else 105
            paddle_region = frame[170:190, :]  # Bottom region where paddle is
            paddle_x = np.mean(np.where(paddle_region > 100)[1]) if np.any(paddle_region > 100) else 80
            
            return ball_y, paddle_x
        except:
            return 105, 80  # Default values
    
    def store_transition(self, obs, action, reward, next_obs, done):
        """Store transition with enhanced reward shaping"""
        obs_processed = self.preprocess_obs(obs)[0].numpy()
        next_obs_processed = self.preprocess_obs(next_obs)[0].numpy() if not done else np.zeros_like(obs_processed)
        
        # Enhanced reward shaping for Pong
        shaped_reward = reward
        
        if reward != 0:
            # Score events get full reward
            shaped_reward = reward
            if reward > 0:
                self.consecutive_hits += 1
            else:
                self.consecutive_hits = 0
        else:
            # Extract game state for intermediate rewards
            ball_y, paddle_x = self.extract_game_features(obs)
            self.ball_y_history.append(ball_y)
            self.paddle_y_history.append(paddle_x)
            
            # Base survival reward
            shaped_reward = 0.01
            
            # Ball tracking bonus - reward for keeping ball in play
            if len(self.ball_y_history) >= 2:
                ball_movement = abs(self.ball_y_history[-1] - self.ball_y_history[-2])
                if ball_movement > 2:  # Ball is moving
                    shaped_reward += 0.02
            
            # Paddle positioning bonus - reward for good paddle position
            if ball_y > 140:  # Ball in lower half
                distance_to_ball = abs(paddle_x - 80)  # Distance from center
                if distance_to_ball < 20:  # Good positioning
                    shaped_reward += 0.03
                elif distance_to_ball < 40:
                    shaped_reward += 0.01
            
            # Action-based shaping
            if action in [2, 3]:  # UP or DOWN movement
                shaped_reward += 0.005  # Slight bonus for movement
            elif action == 0:  # NOOP penalty to encourage action
                shaped_reward -= 0.01
        
        # Store transition
        transition = (obs_processed, action, shaped_reward, next_obs_processed, done)
        self.replay_buffer.append(transition)
        
        # Enhanced priority based on reward and TD error potential
        if abs(reward) > 0:
            priority = 2.0  # High priority for score events
        elif shaped_reward > 0.02:
            priority = 1.0  # Medium priority for good play
        else:
            priority = 0.3  # Lower priority for routine play
        
        self.priorities.append(priority)
    
    def sample_batch(self):
        """Sample batch with prioritized experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Convert priorities to probabilities
        priorities_array = np.array(self.priorities)
        probabilities = priorities_array / priorities_array.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, p=probabilities)
        
        # Get batch
        batch = [self.replay_buffer[i] for i in indices]
        
        return batch, indices
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for i, td_error in zip(indices, td_errors):
            priority = abs(td_error) + 1e-6
            if i < len(self.priorities):
                self.priorities[i] = priority
    
    def train_step(self):
        """Enhanced Double DQN training step with adaptive learning"""
        batch_data = self.sample_batch()
        if batch_data is None:
            return 0.0
            
        batch, indices = batch_data
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
            # Main network selects best actions
            next_actions = self.model(next_states).argmax(1)
            # Target network evaluates those actions
            next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute TD errors for priority updates
        td_errors = (current_q_values.squeeze() - target_q_values).detach().numpy()
        self.update_priorities(indices, td_errors)
        
        # Enhanced loss with importance sampling weights
        td_loss = current_q_values.squeeze() - target_q_values
        
        # Weight losses by priority (importance sampling)
        priorities_tensor = torch.FloatTensor([self.priorities[i] for i in indices])
        weights = (1.0 / (priorities_tensor + 1e-8)).pow(0.4)  # β=0.4 for stability
        weights = weights / weights.max()  # Normalize
        
        weighted_loss = (weights * td_loss.pow(2)).mean()
        
        # Adaptive gradient clipping based on TD error magnitude
        grad_clip_value = min(10.0, max(1.0, np.mean(np.abs(td_errors)) * 5))
        
        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_value)
        self.optimizer.step()
        self.scheduler.step()
        
        # Soft target network updates for stability
        self.steps += 1
        if self.steps % 1000 == 0:  # More frequent soft updates
            tau = 0.005  # Soft update parameter
            for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        
        # Hard update less frequently
        if self.steps % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        # Adaptive epsilon decay based on performance
        if len(self.performance_history) > 10:
            recent_improvement = np.mean(list(self.performance_history)[-5:]) - np.mean(list(self.performance_history)[-10:-5])
            if recent_improvement > 0.5:  # Good progress, reduce exploration
                self.epsilon_decay = 0.999
            elif recent_improvement < -0.2:  # Getting worse, increase exploration
                self.epsilon_decay = 0.9998
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return weighted_loss.item()

def advanced_dqn_training():
    """Main training function with Dueling Double DQN"""
    print("=== ADVANCED DUELING DOUBLE DQN TRAINING ===")
    print("Breaking through -15 plateau with state-of-the-art techniques")
    
    # Register ALE
    try:
        ale_py.register_all()
    except:
        pass
    
    # Create environment
    env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)
    print("SUCCESS: Environment created")
    
    # Load previous model and convert to Dueling architecture
    print("\n1. Creating Dueling DQN model...")
    dueling_model = DuelingDQNNetwork()
    target_model = DuelingDQNNetwork()
    
    # Try to load previous weights if available
    try:
        from conversion.sb3_to_sequential.sb3_to_sequential_converter import SequentialDQNNetwork
        checkpoint = torch.load('continued_sequential_pong_dqn.pt', map_location='cpu', weights_only=False)
        old_model = SequentialDQNNetwork()
        old_model.load_state_dict(checkpoint['model_state_dict'])
        
        print("SUCCESS: Found previous model, transferring compatible weights...")
        
        # Transfer convolutional weights (these should match)
        dueling_model.conv_layers[0].weight.data.copy_(old_model.network.conv1.weight.data)
        dueling_model.conv_layers[0].bias.data.copy_(old_model.network.conv1.bias.data)
        dueling_model.conv_layers[2].weight.data.copy_(old_model.network.conv2.weight.data)
        dueling_model.conv_layers[2].bias.data.copy_(old_model.network.conv2.bias.data)
        dueling_model.conv_layers[4].weight.data.copy_(old_model.network.conv3.weight.data)
        dueling_model.conv_layers[4].bias.data.copy_(old_model.network.conv3.bias.data)
        
        # For linear layers, we need to be more careful with dimensions
        # Old model: fc1 [512, 3136] -> fc2 [6, 512]
        # New model: value_stream [512, 3136] -> [1, 512], advantage_stream [512, 3136] -> [6, 512]
        
        old_fc1_weight = old_model.network.fc1.weight.data  # [512, 3136]
        old_fc1_bias = old_model.network.fc1.bias.data      # [512]
        old_fc2_weight = old_model.network.fc2.weight.data  # [6, 512]
        old_fc2_bias = old_model.network.fc2.bias.data      # [6]
        
        # Split the hidden layer weights between value and advantage streams
        # Value stream gets full fc1 weights but outputs 1 value
        dueling_model.value_stream[0].weight.data.copy_(old_fc1_weight)
        dueling_model.value_stream[0].bias.data.copy_(old_fc1_bias)
        # Value stream final layer: initialize to output mean of old fc2 bias
        dueling_model.value_stream[2].weight.data.copy_(old_fc2_weight.mean(dim=0, keepdim=True))
        dueling_model.value_stream[2].bias.data.copy_(old_fc2_bias.mean().unsqueeze(0))
        
        # Advantage stream gets full fc1 weights and original fc2 weights
        dueling_model.advantage_stream[0].weight.data.copy_(old_fc1_weight)
        dueling_model.advantage_stream[0].bias.data.copy_(old_fc1_bias)
        dueling_model.advantage_stream[2].weight.data.copy_(old_fc2_weight)
        dueling_model.advantage_stream[2].bias.data.copy_(old_fc2_bias)
        
        print("SUCCESS: Weight transfer completed")
        
    except Exception as e:
        print(f"No previous model found ({e}), using fresh initialization")
        
        # Better initialization for Dueling DQN
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        
        dueling_model.apply(init_weights)
    
    # Initialize target network
    target_model.load_state_dict(dueling_model.state_dict())
    target_model.eval()
    
    # Create trainer
    trainer = AdvancedDQNTrainer(dueling_model, target_model)
    print("SUCCESS: Advanced Dueling Double DQN trainer created")
    
    # Training loop with enhanced plateau detection
    print("\n2. Starting enhanced training...")
    episode = 0
    best_avg_reward = -21
    plateau_episodes = 0
    plateau_threshold = 0.2  # Improvement threshold to avoid plateau
    warmup_episodes = 50  # Episodes before starting plateau detection
    
    while episode < 300:  # Extended training
        obs = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        episode_loss = 0
        training_steps = 0
        
        while not done and step_count < 10000:
            action = trainer.select_action(obs, training=True)
            next_obs, reward, done, info = env.step([action])
            
            trainer.store_transition(obs, action, reward[0], next_obs, done[0])
            
            # Train every 4 steps
            if step_count % 4 == 0 and len(trainer.replay_buffer) > 1000:
                loss = trainer.train_step()
                if loss > 0:
                    episode_loss += loss
                    training_steps += 1
            
            obs = next_obs
            episode_reward += reward[0]
            step_count += 1
        
        trainer.recent_rewards.append(episode_reward)
        trainer.performance_history.append(episode_reward)
        avg_recent = np.mean(list(trainer.recent_rewards)[-20:]) if len(trainer.recent_rewards) >= 20 else np.mean(trainer.recent_rewards)
        
        # Plateau detection and counter-measures
        if episode > warmup_episodes and episode % 20 == 0:
            improvement = avg_recent - trainer.last_avg_reward
            if improvement < plateau_threshold:
                plateau_episodes += 1
                print(f"⚠️  Plateau detected! Episodes stuck: {plateau_episodes * 20}")
                
                # Anti-plateau measures
                if plateau_episodes >= 2:  # Stuck for 40+ episodes
                    print("🔧 Applying anti-plateau measures...")
                    # Increase exploration
                    trainer.epsilon = min(0.4, trainer.epsilon * 2)
                    # Reset learning rate
                    for param_group in trainer.optimizer.param_groups:
                        param_group['lr'] = min(0.001, param_group['lr'] * 1.5)
                    # Clear some old experience to encourage fresh learning
                    buffer_size = len(trainer.replay_buffer)
                    keep_recent = int(buffer_size * 0.7)
                    trainer.replay_buffer = deque(list(trainer.replay_buffer)[-keep_recent:], maxlen=200000)
                    trainer.priorities = deque(list(trainer.priorities)[-keep_recent:], maxlen=200000)
                    plateau_episodes = 0
                    print(f"   ✓ Exploration ε: {trainer.epsilon:.3f}")
                    print(f"   ✓ Learning rate: {trainer.optimizer.param_groups[0]['lr']:.6f}")
                    print(f"   ✓ Buffer reset: {len(trainer.replay_buffer)} experiences kept")
            else:
                plateau_episodes = 0  # Reset counter on improvement
            
            trainer.last_avg_reward = avg_recent
        
        # Progress report
        avg_loss = episode_loss / max(training_steps, 1)
        lr = trainer.optimizer.param_groups[0]['lr']
        print(f"Episode {episode + 1:3d}: Reward = {episode_reward:6.1f}, "
              f"Avg20 = {avg_recent:6.2f}, Loss = {avg_loss:.4f}, "
              f"Eps = {trainer.epsilon:.3f}, LR = {lr:.6f}, Buffer = {len(trainer.replay_buffer)}")
        
        # Test every 15 episodes for more frequent feedback
        if (episode + 1) % 15 == 0:
            print(f"\n--- Testing after episode {episode + 1} ---")
            test_rewards = []
            
            dueling_model.eval()
            for test_ep in range(5):
                test_env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=test_ep + 4000)
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
            
            dueling_model.train()
            
            avg_test = np.mean(test_rewards)
            print(f"Test rewards: {[int(r) for r in test_rewards]}")
            print(f"Average test: {avg_test:.2f}")
            
            # Save if improvement
            if avg_test > best_avg_reward:
                improvement = avg_test - best_avg_reward
                best_avg_reward = avg_test
                
                torch.save({
                    'model_state_dict': dueling_model.state_dict(),
                    'model_type': 'DuelingDQN',
                    'episode': episode + 1,
                    'best_avg_reward': best_avg_reward,
                    'test_rewards': test_rewards
                }, 'dueling_dqn_pong.pt')
                
                print(f"*** NEW BEST: {avg_test:.2f} (+{improvement:.2f}) - Dueling DQN saved! ***")
                
                if avg_test >= 21:
                    print("🎉 SUCCESS: Achieved +21 target!")
                    break
                elif avg_test >= 10:
                    print("🚀 BREAKTHROUGH: Positive territory!")
                elif avg_test >= 0:
                    print("🎯 MAJOR PROGRESS: Near breakthrough!")
            
            print()
        
        episode += 1
    
    env.close()
    
    print(f"\n=== ADVANCED TRAINING COMPLETE ===")
    print(f"Best average reward: {best_avg_reward:.2f}")
    print(f"Improvement from -15: +{best_avg_reward + 15:.2f}")
    
    if best_avg_reward >= 21:
        print("🏆 MISSION ACCOMPLISHED: Ready for SNN conversion!")
    elif best_avg_reward >= 0:
        print("🚀 BREAKTHROUGH ACHIEVED: Reached positive territory!")
    elif best_avg_reward > -10:
        print("📈 MAJOR PROGRESS: Significant improvement!")
    else:
        print("💪 PROGRESS MADE: Continue with advanced techniques")
    
    return dueling_model, best_avg_reward

if __name__ == "__main__":
    print("ADVANCED DUELING DOUBLE DQN")
    print("State-of-the-art techniques:")
    print("- Dueling architecture (separate value/advantage streams)")
    print("- Double DQN (reduces overestimation bias)")
    print("- Prioritized experience replay")
    print("- Improved exploration and reward shaping\n")
    
    model, best_reward = advanced_dqn_training()
    
    print(f"\nFINAL RESULT: Advanced DQN achieved {best_reward:.2f} best reward")
    
    if best_reward >= 21:
        print("🏆 SUCCESS: Ready for SNN conversion with dueling_dqn_pong.pt!")
    elif best_reward >= 0:
        print("🎯 EXCELLENT: Major breakthrough achieved!")
    else:
        print("📈 PROGRESS: Advanced architecture showing improvement!")