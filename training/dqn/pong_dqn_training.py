import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque
import random
import cv2
from typing import Tuple, List
import matplotlib.pyplot as plt

class PongQNetwork(nn.Module):
    """Simple ANN for Pong with Conv2d and Linear layers"""
    
    def __init__(self, input_channels: int = 4, n_actions: int = 6):
        super(PongQNetwork, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate size after convolutions for linear layer
        # Input: (4, 84, 84) -> Conv1: (32, 20, 20) -> Conv2: (64, 9, 9) -> Conv3: (64, 7, 7)
        conv_output_size = 64 * 7 * 7  # 3136
        
        # Linear layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, n_actions)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PongPreprocessor:
    """Preprocess Pong frames to black and white"""
    
    def __init__(self, frame_size: Tuple[int, int] = (84, 84)):
        self.frame_size = frame_size
        # Pong color values (approximate)
        self.background_color = [144, 72, 17]  # Brown background
        self.ball_paddle_colors = [
            [236, 236, 236],  # White/light colors
            [213, 130, 74],   # Orange paddle
            [92, 186, 92],    # Green paddle
        ]
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to black/white and resize"""
        # Convert to grayscale first for easier processing
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Create binary mask: white for ball/paddles, black for background
        # Background pixels are dark, ball/paddles are bright
        binary = np.where(gray > 87, 255, 0).astype(np.uint8)  # Threshold around middle
        
        # Resize to target size
        resized = cv2.resize(binary, self.frame_size, interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        return resized.astype(np.float32) / 255.0

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DoubleDQNAgent:
    """Double DQN Agent for Pong"""
    
    def __init__(
        self,
        state_shape: Tuple[int, ...],
        n_actions: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: int = 1000000,
        buffer_size: int = 100000,
        batch_size: int = 32,
        target_update_freq: int = 10000,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        
        # Networks
        self.q_network = PongQNetwork(state_shape[0], n_actions).to(device)
        self.target_network = PongQNetwork(state_shape[0], n_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Tracking
        self.steps = 0
        self.losses = []
        
    def get_epsilon(self) -> float:
        """Calculate current epsilon for epsilon-greedy policy"""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-1. * self.steps / self.epsilon_decay)
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy"""
        if random.random() > self.get_epsilon():
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.n_actions)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self):
        """Update the network using Double DQN"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: Use main network to select actions, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()
        
        self.losses.append(loss.item())
        self.steps += 1
        
        # Update target network
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            print(f"Target network updated at step {self.steps}")

def train_pong_dqn(
    episodes: int = 10000,
    max_steps_per_episode: int = 10000,
    save_interval: int = 1000,
    eval_interval: int = 500
):
    """Train Double DQN on Pong"""
    
    # Environment
    env = gym.make('PongNoFrameskip-v4')
    preprocessor = PongPreprocessor()
    
    # Agent
    state_shape = (4, 84, 84)  # 4 stacked frames
    n_actions = env.action_space.n
    agent = DoubleDQNAgent(state_shape, n_actions)
    
    # Training tracking
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        state = preprocessor.preprocess_frame(state)
        
        # Stack 4 frames
        stacked_frames = deque([state] * 4, maxlen=4)
        state = np.stack(stacked_frames, axis=0)
        
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps_per_episode):
            # Select action
            action = agent.select_action(state)
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Preprocess next state
            next_state = preprocessor.preprocess_frame(next_state)
            stacked_frames.append(next_state)
            next_state = np.stack(stacked_frames, axis=0)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update
            agent.update()
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Logging
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            epsilon = agent.get_epsilon()
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                  f"Epsilon: {epsilon:.3f}, Steps: {agent.steps}")
        
        # Save model
        if episode % save_interval == 0 and episode > 0:
            torch.save({
                'episode': episode,
                'model_state_dict': agent.q_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'episode_rewards': episode_rewards,
            }, f'pong_dqn_episode_{episode}.pth')
            print(f"Model saved at episode {episode}")
        
        # Evaluation
        if episode % eval_interval == 0 and episode > 0:
            eval_reward = evaluate_agent(agent, preprocessor, episodes=5)
            print(f"Evaluation reward: {eval_reward:.2f}")
    
    env.close()
    return agent, episode_rewards

def evaluate_agent(agent, preprocessor, episodes: int = 5):
    """Evaluate the agent"""
    env = gym.make('PongNoFrameskip-v4')
    total_reward = 0
    
    for _ in range(episodes):
        state, _ = env.reset()
        state = preprocessor.preprocess_frame(state)
        stacked_frames = deque([state] * 4, maxlen=4)
        state = np.stack(stacked_frames, axis=0)
        
        episode_reward = 0
        done = False
        
        while not done:
            # Greedy action selection
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                q_values = agent.q_network(state_tensor)
                action = q_values.max(1)[1].item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_state = preprocessor.preprocess_frame(next_state)
            stacked_frames.append(next_state)
            state = np.stack(stacked_frames, axis=0)
            
            episode_reward += reward
        
        total_reward += episode_reward
    
    env.close()
    return total_reward / episodes

if __name__ == "__main__":
    # Train the agent
    agent, rewards = train_pong_dqn()
    
    # Plot training curve
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    # Moving average
    window = 100
    moving_avg = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
    plt.plot(moving_avg)
    plt.title(f'Moving Average Rewards (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    plt.tight_layout()
    plt.savefig('pong_training_curves.png')
    plt.show()
    
    # Save final model
    torch.save({
        'model_state_dict': agent.q_network.state_dict(),
        'episode_rewards': rewards,
    }, 'pong_dqn_final.pth')
    print("Final model saved as 'pong_dqn_final.pth'")