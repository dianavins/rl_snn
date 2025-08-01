import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import gymnasium as gym
from collections import deque
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class DoubleDQN(nn.Module):
    def __init__(self, input_channels: int = 4, n_actions: int = 6):
        super(DoubleDQN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        conv_output_size = 128 * 10 * 10
        
        self.fc1 = nn.Linear(conv_output_size, 1024)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(512, n_actions)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class FrameStack:
    def __init__(self, k=4):
        self.k = k
        self.frames = deque([], maxlen=k)

    def reset(self):
        self.frames.clear()

    def append(self, frame):
        self.frames.append(frame)

    def get(self):
        return np.array(self.frames)


def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = frame[34:194]  # Crop to game area
    frame = cv2.resize(frame, (84, 84))
    frame = frame.astype(np.float32) / 255.0
    return frame


class DoubleDQNAgent:
    def __init__(self, input_channels=4, n_actions=6, lr=3e-4, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.9995,
                 buffer_size=50000, batch_size=64, target_update=500):
        
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        self.q_network = DoubleDQN(input_channels, n_actions).to(device)
        self.target_network = DoubleDQN(input_channels, n_actions).to(device)
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.9)
        
        self.memory = ReplayBuffer(buffer_size)
        
        self.update_target_network()
        self.steps = 0

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.q_network(state)
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.BoolTensor(dones).to(device)
        
        # Reward shaping for Pong
        rewards = torch.clamp(rewards, -1, 1)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values_online = self.q_network(next_states)
            next_actions = next_q_values_online.argmax(dim=1)
            
            next_q_values_target = self.target_network(next_states)
            next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1))
            
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * (~dones).unsqueeze(1))
        
        loss = F.huber_loss(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        if self.steps % self.target_update == 0:
            self.update_target_network()
        
        if self.steps % 1000 == 0:
            self.scheduler.step()
        
        self.steps += 1
        
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()

    def train(self, env, episodes=1000, save_every=200, target_score=20):
        scores = []
        avg_scores = []
        frame_stack = FrameStack(k=4)
        best_avg_score = -float('inf')
        losses = []
        
        # Warmup phase
        print("Warming up replay buffer...")
        obs, _ = env.reset()
        obs = preprocess_frame(obs)
        frame_stack.reset()
        for _ in range(4):
            frame_stack.append(obs)
        state = frame_stack.get()
        
        for _ in range(10000):
            action = random.randrange(self.n_actions)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_obs = preprocess_frame(next_obs)
            frame_stack.append(next_obs)
            next_state = frame_stack.get()
            
            self.remember(state, action, reward, next_state, done)
            state = next_state
            
            if done:
                obs, _ = env.reset()
                obs = preprocess_frame(obs)
                frame_stack.reset()
                for _ in range(4):
                    frame_stack.append(obs)
                state = frame_stack.get()
        
        print("Starting training...")
        for episode in range(episodes):
            obs, _ = env.reset()
            obs = preprocess_frame(obs)
            frame_stack.reset()
            for _ in range(4):
                frame_stack.append(obs)
            state = frame_stack.get()
            
            total_reward = 0
            done = False
            episode_losses = []
            
            while not done:
                action = self.act(state)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                next_obs = preprocess_frame(next_obs)
                frame_stack.append(next_obs)
                next_state = frame_stack.get()
                
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                # Train multiple times per step for faster learning
                for _ in range(2):
                    loss = self.replay()
                    if loss is not None:
                        episode_losses.append(loss)
            
            scores.append(total_reward)
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)
            
            if episode_losses:
                losses.append(np.mean(episode_losses))
            
            if episode % 50 == 0:
                print(f"Episode {episode}, Score: {total_reward:.2f}, Avg Score: {avg_score:.2f}, Epsilon: {self.epsilon:.3f}, Loss: {np.mean(episode_losses) if episode_losses else 0:.4f}")
            
            # Early stopping if target achieved
            if avg_score > target_score:
                print(f"Target score {target_score} achieved at episode {episode}!")
                torch.save(self.q_network.state_dict(), f'double_dqn_pong_best.pth')
                break
            
            # Save best model
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                torch.save(self.q_network.state_dict(), f'double_dqn_pong_best.pth')
            
            if episode % save_every == 0 and episode > 0:
                torch.save(self.q_network.state_dict(), f'double_dqn_pong_{episode}.pth')
        
        return scores, avg_scores, losses

    def evaluate(self, env, episodes=10, render=False):
        self.q_network.eval()
        scores = []
        frame_stack = FrameStack(k=4)
        
        for episode in range(episodes):
            obs, _ = env.reset()
            obs = preprocess_frame(obs)
            frame_stack.reset()
            for _ in range(4):
                frame_stack.append(obs)
            state = frame_stack.get()
            
            total_reward = 0
            done = False
            
            while not done:
                action = self.act(state, training=False)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                next_obs = preprocess_frame(next_obs)
                frame_stack.append(next_obs)
                state = frame_stack.get()
                total_reward += reward
                
                if render:
                    env.render()
            
            scores.append(total_reward)
            print(f"Evaluation Episode {episode + 1}: Score = {total_reward}")
        
        self.q_network.train()
        avg_score = np.mean(scores)
        print(f"Average evaluation score: {avg_score:.2f}")
        return scores, avg_score


def main():
    env = gym.make('ALE/Pong-v5')
    n_actions = env.action_space.n
    
    agent = DoubleDQNAgent(input_channels=4, n_actions=n_actions)
    
    print("Starting optimized Double DQN training...")
    scores, avg_scores, losses = agent.train(env, episodes=1000, target_score=21)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(scores)
    plt.title('Training Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.subplot(1, 3, 2)
    plt.plot(avg_scores)
    plt.title('Average Scores (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.axhline(y=21, color='r', linestyle='--', label='Target Score')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    if losses:
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('double_dqn_optimized_results.png', dpi=150)
    plt.show()
    
    torch.save(agent.q_network.state_dict(), 'double_dqn_pong_final.pth')
    
    print("Evaluating trained agent...")
    eval_scores, avg_eval_score = agent.evaluate(env, episodes=10)
    
    env.close()


if __name__ == "__main__":
    main()