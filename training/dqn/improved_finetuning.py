import copy
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from spikingjelly.clock_driven import functional as sf_func
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# ─── Improved Hyperparameters ──────────────────────────────────────────────────
ENV_ID         = "PongNoFrameskip-v4"
NUM_EPISODES   = 300        # Reduced from 500
GAMMA          = 0.99
LR             = 5e-5       # Reduced for stable finetuning
TARGET_SYNC    = 1000       # Sync by frames, not episodes
BUFFER_SIZE    = 50_000     # Smaller buffer for faster learning
BATCH_SIZE     = 64         # Increased batch size
MIN_REPLAY     = 5_000      # Reduced minimum replay
EPS_START      = 0.3        # Lower initial exploration
EPS_END        = 0.05       # Higher final exploration  
EPS_DECAY      = 30_000     # Faster epsilon decay
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Environment setup with reward shaping ────────────────────────────────────
class PongRewardWrapper:
    def __init__(self, env):
        self.env = env
        self.last_lives = None
        self.last_score_diff = 0
        
    def reset(self):
        obs = self.env.reset()
        self.last_lives = None
        self.last_score_diff = 0
        return obs
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # Extract game info from Atari
        if hasattr(info[0], 'ale.lives'):
            lives = info[0]['ale.lives']
        else:
            lives = 5  # Default Pong lives
            
        # Reward shaping for better learning
        shaped_reward = reward[0]
        
        # Small positive reward for staying alive
        if not done[0]:
            shaped_reward += 0.01
            
        # Penalty for losing life (ball going past paddle)
        if self.last_lives is not None and lives < self.last_lives:
            shaped_reward -= 0.1
            
        self.last_lives = lives
        
        return obs, [shaped_reward], done, info
        
    def __getattr__(self, name):
        return getattr(self.env, name)

env = make_atari_env(ENV_ID, n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)
env = PongRewardWrapper(env)
action_dim = env.action_space.n

def unwrap(obs_tuple):
    return obs_tuple[0]

# ─── Load pretrained model without destroying weights ──────────────────────────
ann_model_path = "/PongNoFrameskip-v4.zip"  # Update path as needed
ann_model = DQN.load(ann_model_path, custom_objects={"replay_buffer_class": None, "optimize_memory_usage": False})

# Don't reinitialize - keep pretrained weights!
q_net = ann_model.policy.q_net
target_net = ann_model.policy.q_net_target

# Only freeze feature extractor, allow Q-head to adapt
for param in q_net.features_extractor.parameters():
    param.requires_grad = False
for param in target_net.features_extractor.parameters():
    param.requires_grad = False

# ─── Prioritized Experience Replay (simplified) ───────────────────────────────
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done", "priority"])

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0
        
    def add(self, *args, priority=1.0):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            self.priorities.append(0)
        
        self.buffer[self.pos] = Transition(*args, priority)
        self.priorities[self.pos] = priority ** self.alpha
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) < batch_size:
            return None
            
        priorities = np.array(self.priorities[:len(self.buffer)])
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        
        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, weights, indices
        
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (priority + 1e-6) ** self.alpha
            
    def __len__(self):
        return len(self.buffer)

replay_buffer = PrioritizedReplayBuffer(BUFFER_SIZE)

# ─── Optimized optimizer with learning rate scheduling ────────────────────────
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, q_net.parameters()), 
    lr=LR, 
    eps=1e-4,
    weight_decay=1e-5
)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

criterion = nn.SmoothL1Loss(reduction='none')  # For prioritized replay

# ─── Improved epsilon schedule ─────────────────────────────────────────────────
def epsilon_by_frame(frame_idx):
    return EPS_END + (EPS_START - EPS_END) * np.exp(-1.0 * frame_idx / EPS_DECAY)

# ─── Optimized Q-value computation ─────────────────────────────────────────────
def compute_q_values(net, obs_batch):
    """Batch process observations for efficiency"""
    # Convert to tensor and normalize
    if not isinstance(obs_batch, torch.Tensor):
        obs_batch = torch.tensor(obs_batch, dtype=torch.float32)
    
    obs_batch = obs_batch.permute(0, 3, 1, 2).to(DEVICE) / 255.0
    
    with torch.no_grad():
        sf_func.reset_net(net)
    
    return net(obs_batch)

# ─── Pre-fill replay buffer ────────────────────────────────────────────────────
print("Pre-filling replay buffer...")
obs = unwrap(env.reset())
for i in range(MIN_REPLAY):
    if i % 1000 == 0:
        print(f"Pre-filling: {i}/{MIN_REPLAY}")
        
    action = env.action_space.sample()
    next_obs, reward, done, _ = env.step([action])
    next_obs, reward, done = next_obs[0], reward[0], done[0]
    
    replay_buffer.add(obs, action, reward, next_obs, done)
    obs = next_obs if not done else unwrap(env.reset())

# ─── Main training loop with improvements ──────────────────────────────────────
frame_idx = 0
best_reward = -21
episode_rewards = deque(maxlen=20)  # Track recent performance
beta = 0.4
beta_increment = (1.0 - beta) / NUM_EPISODES

print("Starting training...")
for ep in range(1, NUM_EPISODES + 1):
    obs = unwrap(env.reset())
    total_reward = 0
    done = False
    episode_loss = 0
    steps_in_episode = 0

    sf_func.reset_net(q_net)
    sf_func.reset_net(target_net)

    while not done:
        frame_idx += 1
        steps_in_episode += 1
        eps = epsilon_by_frame(frame_idx)

        # ε-greedy action selection with batch processing
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = compute_q_values(q_net, obs[None])
                action = q_values.argmax(dim=1).item()

        # Step environment
        next_obs, reward, done, _ = env.step([action])
        next_obs, reward, done = next_obs[0], reward[0], done[0]
        
        # Add to replay buffer with initial priority
        td_error = abs(reward)  # Simple initial priority
        replay_buffer.add(obs, action, reward, next_obs, done, priority=td_error)
        
        obs = next_obs
        total_reward += reward

        # Training step with prioritized experience replay
        if len(replay_buffer) >= MIN_REPLAY and frame_idx % 4 == 0:  # Train every 4 frames
            batch_data = replay_buffer.sample(BATCH_SIZE, beta)
            if batch_data is not None:
                transitions, weights, indices = batch_data
                batch = Transition(*(zip(*[(t.state, t.action, t.reward, t.next_state, t.done) for t in transitions])))
                weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

                state_batch = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch.state])
                next_state_batch = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch.next_state])
                action_batch = torch.tensor(batch.action, dtype=torch.int64, device=DEVICE).unsqueeze(1)
                reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=DEVICE).unsqueeze(1)
                done_batch = torch.tensor(batch.done, dtype=torch.float32, device=DEVICE).unsqueeze(1)

                # Current Q-values
                current_q = compute_q_values(q_net, state_batch).gather(1, action_batch)

                # Double DQN target computation
                with torch.no_grad():
                    next_q_online = compute_q_values(q_net, next_state_batch)
                    next_actions = next_q_online.argmax(dim=1, keepdim=True)
                    next_q_target = compute_q_values(target_net, next_state_batch)
                    next_q = next_q_target.gather(1, next_actions)
                    td_target = reward_batch + GAMMA * (1 - done_batch) * next_q

                # Compute loss with importance sampling
                td_errors = criterion(current_q, td_target)
                loss = (td_errors * weights.unsqueeze(1)).mean()

                # Update priorities
                priorities = td_errors.detach().cpu().numpy().flatten()
                replay_buffer.update_priorities(indices, priorities)

                # Optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)  # Gradient clipping
                optimizer.step()

                episode_loss += loss.item()

        # Update target network by frame count
        if frame_idx % TARGET_SYNC == 0:
            target_net.load_state_dict(q_net.state_dict())

    # Update beta for prioritized replay
    beta = min(1.0, beta + beta_increment)
    
    # Step learning rate scheduler
    if ep % 10 == 0:
        scheduler.step()

    episode_rewards.append(total_reward)
    avg_reward = np.mean(episode_rewards)
    
    # Save best model
    if total_reward > best_reward:
        best_reward = total_reward
        torch.save(q_net.state_dict(), "best_ann_finetuned.pth")

    print(f"Episode {ep:03d}  Reward: {total_reward:6.1f}  Avg: {avg_reward:6.1f}  "
          f"Epsilon: {eps:.3f}  Loss: {episode_loss/max(steps_in_episode//4, 1):.4f}  "
          f"LR: {optimizer.param_groups[0]['lr']:.2e}")

    # Save checkpoint every 25 episodes
    if ep % 25 == 0:
        torch.save({
            'episode': ep,
            'model_state_dict': q_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_reward': best_reward,
            'frame_idx': frame_idx
        }, f"checkpoint_ep{ep}.pth")

print(f"Training completed! Best reward: {best_reward}")