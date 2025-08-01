import torch
import gymnasium as gym
import numpy as np
import cv2
from collections import deque
from sb3_to_sequential_converter import SequentialDQNNetwork

def test_sequential_dqn():
    """Test the Sequential DQN on real Pong environment"""
    print("Testing Sequential DQN on Real Pong")
    print("="*50)
    
    # Load model
    print("Loading Sequential DQN model...")
    checkpoint = torch.load('sequential_pong_dqn.pt', map_location='cpu', weights_only=False)
    model = SequentialDQNNetwork()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully!")
    
    # Create environment
    print("Creating Pong environment...")
    try:
        env = gym.make('PongNoFrameskip-v4')
        print("Environment: PongNoFrameskip-v4")
    except:
        try:
            env = gym.make('ALE/Pong-v5')
            print("Environment: ALE/Pong-v5")
        except Exception as e:
            print(f"Could not create Pong environment: {e}")
            return None
    
    def preprocess_frame(frame):
        """Preprocess frame to match training format"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84))
        return resized.astype(np.float32) / 255.0
    
    rewards = []
    
    # Test 5 episodes
    print(f"\nRunning 5 test episodes...")
    
    for episode in range(5):
        print(f"\nEpisode {episode + 1}:")
        
        # Reset environment
        obs, info = env.reset()
        processed_obs = preprocess_frame(obs)
        frame_stack = deque([processed_obs] * 4, maxlen=4)
        state = np.stack(frame_stack, axis=0)
        
        episode_reward = 0
        steps = 0
        max_steps = 10000
        
        while steps < max_steps:
            # Get action from Sequential DQN
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = model(state_tensor)
                action = q_values.argmax().item()
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update state
            processed_obs = preprocess_frame(obs)
            frame_stack.append(processed_obs)
            state = np.stack(frame_stack, axis=0)
            
            episode_reward += reward
            steps += 1
            
            # Progress update
            if steps % 2000 == 0:
                print(f"  Step {steps}: Current reward = {episode_reward}")
            
            if done:
                break
        
        rewards.append(episode_reward)
        print(f"  Final reward: {episode_reward} (Steps: {steps})")
    
    env.close()
    
    # Results summary
    print(f"\n{'='*50}")
    print("SEQUENTIAL DQN TEST RESULTS")
    print(f"{'='*50}")
    print(f"Episode rewards: {rewards}")
    print(f"Mean reward: {np.mean(rewards):.1f}")
    print(f"Min reward: {min(rewards)}")
    print(f"Max reward: {max(rewards)}")
    
    # Performance assessment
    perfect_games = sum(1 for r in rewards if r >= 21)
    good_games = sum(1 for r in rewards if r >= 15)
    positive_games = sum(1 for r in rewards if r >= 0)
    
    print(f"\nPerformance Analysis:")
    print(f"Perfect games (+21): {perfect_games}/5 ({perfect_games/5*100:.1f}%)")
    print(f"Very good games (+15): {good_games}/5 ({good_games/5*100:.1f}%)")
    print(f"Positive games (≥0): {positive_games}/5 ({positive_games/5*100:.1f}%)")
    
    # Final verdict
    if perfect_games >= 4:
        print(f"\nVERDICT: EXCELLENT! Sequential DQN achieves +21 reward consistently!")
    elif perfect_games >= 2:
        print(f"\nVERDICT: VERY GOOD! Sequential DQN achieves +21 reward frequently!")
    elif good_games >= 3:
        print(f"\nVERDICT: GOOD! Sequential DQN shows strong performance!")
    elif positive_games >= 3:
        print(f"\nVERDICT: MODERATE! Sequential DQN shows positive performance!")
    else:
        print(f"\nVERDICT: POOR! Sequential DQN needs improvement!")
    
    return rewards

if __name__ == "__main__":
    results = test_sequential_dqn()
    
    if results:
        print(f"\nFINAL ANSWER: Sequential DQN achieved rewards of {results}")
        if max(results) >= 21:
            print("The Sequential DQN CAN achieve +21 reward!")
        else:
            print(f"The Sequential DQN achieved maximum reward of {max(results)}")
    else:
        print("Test could not be completed due to environment issues.")