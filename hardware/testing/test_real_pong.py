"""
Test Sequential DQN on real Pong - simplified version
"""

import subprocess
import sys
import os

def install_rom_package():
    """Install ROM package"""
    print("Installing AutoROM...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "autorom[accept-rom-license]"])
        print("AutoROM installed successfully")
        return True
    except:
        print("AutoROM installation failed")
        return False

def download_roms():
    """Download Atari ROMs"""
    print("Downloading Atari ROMs...")
    try:
        subprocess.check_call([sys.executable, "-m", "autorom", "--accept-license"])
        print("ROMs downloaded successfully")
        return True
    except:
        print("ROM download failed")
        return False

def test_pong_environment():
    """Test if Pong environment works"""
    try:
        import gymnasium as gym
        
        # Try different environment names
        env_names = ["ALE/Pong-v5", "PongNoFrameskip-v4"]
        
        for env_name in env_names:
            try:
                print(f"Testing {env_name}...")
                env = gym.make(env_name)
                obs, info = env.reset()
                print(f"SUCCESS: {env_name} works!")
                print(f"Observation shape: {obs.shape}")
                env.close()
                return env_name
            except Exception as e:
                print(f"FAILED: {env_name} - {e}")
        
        return None
        
    except Exception as e:
        print(f"Environment test failed: {e}")
        return None

def run_actual_pong_test(env_name):
    """Run actual Pong test"""
    print(f"Running Sequential DQN on {env_name}...")
    
    try:
        import torch
        import gymnasium as gym
        import numpy as np
        from collections import deque
        import cv2
        from conversion.sb3_to_sequential.sb3_to_sequential_converter import SequentialDQNNetwork
        
        # Load model
        checkpoint = torch.load('sequential_pong_dqn.pt', map_location='cpu', weights_only=False)
        model = SequentialDQNNetwork()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Model loaded successfully")
        
        # Create environment
        env = gym.make(env_name)
        
        # Preprocessing function
        def preprocess_frame(frame):
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame
            resized = cv2.resize(gray, (84, 84))
            return resized.astype(np.float32) / 255.0
        
        # Run one episode
        obs, info = env.reset()
        processed_obs = preprocess_frame(obs)
        frame_stack = deque([processed_obs] * 4, maxlen=4)
        state = np.stack(frame_stack, axis=0)
        
        episode_reward = 0
        steps = 0
        max_steps = 3000
        
        print("Starting Pong game...")
        
        while steps < max_steps:
            # Get action from Sequential DQN
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = model(state_tensor)
                action = q_values.argmax().item()
            
            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update state
            processed_obs = preprocess_frame(obs)
            frame_stack.append(processed_obs)
            state = np.stack(frame_stack, axis=0)
            
            episode_reward += reward
            steps += 1
            
            # Progress update
            if steps % 1000 == 0:
                print(f"Step {steps}: Current reward = {episode_reward}")
            
            if done:
                break
        
        env.close()
        
        # Results
        print("="*50)
        print("ACTUAL PONG TEST RESULTS")
        print("="*50)
        print(f"Environment: {env_name}")
        print(f"Final reward: {episode_reward}")
        print(f"Steps taken: {steps}")
        print(f"Game finished: {'Yes' if done else 'No (timeout)'}")
        
        # Performance assessment
        if episode_reward >= 21:
            print("RESULT: PERFECT! Sequential DQN achieved +21 reward!")
            print("This confirms the conversion was successful.")
        elif episode_reward >= 15:
            print("RESULT: EXCELLENT! Very high performance.")
        elif episode_reward >= 5:
            print("RESULT: GOOD! Positive performance achieved.")
        elif episode_reward >= 0:
            print("RESULT: MODERATE! Non-negative performance.")
        else:
            print("RESULT: POOR! Negative performance.")
        
        print(f"\nACTUAL REWARD: {episode_reward}")
        print("This is the real performance, not a simulation.")
        
        return episode_reward
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("TESTING SEQUENTIAL DQN ON REAL PONG")
    print("="*50)
    
    # Check if model exists
    if not os.path.exists('sequential_pong_dqn.pt'):
        print("ERROR: Sequential DQN model not found!")
        return
    
    # Step 1: Install ROM package if needed
    try:
        import autorom
        print("AutoROM already available")
    except ImportError:
        if not install_rom_package():
            print("Cannot proceed without AutoROM")
            return
    
    # Step 2: Download ROMs if needed
    if not download_roms():
        print("Trying to continue without ROM download...")
    
    # Step 3: Test environment
    working_env = test_pong_environment()
    
    if working_env:
        # Step 4: Run actual test
        reward = run_actual_pong_test(working_env)
        
        if reward is not None:
            print("\n" + "="*50)
            print("FINAL ANSWER")
            print("="*50)
            print(f"Sequential DQN actual Pong reward: {reward}")
            
            if reward >= 21:
                print("YES - Sequential DQN achieves +21 reward!")
            else:
                print(f"NO - Sequential DQN achieves {reward} reward (not +21)")
        else:
            print("Test could not be completed due to errors")
    else:
        print("Could not set up Pong environment")
        print("\nTo test manually:")
        print("1. pip install 'autorom[accept-rom-license]'")
        print("2. python -m autorom --accept-license")
        print("3. python -c \"import gymnasium as gym; env = gym.make('ALE/Pong-v5'); print('Success')\"")
        print("4. python test_real_pong.py")

if __name__ == "__main__":
    main()