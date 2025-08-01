"""
Setup script to install Pong environment and test the Sequential DQN
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def setup_atari_environment():
    """Set up the Atari environment step by step"""
    print("Setting up Atari Pong environment...")
    
    # Step 1: Install required packages
    packages = [
        "gymnasium[atari]",
        "ale-py", 
        "autorom[accept-rom-license]"
    ]
    
    print("\n1. Installing required packages...")
    for package in packages:
        print(f"   Installing {package}...")
        if install_package(package):
            print(f"   ✓ {package} installed successfully")
        else:
            print(f"   ✗ Failed to install {package}")
            return False
    
    # Step 2: Download ROMs
    print("\n2. Downloading Atari ROMs...")
    try:
        import autorom
        autorom.main(["--accept-license"])  # Accept license and download
        print("   ✓ ROMs downloaded successfully")
    except Exception as e:
        print(f"   ✗ ROM download failed: {e}")
        print("   Trying alternative method...")
        
        # Alternative: command line approach
        try:
            subprocess.check_call([sys.executable, "-m", "autorom", "--accept-license"])
            print("   ✓ ROMs downloaded via command line")
        except subprocess.CalledProcessError:
            print("   ✗ ROM download failed via command line too")
            return False
    
    # Step 3: Test environment
    print("\n3. Testing Pong environment...")
    try:
        import gymnasium as gym
        
        # Try different Pong environment names
        env_names = [
            "ALE/Pong-v5",
            "PongNoFrameskip-v4", 
            "Pong-v4",
            "PongDeterministic-v4"
        ]
        
        working_env = None
        for env_name in env_names:
            try:
                env = gym.make(env_name)
                print(f"   ✓ {env_name} works!")
                working_env = env_name
                
                # Quick test
                obs, info = env.reset()
                print(f"   ✓ Observation shape: {obs.shape}")
                print(f"   ✓ Action space: {env.action_space}")
                env.close()
                break
                
            except Exception as e:
                print(f"   ✗ {env_name} failed: {e}")
        
        if working_env:
            print(f"\n✓ SUCCESS: {working_env} is ready for testing!")
            return working_env
        else:
            print("\n✗ No Pong environments are working")
            return None
            
    except ImportError:
        print("   ✗ Gymnasium not available")
        return None

def test_sequential_dqn_on_pong(env_name):
    """Test the Sequential DQN on actual Pong"""
    print(f"\nTesting Sequential DQN on {env_name}...")
    
    try:
        import torch
        import gymnasium as gym
        import numpy as np
        from collections import deque
        import cv2
        from sb3_to_sequential_converter import SequentialDQNNetwork
        
        # Load model
        print("Loading Sequential DQN model...")
        checkpoint = torch.load('sequential_pong_dqn.pt', map_location='cpu', weights_only=False)
        model = SequentialDQNNetwork()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("✓ Model loaded")
        
        # Create environment
        env = gym.make(env_name)
        print(f"✓ Environment created: {env_name}")
        
        # Simple preprocessing function
        def preprocess_frame(frame):
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame
            resized = cv2.resize(gray, (84, 84))
            return resized.astype(np.float32) / 255.0
        
        # Run one test episode
        print("\nRunning test episode...")
        obs, info = env.reset()
        processed_obs = preprocess_frame(obs)
        
        # Stack 4 frames
        frame_stack = deque([processed_obs] * 4, maxlen=4)
        state = np.stack(frame_stack, axis=0)
        
        episode_reward = 0
        steps = 0
        max_steps = 2000
        
        while steps < max_steps:
            # Get action from model
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
            
            # Print progress
            if steps % 500 == 0:
                print(f"   Step {steps}: Reward = {episode_reward}")
            
            if done:
                break
        
        env.close()
        
        print(f"\n--- ACTUAL PONG TEST RESULTS ---")
        print(f"Environment: {env_name}")
        print(f"Episode reward: {episode_reward}")
        print(f"Steps taken: {steps}")
        print(f"Game completed: {'Yes' if done else 'No (timeout)'}")
        
        # Assessment
        if episode_reward >= 21:
            print("🎉 PERFECT! Achieved +21 reward (perfect game)!")
        elif episode_reward >= 15:
            print("🎯 EXCELLENT! High positive reward achieved!")
        elif episode_reward >= 5:
            print("✅ GOOD! Positive reward achieved!")
        elif episode_reward >= 0:
            print("👍 DECENT! Non-negative performance!")
        else:
            print("⚠️ NEEDS WORK! Negative reward...")
        
        return episode_reward
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("="*60)
    print("SEQUENTIAL DQN PONG ENVIRONMENT SETUP & TESTING")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists('sequential_pong_dqn.pt'):
        print("✗ Sequential DQN model not found!")
        print("Run: python sb3_to_sequential_converter.py")
        return
    
    # Setup environment
    working_env = setup_atari_environment()
    
    if working_env:
        # Test on actual Pong
        reward = test_sequential_dqn_on_pong(working_env)
        
        if reward is not None:
            print(f"\n{'='*60}")
            print("FINAL RESULT")
            print(f"{'='*60}")
            print(f"Actual Pong reward: {reward}")
            
            if reward >= 21:
                print("SUCCESS: Sequential DQN achieves +21 reward!")
            else:
                print(f"RESULT: Sequential DQN achieved {reward} reward")
                print("This is the actual performance, not a simulation.")
        else:
            print("\n✗ Testing failed - could not evaluate on Pong")
    else:
        print("\n✗ Environment setup failed")
        print("\nManual setup instructions:")
        print("1. pip install 'gymnasium[atari]' ale-py 'autorom[accept-rom-license]'")
        print("2. python -m autorom --accept-license")
        print("3. python evaluate_sequential_pong.py")

if __name__ == "__main__":
    main()