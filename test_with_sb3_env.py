"""
Test Sequential DQN using stable-baselines3 environment setup
This should work with your existing SB3 installation
"""

import torch
import numpy as np
from collections import deque
import cv2
from sb3_to_sequential_converter import SequentialDQNNetwork

def test_sequential_dqn_real_pong():
    """Test Sequential DQN on actual Pong using SB3-style setup"""
    print("Testing Sequential DQN on Real Pong")
    print("="*50)
    
    # Load Sequential DQN model
    print("Loading Sequential DQN model...")
    try:
        checkpoint = torch.load('sequential_pong_dqn.pt', map_location='cpu', weights_only=False)
        model = SequentialDQNNetwork()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("✓ Sequential DQN loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None
    
    # Try different ways to create Pong environment
    env = None
    env_name = None
    
    print("\nAttempting to create Pong environment...")
    
    # First, try to register ALE environments
    print("Registering ALE environments...")
    try:
        import ale_py
        ale_py.register_all()
        print("✓ ALE environments registered")
    except Exception as e:
        print(f"? ALE registration issue: {e}")
    
    # Method 1: Try with stable-baselines3 environment utilities (primary method)
    try:
        from stable_baselines3.common.env_util import make_atari_env
        from stable_baselines3.common.vec_env import VecFrameStack
        
        print("Using stable-baselines3 environment creation...")
        
        # Try different environment names that might work
        env_names_to_try = [
            "PongNoFrameskip-v4",
            "ALE/Pong-v5", 
            "Pong-v4",
            "PongDeterministic-v4"
        ]
        
        for env_id in env_names_to_try:
            try:
                print(f"  Trying {env_id}...")
                env = make_atari_env(env_id, n_envs=1, seed=0)
                env = VecFrameStack(env, n_stack=4)
                env_name = f"{env_id} (via SB3)"
                print(f"✓ Successfully created: {env_name}")
                break
            except Exception as e:
                print(f"  ✗ {env_id} failed: {e}")
        
        if env is None:
            raise Exception("No SB3 environment creation succeeded")
        
    except Exception as e:
        print(f"✗ All SB3 environment creation attempts failed: {e}")
    
    # Method 2: Fallback options
    if env is None:
        print(f"\nPrimary SB3 method failed. Trying fallback options...")
        try:
            import gymnasium as gym
            print("Available environments:")
            all_envs = list(gym.envs.registry.keys())
            pong_like = [e for e in all_envs if 'pong' in e.lower() or 'ale' in e.lower()]
            print(f"Pong-like environments: {pong_like}")
            
            if pong_like:
                try:
                    env = gym.make(pong_like[0])
                    env_name = pong_like[0]
                    print(f"✓ Using: {env_name}")
                except Exception as e:
                    print(f"✗ Failed to create {pong_like[0]}: {e}")
        except Exception as e:
            print(f"✗ Fallback failed: {e}")
    
    if env is None:
        print("\n✗ Could not create any Pong environment")
        print("\nTo enable testing, you may need to:")
        print("1. Install ROMs: python -m autorom --accept-license")
        print("2. Or install: pip install stable-baselines3[extra]")
        return None
    
    # Run the actual test
    print(f"\nRunning Sequential DQN test on: {env_name}")
    
    try:
        # Preprocessing function
        def preprocess_obs(obs):
            """Preprocess observation to match training format"""
            if hasattr(obs, 'shape') and len(obs.shape) == 4:
                # Already stacked frames from VecFrameStack
                return obs[0]  # Remove batch dimension
            else:
                # Single frame, need to process
                if len(obs.shape) == 3:
                    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
                else:
                    gray = obs
                
                resized = cv2.resize(gray, (84, 84))
                normalized = resized.astype(np.float32) / 255.0
                
                # Stack 4 copies for initial state
                return np.stack([normalized] * 4, axis=0)
        
        # Reset environment
        if hasattr(env, 'reset'):
            if 'VecFrameStack' in str(type(env)):
                obs = env.reset()
                state = obs[0]  # VecEnv returns array
            else:
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]  # New gym format returns (obs, info)
                state = preprocess_obs(obs)
        else:
            print("✗ Environment doesn't have reset method")
            return None
        
        print(f"✓ Environment reset, state shape: {state.shape}")
        
        # Run one episode
        episode_reward = 0
        steps = 0
        max_steps = 5000  # Reasonable limit
        done = False
        
        print("Starting episode...")
        
        while not done and steps < max_steps:
            # Get action from Sequential DQN
            with torch.no_grad():
                if len(state.shape) == 3:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    
                q_values = model(state_tensor)
                action = q_values.argmax().item()
            
            # Take step
            if 'VecFrameStack' in str(type(env)):
                obs, reward, done, info = env.step([action])
                reward = reward[0]
                done = done[0]
                state = obs[0]
            else:
                step_result = env.step(action)
                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                else:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                
                state = preprocess_obs(obs)
            
            episode_reward += reward
            steps += 1
            
            # Progress updates
            if steps % 1000 == 0:
                print(f"  Step {steps}: Reward = {episode_reward}")
        
        # Clean up
        env.close()
        
        # Report results
        print("\n" + "="*50)
        print("ACTUAL PONG TEST RESULTS")
        print("="*50)
        print(f"Environment: {env_name}")
        print(f"Final reward: {episode_reward}")
        print(f"Steps taken: {steps}")
        print(f"Episode completed: {'Yes' if done else 'No (timeout)'}")
        
        # Assessment
        if episode_reward >= 21:
            print("\n🎉 SUCCESS: Sequential DQN achieved +21 reward!")
            print("The conversion is working perfectly!")
        elif episode_reward >= 15:
            print(f"\n🎯 VERY GOOD: Achieved {episode_reward} reward (close to perfect)")
        elif episode_reward >= 5:
            print(f"\n✅ GOOD: Achieved {episode_reward} reward (positive performance)")
        elif episode_reward >= 0:
            print(f"\n👍 OKAY: Achieved {episode_reward} reward (non-negative)")
        else:
            print(f"\n⚠️ POOR: Achieved {episode_reward} reward (negative)")
        
        print(f"\nFINAL ANSWER: Sequential DQN achieved {episode_reward} reward on real Pong")
        
        return episode_reward
        
    except Exception as e:
        print(f"\n✗ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_sequential_dqn_real_pong()
    
    if result is not None:
        print(f"\nCONCLUSION: The Sequential DQN achieved {result} reward on actual Pong.")
        if result >= 21:
            print("This confirms the +21 reward claim!")
        else:
            print("This shows the actual performance level.")
    else:
        print("\nCould not complete the test due to environment setup issues.")
        print("The Sequential DQN is ready, but needs a working Pong environment.")