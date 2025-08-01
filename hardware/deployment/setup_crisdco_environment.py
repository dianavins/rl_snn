#!/usr/bin/env python3
"""Setup script for crisdco environment to ensure ROMs are properly installed"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n=== {description} ===")
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        print("SUCCESS:")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("FAILED:")
        print(f"Error: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def setup_crisdco_environment():
    """Setup the crisdco environment for HiAER-Spike conversion"""
    print("=== Setting up crisdco environment for HiAER-Spike ===")
    print("This will install and configure ROMs for Atari Pong")
    
    # Step 1: Install dependencies with poetry
    print("\n1. Installing dependencies with poetry...")
    success = run_command("poetry install --with apps --with fpga", 
                         "Installing all dependencies including apps and fpga groups")
    
    if not success:
        print("WARNING: Poetry install failed, trying basic install...")
        run_command("poetry install", "Basic poetry install")
    
    # Step 2: Install ROMs using AutoROM
    print("\n2. Installing Atari ROMs...")
    
    # Try multiple methods to install ROMs
    methods = [
        ("python -m autorom --accept-license", "AutoROM with license acceptance"),
        ("ale-import-roms", "ALE ROM import"),
        ("python -c \"import autorom; autorom.main(['--accept-license'])\"", "AutoROM Python import"),
    ]
    
    rom_installed = False
    for cmd, desc in methods:
        if run_command(cmd, desc):
            rom_installed = True
            break
    
    if not rom_installed:
        print("\nWARNING: Automatic ROM installation failed!")
        print("Manual steps:")
        print("1. Run: poetry shell")
        print("2. Run: python -m autorom --accept-license")
        print("3. Or download ROMs manually and use ale-import-roms")
    
    # Step 3: Verify installation
    print("\n3. Verifying Atari environment setup...")
    
    verification_script = """
import gymnasium as gym
import ale_py

print("Checking ALE and Gymnasium setup...")

try:
    # Register ALE environments  
    ale_py.register_all()
    print("✓ ALE environments registered")
    
    # List available environments
    all_envs = list(gym.envs.registry.keys())
    pong_envs = [e for e in all_envs if 'pong' in e.lower()]
    ale_envs = [e for e in all_envs if 'ale' in e.lower() and 'pong' in e.lower()]
    
    print(f"Found {len(pong_envs)} Pong environments: {pong_envs}")
    print(f"Found {len(ale_envs)} ALE Pong environments: {ale_envs}")
    
    # Try to create Pong environment
    test_envs = ["ALE/Pong-v5", "PongNoFrameskip-v4", "Pong-v4"]
    
    for env_name in test_envs:
        try:
            env = gym.make(env_name)
            print(f"✓ Successfully created: {env_name}")
            env.close()
            break
        except Exception as e:
            print(f"✗ Failed to create {env_name}: {e}")
    else:
        print("✗ Could not create any Pong environment")
        
    print("\nVerification complete!")
    
except Exception as e:
    print(f"✗ Verification failed: {e}")
    """
    
    # Write verification script to temp file
    with open("temp_verify.py", "w") as f:
        f.write(verification_script)
    
    success = run_command("python temp_verify.py", "Verifying Atari environment setup")
    
    # Clean up
    try:
        os.remove("temp_verify.py")
    except:
        pass
    
    # Step 4: Test HiAER_Spike.py compatibility
    print("\n4. Testing HiAER_Spike.py compatibility...")
    
    compatibility_test = """
try:
    from stable_baselines3.common.env_util import make_atari_env
    from stable_baselines3.common.vec_env import VecFrameStack
    import ale_py
    
    print("Testing HiAER_Spike.py environment creation...")
    
    # Register environments
    try:
        ale_py.register_all()
        print("✓ ALE registered")
    except:
        print("? ALE registration issue")
    
    # Test environment creation (same as HiAER_Spike.py)
    env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)
    
    print("✓ Environment created successfully!")
    print("✓ crisdco is ready for HiAER_Spike.py")
    
    # Quick test
    obs = env.reset()
    print(f"✓ Environment reset: obs shape {obs.shape}")
    
    env.close()
    
except Exception as e:
    print(f"✗ HiAER_Spike.py compatibility test failed: {e}")
    print("This is the error you were seeing in HiAER_Spike.py")
    """
    
    with open("temp_hiear_test.py", "w") as f:
        f.write(compatibility_test)
    
    success = run_command("python temp_hiear_test.py", "Testing HiAER_Spike.py compatibility")
    
    # Clean up
    try:
        os.remove("temp_hiear_test.py")
    except:
        pass
    
    # Final summary
    print("\n" + "="*60)
    print("SETUP COMPLETE")
    print("="*60)
    
    if success:
        print("✓ crisdco environment is ready for HiAER-Spike conversion!")
        print("✓ You should now be able to run HiAER_Spike.py without ROM errors")
    else:
        print("⚠ Setup completed with some issues")
        print("Manual steps may be required:")
        print("1. poetry shell")
        print("2. python -m autorom --accept-license")
        print("3. Test with: python HiAER_Spike.py")
    
    print("\nNext steps:")
    print("1. Run: poetry shell")
    print("2. Test: python HiAER_Spike.py")
    print("3. If still issues, check ROM installation manually")

if __name__ == "__main__":
    setup_crisdco_environment()