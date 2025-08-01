#!/usr/bin/env python3
"""Debug environment availability"""

import gymnasium as gym

print("Checking available Atari environments...")

# Get all registered environments
all_envs = list(gym.envs.registry.keys())

# Filter for Atari/Pong environments
atari_envs = [e for e in all_envs if 'ale' in e.lower() or 'atari' in e.lower()]
pong_envs = [e for e in all_envs if 'pong' in e.lower()]

print(f"\nAll Atari-like environments ({len(atari_envs)}):")
for env in sorted(atari_envs):
    print(f"  {env}")

print(f"\nAll Pong-like environments ({len(pong_envs)}):")
for env in sorted(pong_envs):
    print(f"  {env}")

# Test specific environment creation
test_envs = [
    "PongNoFrameskip-v4",
    "ALE/Pong-v5", 
    "Pong-v4",
    "PongDeterministic-v4"
]

print(f"\nTesting specific environments:")
for env_name in test_envs:
    try:
        env = gym.make(env_name)
        print(f"  ✓ {env_name}: SUCCESS")
        env.close()
    except Exception as e:
        print(f"  ✗ {env_name}: {e}")

# Check if ALE is properly installed
print(f"\nChecking ALE installation:")
try:
    import ale_py
    print(f"  ✓ ale_py version: {ale_py.__version__}")
except ImportError as e:
    print(f"  ✗ ale_py not available: {e}")

try:
    import ale_py.roms
    print(f"  ✓ ale_py.roms available")
except ImportError as e:
    print(f"  ✗ ale_py.roms not available: {e}")

# Check gymnasium version
print(f"\nGymnasium version: {gym.__version__}")

# Try to register ALE if needed
print(f"\nAttempting to register ALE environments...")
try:
    # Method 1: Direct import to trigger registration
    import ale_py
    ale_py.register_all()  
    print("  ✓ ALE environments registered via ale_py.register_all()")
except Exception as e:
    print(f"  ✗ ale_py.register_all() failed: {e}")

try:
    # Method 2: Manual registration
    from ale_py import ALEEnv
    from ale_py.roms import Pong
    gym.register(
        id='ALE/Pong-v5',
        entry_point='ale_py.env:ALEEnv',
        kwargs={'game': 'Pong', 'obs_type': 'rgb', 'frameskip': 1}
    )
    print("  ✓ Manual ALE/Pong-v5 registration attempted")
except Exception as e:
    print(f"  ✗ Manual registration failed: {e}")

# Method 3: Try gymnasium[atari] registration
try:
    gym.envs.registration.load('ALE')
    print("  ✓ ALE namespace loaded")
except Exception as e:
    print(f"  ✗ ALE namespace load failed: {e}")

# Recheck environments after registration attempts
all_envs_after = list(gym.envs.registry.keys())
pong_envs_after = [e for e in all_envs_after if 'pong' in e.lower()]
ale_envs_after = [e for e in all_envs_after if 'ale' in e.lower()]

print(f"\nPong environments after registration attempts ({len(pong_envs_after)}):")
for env in sorted(pong_envs_after):
    print(f"  {env}")

print(f"\nALE environments after registration attempts ({len(ale_envs_after)}):")
for env in sorted(ale_envs_after):
    print(f"  {env}")

# Final test of environment creation
print(f"\nFinal environment creation test:")
for env_name in sorted(set(pong_envs_after + ale_envs_after)):
    try:
        env = gym.make(env_name)
        print(f"  ✓ {env_name}: SUCCESS")
        env.close()
    except Exception as e:
        print(f"  ✗ {env_name}: {e}")