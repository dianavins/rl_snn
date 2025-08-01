"""
Install dependencies needed for Pong evaluation
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")
        return False

def main():
    print("Installing dependencies for Pong evaluation...")
    
    packages = [
        "gymnasium[atari]",
        "opencv-python",
        "ale-py"
    ]
    
    success_count = 0
    
    for package in packages:
        print(f"\nInstalling {package}...")
        if install_package(package):
            print(f"✓ Successfully installed {package}")
            success_count += 1
        else:
            print(f"✗ Failed to install {package}")
    
    print(f"\nInstallation summary: {success_count}/{len(packages)} packages installed")
    
    if success_count == len(packages):
        print("All dependencies installed! You can now run:")
        print("python evaluate_sequential_pong.py")
    else:
        print("Some packages failed to install. You can try manually:")
        for package in packages:
            print(f"  pip install {package}")

if __name__ == "__main__":
    main()