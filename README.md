# Reinforcement Learning with Spiking Neural Networks (RL-SNN)

A comprehensive project for training and deploying Spiking Neural Networks (SNNs) for Atari Pong using the HiAER Spike neuromorphic computing platform. This project converts traditional Deep Q-Network (DQN) models to SNNs and quantizes them for efficient hardware deployment.

## 🎯 Overview

This repository implements a complete pipeline for:
- Training Deep Q-Networks (DQN) with advanced techniques for Atari Pong
- Converting trained models from ANN to SNN format
- Quantizing networks for neuromorphic hardware deployment
- Deploying models on the HiAER Spike neuromorphic computing platform

### Key Features

- **Advanced DQN Training**: Enhanced with reward shaping, prioritized experience replay, and plateau detection
- **ANN to SNN Conversion**: Seamless conversion pipeline with validation
- **Hardware Optimization**: Quantization and optimization for neuromorphic deployment
- **Comprehensive Validation**: End-to-end testing and performance analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.10+ (< 3.11)
- Poetry for dependency management
- External dependencies: `connectome_utils`, `fxpmath`, `hs_bridge` (located in parent directories)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rl_snn
   ```

2. **Install dependencies**:
   ```bash
   # Basic installation
   poetry install
   
   # With ML/visualization dependencies
   poetry install --with apps
   
   # With FPGA bridge dependencies
   poetry install --with fpga
   
   # With development tools
   poetry install --with dev
   ```

3. **Install Atari ROMs** (required for Pong environment):
   ```bash
   python install_pong_deps.py
   ```

### Quick Usage

```bash
# Run the main conversion pipeline
python conversion/ann_to_snn/HiAER_Spike.py

# Train an improved DQN model
python training/dqn/improved_finetuning.py

# Convert Stable-Baselines3 model to sequential format
python conversion/sb3_to_sequential/demo_conversion.py
```

## 📁 Project Structure

```
rl_snn/
├── conversion/                 # Model conversion utilities
│   ├── ann_to_snn/            # ANN to SNN conversion
│   ├── sb3_to_sequential/     # SB3 to PyTorch conversion
│   └── validation/            # Conversion validation tools
├── training/                   # Training scripts and checkpoints
│   ├── dqn/                   # DQN training implementations
│   ├── sequential/            # Sequential model training
│   ├── advanced/              # Advanced training techniques
│   └── checkpoints/           # Model checkpoints
├── hardware/                   # Hardware deployment
│   ├── hs_api/                # HiAER Spike API
│   ├── deployment/            # Deployment scripts
│   └── testing/               # Hardware testing
├── models/                     # Trained models
│   ├── pretrained/            # Pre-trained ANN models
│   ├── sequential/            # Sequential PyTorch models
│   ├── snn/                   # Spiking neural network models
│   └── converted/             # Hardware-converted models
└── notebooks/                  # Jupyter notebooks for experimentation
```

## 🎮 Training Models

### Basic DQN Training

```bash
# Train from scratch
python training/dqn/ann_from_scratch.py

# Improved training with advanced techniques
python training/dqn/improved_finetuning.py
```

### Advanced Features

The improved training includes:
- **Enhanced Reward Shaping**: Game-state aware rewards with ball tracking
- **Adaptive Exploration**: Performance-based epsilon adjustment
- **Prioritized Experience Replay**: Importance sampling with dynamic priorities
- **Plateau Detection**: Automatic detection and counter-measures
- **Advanced Network Updates**: Soft target updates with improved stability

### Training Sequential Models

```bash
# Create sequential model from Stable-Baselines3
python conversion/sb3_to_sequential/demo_conversion.py

# Train sequential model from scratch
python training/sequential/train_sequential_from_scratch.py
```

## ⚙️ Model Conversion

### ANN to SNN Conversion

```bash
# Main conversion pipeline
python conversion/ann_to_snn/HiAER_Spike.py

# Direct PyTorch to HiAER conversion
python conversion/ann_to_snn/direct_pytorch_to_hiaer.py

# Benchmark conversion performance
python conversion/ann_to_snn/benchmark_conversion.py
```

### SB3 to Sequential Conversion

```bash
# Convert Stable-Baselines3 DQN to PyTorch Sequential
python conversion/sb3_to_sequential/demo_conversion.py

# Debug SB3 architecture
python conversion/sb3_to_sequential/debug_sb3_architecture.py
```

## 🔧 Hardware Deployment

### HiAER Spike Platform

The project uses the HiAER Spike neuromorphic computing platform for deployment:

```python
from hardware.hs_api.api import CRI_network
from hardware.hs_api.converter import CRI_Converter, Quantize_Network

# Initialize network
network = CRI_network()

# Convert and quantize model
converter = CRI_Converter()
quantizer = Quantize_Network()

# Deploy to hardware
converter.convert_model(model)
network.deploy_model(converted_model)
```

### Key Classes

- **`CRI_network`**: Main interface for hardware network simulation
- **`CRI_Converter`**: Converts SNN models to hardware format
- **`Quantize_Network`**: Quantizes weights and activations
- **`LIF_neuron`**: Leaky Integrate-and-Fire neuron model
- **`ANN_neuron`**: Artificial neural network neuron model

## 📊 Model Performance

### Training Results

The advanced DQN implementation achieves significant improvements:
- Breaks through -18 reward plateau using enhanced reward shaping
- Achieves stable learning with adaptive exploration strategies
- Reaches positive rewards through improved game understanding

### Model Files

- **`models/snn/snn_pong_q_net.pth`**: Trained SNN checkpoint
- **`models/converted/fused_snn_pong.pt`**: Hardware-optimized fused model
- **`models/pretrained/double_dqn_pong_best.pth`**: Best performing DQN model

## 🧪 Testing and Validation

```bash
# Test conversion accuracy
python conversion/validation/validate_sequential_dqn.py

# Test hardware integration
python hardware/testing/test_integration.py

# Evaluate model performance
python training/dqn/evaluate_pong_simple.py
```

## 🛠️ Development

### Code Quality

```bash
# Run linting
poetry run pylint <file>

# Format code (88 character line length)
poetry run black <file>
```

### Development Dependencies

- **pylint**: Code linting
- **black**: Code formatting (88 character line length)

## 📚 Documentation

### Notebooks

- **`HiAER_Spike.ipynb`**: Interactive conversion pipeline
- **`spikingjelly.ipynb`**: SNN experiments and analysis
- **`finetuning.ipynb`**: Model fine-tuning experiments
- **`network_pruning.ipynb`**: Network pruning experiments

### Additional Documentation

- **`CLAUDE.md`**: Development guidance for Claude Code
- **`IMPROVEMENTS_SUMMARY.md`**: Advanced DQN improvements documentation
- **`README_SB3_TO_SEQUENTIAL.md`**: SB3 conversion guide

## 🎯 Use Cases

### Research Applications
- Neuromorphic computing research
- Spiking neural network development
- Reinforcement learning optimization
- Hardware-software co-design

### Industry Applications
- Low-power AI deployment
- Real-time game AI
- Embedded systems
- Edge computing

## 🤝 Contributing

1. Follow the existing code style (88 character line length)
2. Run linting and formatting before commits
3. Add tests for new functionality
4. Update documentation as needed

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 👥 Authors

- Gwenevere Frank <jfrank@ucsd.edu>

## 🔗 Dependencies

### Core Dependencies
- PyTorch 2.0+
- Stable-Baselines3
- Gymnasium (with Atari support)
- NumPy, OpenCV, PyYAML

### Neuromorphic Dependencies
- SpikingJelly
- SNNTorch
- Custom HiAER Spike API

### External Dependencies
- `connectome_utils`: Network connectivity utilities
- `fxpmath`: Fixed-point mathematics
- `hs_bridge`: Hardware bridge interface

## 📈 Roadmap

- [ ] Enhanced SNN architectures
- [ ] Additional Atari game support
- [ ] Real-time hardware deployment
- [ ] Performance optimization
- [ ] Extended validation suite

---

For detailed usage instructions, see the individual script documentation and Jupyter notebooks.