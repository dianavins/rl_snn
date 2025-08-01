# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a reinforcement learning project that trains and deploys Spiking Neural Networks (SNNs) for Atari Pong using the HiAER Spike neuromorphic computing platform. The project converts traditional DQN models to SNNs and quantizes them for hardware deployment.

## Development Environment

This project uses Poetry for dependency management:
- Python 3.10+ (< 3.11)
- Main dependencies: spikingjelly, snntorch, stable-baselines3, torch
- Development tools: pylint, black (with 88 char line length)

### Common Commands

```bash
# Install dependencies
poetry install

# Install with specific groups
poetry install --with apps     # ML/visualization dependencies
poetry install --with fpga     # FPGA bridge dependencies  
poetry install --with dev      # Development tools

# Run linting
poetry run pylint <file>

# Format code (88 character line length)
poetry run black <file>

# Run main conversion pipeline
python HiAER_Spike.py

# Run improved finetuning
python improved_finetuning.py
```

## Architecture

### Core Components

1. **hs_api/** - Hardware abstraction layer for HiAER Spike platform
   - `api.py` - CRI_network class for network initialization and simulation
   - `converter.py` - Tools for converting PyTorch models to SNN format and quantization
   - `neuron_models.py` - Neuron model definitions (LIF, ANN)
   - `_simple_sim.py` - Simple simulation backend
   - `_intermediate_format.py` - Intermediate representation handling

2. **HiAER_Spike.py** - Main conversion pipeline script that:
   - Loads pre-trained DQN models 
   - Quantizes networks using custom Quantize_Network class
   - Converts to CRI format using CRI_Converter
   - Deploys to HiAER Spike hardware simulation

3. **Training Scripts**:
   - `improved_finetuning.py` - Enhanced DQN finetuning with reward shaping
   - `pong_dqn_training.py` - Original DQN training for Atari Pong
   - `ann_from_scratch.py` - ANN training from scratch
   - `network_pruning.py` - Network pruning utilities

4. **Jupyter Notebooks**:
   - `HiAER_Spike.ipynb` - Interactive version of main pipeline
   - `spikingjelly.ipynb` - SNN experiments and analysis
   - `finetuning.ipynb` - Model fine-tuning experiments
   - `ann_pruning.ipynb` - Network pruning experiments

### Key Classes

- `CRI_network` (hs_api/api.py) - Main interface for hardware network simulation
- `CRI_Converter` (hs_api/converter.py) - Converts SNN models to hardware format
- `Quantize_Network` (hs_api/converter.py) - Quantizes network weights and activations
- `LIF_neuron` (hs_api/neuron_models.py) - Leaky Integrate-and-Fire neuron model
- `ANN_neuron` (hs_api/neuron_models.py) - Artificial neural network neuron model
- `PongRewardWrapper` (improved_finetuning.py) - Reward shaping wrapper for Pong environment

### Model Files

- `snn_pong_q_net.pth` - Trained SNN checkpoint
- `snn_pong_q_net_full.pt` - Full SNN model with architecture
- `fused_snn_pong.pt` - Hardware-optimized fused model

## Development Notes

- The project requires external dependencies (`connectome_utils`, `fxpmath`, `hs_bridge`) that are path-based local packages located in parent directories
- Models are specifically trained for Atari Pong (4x84x84 input shape) using "PongNoFrameskip-v4" environment
- Two quantization approaches: PyTorch dynamic quantization and custom `Quantize_Network` with configurable alpha scaling (default alpha=4)
- Network conversion supports 1-step inference for real-time deployment on HiAER Spike hardware
- The improved finetuning script includes reward shaping and optimized hyperparameters for better convergence
- Code formatting enforces 88-character line length limit via Black formatter