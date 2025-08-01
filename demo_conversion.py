"""
Demo script showing the SB3-to-Sequential conversion in action
"""

import torch
from sb3_to_sequential_converter import create_ann_from_snn

def main():
    print("=" * 60)
    print("SB3 DQN TO SEQUENTIAL NETWORK CONVERSION DEMO")
    print("=" * 60)
    
    try:
        # Convert the existing SNN model to Sequential
        print("1. Loading and converting SNN model...")
        sequential_net = create_ann_from_snn('snn_pong_q_net.pth')
        print("   Conversion successful!")
        
        # Show the architecture
        print("\n2. Sequential Network Architecture:")
        print(sequential_net)
        
        # Test inference with different inputs
        print("\n3. Testing inference on different inputs:")
        
        test_cases = [
            ("Random state", torch.randn(1, 4, 84, 84)),
            ("Zero state", torch.zeros(1, 4, 84, 84)),
            ("Batch of 4", torch.randn(4, 4, 84, 84)),
        ]
        
        sequential_net.eval()
        
        for name, test_input in test_cases:
            with torch.no_grad():
                q_values = sequential_net(test_input)
                predicted_action = q_values.argmax(dim=1)
                
            print(f"   {name}:")
            print(f"     Input shape: {test_input.shape}")
            print(f"     Q-values: {q_values.numpy().round(3)}")
            print(f"     Predicted action(s): {predicted_action.numpy()}")
        
        # Show model size
        print("\n4. Model Information:")
        total_params = sum(p.numel() for p in sequential_net.parameters())
        trainable_params = sum(p.numel() for p in sequential_net.parameters() if p.requires_grad)
        
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        # Test saving
        print("\n5. Saving converted model...")
        torch.save({
            'model_state_dict': sequential_net.state_dict(),
            'architecture': 'SequentialDQNNetwork',
            'conversion_source': 'SNN_to_Sequential_Demo'
        }, 'sequential_pong_demo.pt')
        print("   Model saved as 'sequential_pong_demo.pt'")
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("The Sequential network is ready for:")
        print("• Direct PyTorch training and inference")
        print("• SNN conversion using SpikingJelly")
        print("• Hardware deployment pipeline")
        print("• Integration with existing codebase")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()