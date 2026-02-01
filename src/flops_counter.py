"""
FLOPs Counter for Model Complexity Analysis
"""
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count


def count_flops(model, input_shape=(1, 3, 32, 32), device='cuda'):
    """
    Count FLOPs for the given model
    
    Args:
        model (nn.Module): PyTorch model
        input_shape (tuple): Input tensor shape (batch_size, channels, height, width)
        device (str): Device to run analysis on
    
    Returns:
        dict: Dictionary containing FLOPs and parameter information
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)
    
    # Count FLOPs using fvcore
    flops = FlopCountAnalysis(model, dummy_input)
    total_flops = flops.total()
    
    # Count parameters
    total_params = parameter_count(model)['']
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Format numbers
    def format_number(num):
        """Format large numbers with appropriate suffix"""
        if num >= 1e9:
            return f"{num/1e9:.2f}G"
        elif num >= 1e6:
            return f"{num/1e6:.2f}M"
        elif num >= 1e3:
            return f"{num/1e3:.2f}K"
        else:
            return f"{num}"
    
    results = {
        'total_flops': total_flops,
        'total_flops_formatted': format_number(total_flops),
        'total_params': total_params,
        'total_params_formatted': format_number(total_params),
        'trainable_params': trainable_params,
        'trainable_params_formatted': format_number(trainable_params)
    }
    
    return results


def print_flops_analysis(model, input_shape=(1, 3, 32, 32), device='cuda'):
    """
    Print detailed FLOPs analysis
    
    Args:
        model (nn.Module): PyTorch model
        input_shape (tuple): Input tensor shape
        device (str): Device to run analysis on
    """
    results = count_flops(model, input_shape, device)
    
    print("=" * 60)
    print("MODEL COMPLEXITY ANALYSIS")
    print("=" * 60)
    print(f"Input Shape: {input_shape}")
    print("-" * 60)
    print(f"Total FLOPs: {results['total_flops_formatted']} ({results['total_flops']:,})")
    print(f"Total Parameters: {results['total_params_formatted']} ({results['total_params']:,})")
    print(f"Trainable Parameters: {results['trainable_params_formatted']} ({results['trainable_params']:,})")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    from model import get_model
    
    print("Testing FLOPs Counter...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Get model
    model = get_model(device=device)
    
    # Count FLOPs
    results = print_flops_analysis(model, input_shape=(1, 3, 32, 32), device=device)
    
    print("\nFLOPs counter test successful!")
