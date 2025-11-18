def visualize_hotspots(ptx_code, model, device):
    """
    Input: Raw PTX
    Output: Annotated PTX with hotspot highlighting
    """
    tokenizer = PTXTokenizer()
    tokens = tokenizer.tokenize_ptx(ptx_code)
    token_ids = tokenizer.encode_tokens(tokens)
    token_tensor = torch.tensor([token_ids], device=device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(token_tensor)
    
    hotspots = outputs['hotspots'][0].cpu().numpy()
    
    # Annotate PTX lines with hotspot scores
    lines = ptx_code.split('\n')
    annotated = []
    
    for i, line in enumerate(lines):
        if i < len(hotspots):
            score = hotspots[i]
            color_code = ''
            if score > 0.7:
                color_code = '\033[91m'  # Red: severe hotspot
            elif score > 0.4:
                color_code = '\033[93m'  # Yellow: moderate
            else:
                color_code = '\033[92m'  # Green: cool
            
            annotated.append(f"{color_code}{line} [{score:.2f}]\033[0m")
        else:
            annotated.append(line)
    
    return '\n'.join(annotated)

# Generate heatmap
import matplotlib.pyplot as plt
import numpy as np

def plot_cache_interference_prediction(ptx_code, model, device):
    """Visualize predicted cache impact across code"""
    tokenizer = PTXTokenizer()
    tokens = tokenizer.tokenize_ptx(ptx_code)
    token_ids = tokenizer.encode_tokens(tokens)
    
    # Compute hotspots at different cache pressure levels
    cache_pressures = np.linspace(0, 1, 5)
    hotspot_matrix = []
    
    for pressure in cache_pressures:
        token_tensor = torch.tensor([token_ids], device=device)
        contention = {
            'enemy_intensity': pressure,
            'cache_pressure': pressure,
            'l2_size_mb': 4.0
        }
        
        with torch.no_grad():
            outputs = model(token_tensor, contention)
        hotspot_matrix.append(outputs['hotspots'][0].cpu().numpy())
    
    # Plot heatmap
    plt.figure(figsize=(14, 6))
    plt.imshow(hotspot_matrix, aspect='auto', cmap='RdYlGn_r')
    plt.colorbar(label='Hotspot Severity')
    plt.xlabel('Instruction Index')
    plt.ylabel('Cache Pressure (Enemy Intensity)')
    plt.yticks(range(len(cache_pressures)), [f'{p:.1f}' for p in cache_pressures])
    plt.title('GPU Kernel Hotspot Prediction under Cache Contention')
    plt.tight_layout()
    plt.savefig('hotspots_heatmap.png', dpi=150)
