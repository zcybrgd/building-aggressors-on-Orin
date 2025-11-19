def extract_ptx_features(ptx_code: str):
    """Extract high-level features from PTX"""
    features = {
        'num_global_loads': len(re.findall(r'ld\.global', ptx_code)),
        'num_global_stores': len(re.findall(r'st\.global', ptx_code)),
        'num_shared_accesses': len(re.findall(r'\.(shared)', ptx_code)),
        'num_compute_ops': len(re.findall(r'(mad|add|mul|fma)', ptx_code)),
        'num_branches': len(re.findall(r'@.*bra', ptx_code)),
        'num_predicates': len(re.findall(r'@%p\d+', ptx_code)),
        'memory_footprint_estimate': estimate_working_set(ptx_code),
        'compute_intensity': estimate_compute_intensity(ptx_code),
    }
    return features

def estimate_working_set(ptx_code: str):
    """Heuristic: infer working set from address patterns"""
    # Extract array accesses: [%rd1 + offset]
    accesses = re.findall(r'\[%rd\d+(?:\s*\+\s*(\d+))?\]', ptx_code)
    if not accesses:
        return 0
    
    max_offset = max(int(a) if a else 0 for a in accesses)
    return max_offset + 4096  # Conservative estimate
