import torch
from torch.utils.cpp_extension import load
import subprocess
import os
from pathlib import Path

# 1. COMPILER avec PTX
print("=== COMPILATION ===")
matmul = load(
    name='matmul',
    sources=['matmul.cu'],
    extra_cuda_cflags=[
        '-O3','-gencode=arch=compute_75,code=sm_75', '--ptxas-options=-v', '-lineinfo' ],
    verbose=True
)

# 2. TROUVER le fichier compilé (.pyd sur Windows, .so sur Linux)
module_path = matmul.__file__
print(f"\nModule compilé: {module_path}")

# 3. EXTRAIRE PTX
print("\n=== EXTRACTION PTX ===")
ptx_file = "matmul_kernel.ptx"
try:
    result = subprocess.run(
        ['cuobjdump', '-ptx', module_path], stdout=open(ptx_file, 'w'), stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        print(f"PTX sauvegardé: {ptx_file}")
        with open(ptx_file, 'r') as f:
            lines = f.readlines()
            print(f"  Taille: {len(lines)} lignes")
            print("  Aperçu:")
            for line in lines[:10]:
                print(f"    {line.rstrip()}")
    else:
        print(f"Erreur cuobjdump: {result.stderr}")
except FileNotFoundError:
    print("cuobjdump non trouvé. Installe CUDA Toolkit.")

# 4. EXTRAIRE SASS
print("\n=== EXTRACTION SASS ===")
sass_file = "matmul_kernel.sass"
try:
    subprocess.run(
        ['cuobjdump', '-sass', module_path],
        stdout=open(sass_file, 'w'),
        stderr=subprocess.PIPE
    )
    print(f"SASS sauvegardé: {sass_file}")
except:
    print("SASS extraction échouée")

# 5. TEST FONCTIONNEL
print("\n=== TEST KERNEL ===")
A = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
B = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)

# Warm-up
for _ in range(3):
    C = matmul.forward(A, B)
torch.cuda.synchronize()

# Test avec timing
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
C = matmul.forward(A, B)
end.record()
torch.cuda.synchronize()

time_ms = start.elapsed_time(end)
print(f"  Temps GPU: {time_ms:.3f} ms")
print(f"  Shape: {C.shape}")

#ncu --metrics gpu__time_duration.sum python setup.py
