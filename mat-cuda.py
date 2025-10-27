import time
import numpy as np
import cupy as cp


# Configuración del tamaño de la matriz
N = 10000  

print(f"Multiplicando matrices de {N}x{N}...\n")

# Calculo en cPU
# Generar dos matrices con valores aleatorios en CPU
A_cpu = np.random.rand(N, N).astype(np.float32)
B_cpu = np.random.rand(N, N).astype(np.float32)

# Medir el tiempo antes y después de la multiplicación en CPU
start_cpu = time.time()
C_cpu = A_cpu @ B_cpu  # Operación matricial usando BLAS en CPU
end_cpu = time.time()
cpu_time = end_cpu - start_cpu

print(f"CPU terminada en: {cpu_time:.3f} segundos")


# Calculo en GPU
# Las matrices generadas en NumPy (host) se copian a la memoria de la GPU (device)
A_gpu = cp.asarray(A_cpu)
B_gpu = cp.asarray(B_cpu)

# Asegurar que la GPU ya terminó las transferencias antes de medir el tiempo
cp.cuda.Stream.null.synchronize()

# Medimr el tiempo de la multiplicación en GPU
start_gpu = time.time()
C_gpu = A_gpu @ B_gpu  # Multiplicación matricial ejecutada en CUDA
cp.cuda.Stream.null.synchronize()  # Esperar a que la GPU termine el cálculo
end_gpu = time.time()

gpu_time = end_gpu - start_gpu
print(f"GPU terminada en: {gpu_time:.3f} segundos")

# Calcular cuánto más rápido es la GPU comparada con la CPU
speedup = cpu_time / gpu_time
print(f"\nAceleración GPU vs CPU: {speedup:.2f}x más rápido")

print("\nPrograma terminado")
