import numpy as np

index_to_replace = 8
mean, std = 0, 1  # Standard normal distribution parameters
num_samples = 10
samples = np.linspace(mean - 3*std, mean + 3*std, num_samples)

for i,sample in enumerate(samples):
    print(f"Sample {i+1}: {sample}")