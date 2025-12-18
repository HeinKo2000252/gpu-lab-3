
---

## ** GPU.cu** (CUDA program)

```cuda
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define N 10000000  // 10 million elements

/**
 * GPU Kernel: Vector addition with transformation
 * Each element: data[i] = data[i] * multiplier + sin(i * 0.001f)
 */
__global__ void compute_kernel(float* data, int n, int multiplier) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        data[idx] = data[idx] * multiplier + sinf(idx * 0.001f);
    }
}

/**
 * Main function for GPU computation
 * Usage: ./GPU [multiplier]
 * Default multiplier = 1
 */
int main(int argc, char** argv) {
    int multiplier = 1;
    if (argc > 1) {
        multiplier = atoi(argv[1]);
    }
    
    printf("=== GPU Parallel Computation ===\n");
    printf("Hostname: ");
    system("hostname");
    printf("Multiplier: %d\n", multiplier);
    printf("Elements: %d\n", N);
    
    // Host memory allocation
    float *h_data = (float*)malloc(N * sizeof(float));
    if (h_data == NULL) {
        printf("Error: Host memory allocation failed\n");
        return 1;
    }
    
    // Initialize with ones
    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;
    }
    
    // Device memory allocation
    float *d_data;
    cudaError_t err = cudaMalloc(&d_data, N * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        free(h_data);
        return 1;
    }
    
    // Copy data to device
    err = cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA Memcpy Error: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        free(h_data);
        return 1;
    }
    
    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Kernel configuration
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Launch kernel with timing
    cudaEventRecord(start);
    compute_kernel<<<gridSize, blockSize>>>(d_data, N, multiplier);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Get execution time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result back to host
    err = cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA Memcpy Error: %s\n", cudaGetErrorString(err));
    }
    
    // Calculate checksum (first 1000 elements)
    float checksum = 0.0f;
    for (int i = 0; i < 1000 && i < N; i++) {
        checksum += h_data[i];
    }
    
    // Print results
    printf("GPU Time: %.3f ms\n", milliseconds);
    printf("Data Size: %.1f MB\n", N * sizeof(float) / (1024.0f * 1024.0f));
    printf("Throughput: %.1f GB/s\n", 
           (N * sizeof(float) * 2) / (milliseconds * 1e6f));
    printf("Checksum: %.3f\n", checksum);
    printf("====================\n");
    
    // Cleanup
    cudaFree(d_data);
    free(h_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
