// CUDA program to initialize a 1D array on the GPU and transfer it back to the host
// This demonstrates basic GPU memory management and kernel execution

#include <stdio.h>

// Define the size of the array to initialize
#define SIZE 8

// CUDA kernel function that runs on the GPU
// This kernel initializes each element of the array with its index value
// Parameters:
//   x: pointer to the array in GPU memory
//   n: size of the array
__global__ void oned_init_kernel (float *x, int n) {
  // Calculate the global thread index using CUDA's thread indexing
  // blockIdx.x: the block index within the grid
  // blockDim.x: the number of threads per block
  // threadIdx.x: the thread index within the block
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Only initialize if the thread index is within the array bounds
  // This prevents out-of-bounds access when the number of threads exceeds the array size
  if (i < n) {
    x[i] = i;  // Set each element to its index value (0, 1, 2, ..., n-1)
  }
}

// Host function that manages GPU memory and kernel execution
// This function allocates GPU memory, launches the kernel, and copies results back
// Parameters:
//   x: pointer to the host array where results will be stored
//   n: size of the array
void oned_init (float *x, int n) {
  cudaError_t err;
  float *x_d;  // Pointer to device (GPU) memory
  
  // Configure the kernel launch parameters
  const unsigned int numThreadsPerBlock = 32;  // Number of threads per block
  // Calculate the number of blocks needed to cover all array elements
  // The formula ensures we have enough blocks even if n is not divisible by numThreadsPerBlock
  const unsigned int numBlocks = (n + numThreadsPerBlock - 1) / numThreadsPerBlock;

  // Allocate memory on the GPU for the array
  err = cudaMalloc ((void **) &x_d, n * sizeof (float));
  if (err != cudaSuccess) {
    printf ("cudaMalloc error for x_d: %s\n", cudaGetErrorString (err));
    exit(1);
  }

  // Launch the kernel on the GPU
  // Syntax: kernel_name <<<numBlocks, numThreadsPerBlock>>> (arguments)
  // This creates a grid of blocks, each containing numThreadsPerBlock threads
  oned_init_kernel <<<numBlocks, numThreadsPerBlock>>> (x_d, n);

  // Wait for all GPU threads to complete before proceeding
  // This ensures the kernel has finished executing before we copy data back
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf ("cudaDeviceSynchronize error: %s\n", cudaGetErrorString (err));
    exit(1);
  }

  // Copy the initialized array from GPU memory back to host memory
  // cudaMemcpyDeviceToHost indicates copying from device to host
  err = cudaMemcpy (x, x_d, n * sizeof (float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    printf ("cudaMemcpy error for x: %s\n", cudaGetErrorString (err));
    exit(1);
  }
  
  // Free the GPU memory that was allocated earlier
  err = cudaFree (x_d);
  if (err != cudaSuccess) {
    printf ("cudaFree error for x_d: %s\n", cudaGetErrorString (err));
    exit(1);
  }
  return;
}

// Main function: entry point of the program
int main (void) {
    float oned[SIZE];  // Host array to store the initialized values
    int i;

    // Initialize the array on the GPU and copy results back to host
    oned_init(oned,SIZE);


    // Print the array using traditional array indexing
    for (i = 0; i < SIZE; i++) {
        printf("[%1d]:%3.1f ",i,oned[i]);
    }
    printf("\n\n\n");

    // Print the array using pointer arithmetic (demonstrates an alternative access method)
    for (i = 0; i < SIZE; i++) {
        printf("[%1d]:%3.1f ",i,*(oned + i));
    }
    printf("\n\n\n");
    return(0);
}
