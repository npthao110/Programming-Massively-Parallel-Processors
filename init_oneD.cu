
#include <stdio.h>

#define SIZE 8

__global__ void oned_init_kernel (float *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] = i;
  }
}

void oned_init (float *x, int n) {
  cudaError_t err;
  float *x_d;
  const unsigned int numThreadsPerBlock = 32;
  const unsigned int numBlocks = (n + numThreadsPerBlock - 1) / numThreadsPerBlock;

  err = cudaMalloc ((void **) &x_d, n * sizeof (float));
  if (err != cudaSuccess) {
    printf ("cudaMalloc error for x_d: %s\n", cudaGetErrorString (err));
    exit(1);
  }

  oned_init_kernel <<<numBlocks, numThreadsPerBlock>>> (x_d, n);

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf ("cudaDeviceSynchronize error: %s\n", cudaGetErrorString (err));
    exit(1);
  }

  err = cudaMemcpy (x, x_d, n * sizeof (float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    printf ("cudaMemcpy error for x: %s\n", cudaGetErrorString (err));
    exit(1);
  }
  err = cudaFree (x_d);
  if (err != cudaSuccess) {
    printf ("cudaFree error for x_d: %s\n", cudaGetErrorString (err));
    exit(1);
  }
  return;
}

int main (void) {
    float oned[SIZE];
    int i;

    oned_init(oned,SIZE);


    //print oned using traditional indexing
    for (i = 0; i < SIZE; i++) {
        printf("[%1d]:%3.1f ",i,oned[i]);
    }
    printf("\n\n\n");

    //print oned using pointers
    for (i = 0; i < SIZE; i++) {
        printf("[%1d]:%3.1f ",i,*(oned + i));
    }
    printf("\n\n\n");
    return(0);
}
