
#include <stdio.h>

__global__ void vecadd_kernel (float *x, float *y, float *z, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    z[i] = x[i] + y[i];
    if (i < 50)
      printf("z[%4d] = %f\n",i, z[i]);
  }
}

void vec_add (float *x, float *y, float *z, int n) {
  cudaError_t err;
  float *x_d, *y_d, *z_d;
  const unsigned int numThreadsPerBlock = 512;
  const unsigned int numBlocks = (n + numThreadsPerBlock - 1) / numThreadsPerBlock;

  err = cudaMalloc ((void **) &x_d, n * sizeof (float));
  if (err != cudaSuccess) {
    printf ("cudaMalloc error for x_d: %s\n", cudaGetErrorString (err));
    exit(1);
  }
  err = cudaMalloc ((void **) &y_d, n * sizeof (float));
  if (err != cudaSuccess) {
    printf ("cudaMalloc error for y_d: %s\n", cudaGetErrorString (err));
    exit(1);
  }
  err = cudaMalloc ((void **) &z_d, n * sizeof (float));
  if (err != cudaSuccess) {
    printf ("cudaMalloc error for z_d: %s\n", cudaGetErrorString (err));
    exit(1);
  }

  err = cudaMemcpy (x_d, x, n * sizeof (float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    printf ("cudaMemcpy error for x_d: %s\n", cudaGetErrorString (err));
    exit(1);
  }
  err = cudaMemcpy (y_d, y, n * sizeof (float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    printf ("cudaMemcpy error for y_d: %s\n", cudaGetErrorString (err));
    exit(1);
  }

  vecadd_kernel <<<numBlocks, numThreadsPerBlock>>> (x_d, y_d, z_d, n);

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf ("cudaDeviceSynchronize error: %s\n", cudaGetErrorString (err));
    exit(1);
  }

  err = cudaMemcpy (z, z_d, n * sizeof (float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    printf ("cudaMemcpy error for z: %s\n", cudaGetErrorString (err));
    exit(1);
  }
  err = cudaFree (x_d);
  if (err != cudaSuccess) {
    printf ("cudaFree error for x_d: %s\n", cudaGetErrorString (err));
    exit(1);
  }
  err = cudaFree (y_d);
  if (err != cudaSuccess) {
    printf ("cudaFree error for y_d: %s\n", cudaGetErrorString (err));
    exit(1);
  }

  err = cudaFree (z_d);
  if (err != cudaSuccess) {
    printf ("cudaFree error for z_d: %s\n", cudaGetErrorString (err));
    exit(1);
  }

  return;


}

int main (void) {
  float x[1000], y[1000], z[1000];
  int i;
  for (i = 0; i < 1000; i++) {
    x[i] = 1.0;
    y[i] = 2.0;
  }

  /* call to a function to set up the vector addition on the gpu */
  vec_add (x, y, z, 1000);

  /* print the results */

  putchar('\n');
  for (i = 0; i < 10; i++) {
    printf ("z[%2d] = %f\n",i,z[i]);
  }
}
