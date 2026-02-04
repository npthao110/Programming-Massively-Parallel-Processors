
#include <stdio.h>

#define ROW 8
#define COL 7

__global__ void twod_init_kernel (float *x, unsigned int n, unsigned int m) {
  unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
  //printf("twod_init_kernel - blockDim.x = %u  blockIdx.x = %u  threadIdx.x = %u  blockDim.y = %u  blockIdx.y = %u  threadIdx.y = %u  row = %u  col = %u\n",blockDim.x,blockIdx.x,threadIdx.x,blockDim.y,blockIdx.y,threadIdx.y,row,col);
  unsigned int i = row * m + col;
  if (row < n && col < m) {
    printf("twod_init_kernel - blockDim.x = %u  blockIdx.x = %u  threadIdx.x = %u  blockDim.y = %u  blockIdx.y = %u  threadIdx.y = %u  row = %u  col = %u i = %u\n",blockDim.x,blockIdx.x,threadIdx.x,blockDim.y,blockIdx.y,threadIdx.y,row,col,i);
    x[i] = row;
  }
}

void twod_init (float x[][COL], unsigned int n, unsigned int m) {
  cudaError_t err;
  unsigned int size = n * m;
  float block_xdiv = 4.0;
  float block_ydiv = 2.0;
  unsigned int block_xdim = ceil(n/block_xdiv);
  unsigned int block_ydim = ceil(m/block_ydiv);
  float *x_d;
  //use dim3 arguments as x, y, and z in that order
  dim3 dimGrid (block_xdim, block_ydim, 1);
  dim3 dimBlock (block_xdiv, block_ydiv, 1);

  printf ("twod_init - n = %u  m = %u  block_xdim = %u  block_ydim = %u total threads = %u\n",n,m,block_xdim, block_ydim, block_xdim*block_ydim*(int)block_xdiv*(int)block_ydiv);

  //const unsigned int numThreadsPerBlock = 32;
  //const unsigned int numBlocks = (n + numThreadsPerBlock - 1) / numThreadsPerBlock;

  err = cudaMalloc ((void **) &x_d, size * sizeof (float));
  if (err != cudaSuccess) {
    printf ("cudaMalloc error for x_d: %s\n", cudaGetErrorString (err));
    exit(1);
  }

  twod_init_kernel <<<dimGrid, dimBlock>>> (x_d, n, m);

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf ("cudaDeviceSynchronize error: %s\n", cudaGetErrorString (err));
    exit(1);
  }

  err = cudaMemcpy (*x, x_d, size * sizeof (float), cudaMemcpyDeviceToHost);
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
    float twod[ROW][COL];
    unsigned int i, j;

    twod_init(twod,ROW,COL);


    //print twod using traditional indexing
    for (i = 0; i < ROW; i++) {
        for (j = 0; j < COL; j++) {
            printf("[%1d,%1d]:%3.1f ",i,j,twod[i][j]);
        }
        printf("\n");
    }
    printf("\n\n\n");

    //print twod using pointers
    for (i = 0; i < ROW; i++) {
        for (j = 0; j < COL; j++) {
            printf("[%1d,%1d]:%3.1f ",i,j,*(*(twod + i)+j));
        }
        printf("\n");
    }
    printf("\n\n\n");

    //print twod using pointers and equation
    for (i = 0; i < ROW; i++) {
        for (j = 0; j < COL; j++) {
            printf("[%1d,%1d]:%3.1f ",i,j,*(*twod + i*COL + j));
        }
        printf("\n");
    }
    printf("\n\n\n");

    return(0);
}
