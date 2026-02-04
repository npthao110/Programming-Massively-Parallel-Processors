
#include <stdio.h>

#define ROW 8
#define COL 7
#define PLANE 6

__global__ void threed_init_kernel (float *x, unsigned int n, unsigned int m, unsigned int o) {
  unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int plane = blockIdx.z * blockDim.z + threadIdx.z;
  unsigned int i = plane * n * m + row * m + col;
  if (row < n && col < m && plane < o) {
    printf("threed_init_kernel - bDim.x = %u  bIdx.x = %u  thrIdx.x = %u  bDim.y = %u  bIdx.y = %u  thrIdx.y = %u  bDim.z = %u  bIdx.z = %u  thrIdx.z = %u  row = %u  col = %u i = %u\n",blockDim.x,blockIdx.x,threadIdx.x,blockDim.y,blockIdx.y,threadIdx.y,blockDim.z,blockIdx.z,threadIdx.z,row,col,i);
    x[i] = row;
  }
}

void threed_init (float x[][ROW][COL], unsigned int n, unsigned int m, unsigned int o) {
  cudaError_t err;
  unsigned int size = n * m * o;
  float block_xdiv = 4.0;
  float block_ydiv = 2.0;
  float block_zdiv = 3.0;
  unsigned int block_xdim = ceil(n/block_xdiv);
  unsigned int block_ydim = ceil(m/block_ydiv);
  unsigned int block_zdim = ceil(o/block_zdiv);
  float *x_d;
  //use dim3 arguments as x, y, and z in that order
  dim3 dimGrid (block_xdim, block_ydim, block_zdim);
  dim3 dimBlock (block_xdiv, block_ydiv, block_zdiv);

  printf ("threed_init - n = %u  m = %u  o = %u  block_xdim = %u  block_ydim = %u block_zdim = %u  total threads = %u\n",n,m,o,block_xdim, block_ydim, block_zdim, block_xdim*block_ydim*block_zdim*(int)block_xdiv*(int)block_ydiv*(int)block_zdiv);

  err = cudaMalloc ((void **) &x_d, size * sizeof (float));
  if (err != cudaSuccess) {
    printf ("cudaMalloc error for x_d: %s\n", cudaGetErrorString (err));
    exit(1);
  }

  threed_init_kernel <<<dimGrid, dimBlock>>> (x_d, n, m, o);

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
    float threed[PLANE][ROW][COL];
    unsigned int i, j, k;

    threed_init(threed,ROW,COL,PLANE);

    // Accessing 3D array elements
    for (k = 0; k < PLANE; k++) {
       for (i = 0; i < ROW; i++) {
           for (j = 0; j < COL; j++) {
                printf("(%1d,%1d,%1d):%3.1f ",i,j,k,threed[k][i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n\n");

    // Accessing 3D array using pointer arithmetic
    for (k = 0; k < PLANE; k++) {
       for (i = 0; i < ROW; i++) {
           for (j = 0; j < COL; j++) {
                 printf("(%1d,%1d,%1d):%3.1f ",i,j,k,*(*(*(threed + k) + i) + j));
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n\n");

    // Accessing 3D array using single pointer arithmetic
    // Address(i,j,k) = base + (i*d2*d3 + j*d3 + k)*sizeof(type)
    for (k = 0; k < PLANE; k++) {
       for (i = 0; i < ROW; i++) {
           for (j = 0; j < COL; j++) {
                 printf("(%1d,%1d,%1d):%3.1f ",i,j,k,*(**threed + k*ROW*COL + i*COL + j));
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n\n");

    return(0);
}
