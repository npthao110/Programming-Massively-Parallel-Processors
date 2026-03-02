// Assignment 3 – Tiled Box Blur (3x3) using shared memory tiling + halo
// Input : input.ppm (P6 RGB, maxval 255)
// Output: output.pgm (P5 grayscale, blurred)

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(1); \
  } \
} while (0)

// ------------------------------------------------------------
// PPM/PGM I/O (no libraries)
// Supports P6 PPM (binary RGB, maxval 255) with optional comments.
// ------------------------------------------------------------
static void skip_ws_and_comments(FILE* f) {
  int c;
  while ((c = fgetc(f)) != EOF) {
    if (c == '#') { // comment line
      while ((c = fgetc(f)) != EOF && c != '\n') {}
      continue;
    }
    if (c == ' ' || c == '\n' || c == '\r' || c == '\t') continue;
    ungetc(c, f);
    break;
  }
}

static bool read_ppm_p6(const char* path, uint8_t** rgb_out, int* w_out, int* h_out) {
  FILE* fp = fopen(path, "rb");
  if (!fp) { perror("fopen"); return false; }

  char magic[3] = {0};
  if (fread(magic, 1, 2, fp) != 2) { fclose(fp); return false; }
  if (magic[0] != 'P' || magic[1] != '6') {
    fprintf(stderr, "Not a P6 PPM: %s\n", path);
    fclose(fp);
    return false;
  }

  skip_ws_and_comments(fp);
  int w = 0, h = 0, maxval = 0;

  if (fscanf(fp, "%d", &w) != 1) { fclose(fp); return false; }
  skip_ws_and_comments(fp);
  if (fscanf(fp, "%d", &h) != 1) { fclose(fp); return false; }
  skip_ws_and_comments(fp);
  if (fscanf(fp, "%d", &maxval) != 1) { fclose(fp); return false; }
  if (maxval != 255) {
    fprintf(stderr, "Unsupported maxval %d (expected 255)\n", maxval);
    fclose(fp);
    return false;
  }

  // consume single whitespace after header
  int c = fgetc(fp);
  if (c == '\r') { int c2 = fgetc(fp); if (c2 != '\n') ungetc(c2, fp); }
  else if (c != '\n' && c != ' ' && c != '\t') ungetc(c, fp);

  size_t bytes = (size_t)w * (size_t)h * 3;
  uint8_t* rgb = (uint8_t*)malloc(bytes);
  if (!rgb) { fprintf(stderr, "malloc failed\n"); fclose(fp); return false; }

  size_t got = fread(rgb, 1, bytes, fp);
  fclose(fp);
  if (got != bytes) {
    fprintf(stderr, "PPM read truncated: expected %zu, got %zu\n", bytes, got);
    free(rgb);
    return false;
  }

  *rgb_out = rgb;
  *w_out = w;
  *h_out = h;
  return true;
}

static bool write_pgm_p5(const char* path, const uint8_t* gray, int w, int h) {
  FILE* fp = fopen(path, "wb");
  if (!fp) { perror("fopen"); return false; }
  fprintf(fp, "P5\n%d %d\n255\n", w, h);
  size_t bytes = (size_t)w * (size_t)h;
  size_t wrote = fwrite(gray, 1, bytes, fp);
  fclose(fp);
  if (wrote != bytes) {
    fprintf(stderr, "PGM write truncated: expected %zu, wrote %zu\n", bytes, wrote);
    return false;
  }
  return true;
}

// ------------------------------------------------------------
// GPU kernels
// ------------------------------------------------------------

// Interleaved RGB (rgb[3*i + 0..2]) -> grayscale
__global__ void rgb_to_gray_interleaved_kernel(const uint8_t* rgb, uint8_t* gray,
                                               unsigned int width, unsigned int height) {
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < height && col < width) {
    unsigned int idx = row * width + col;
    unsigned int base = 3u * idx;
    // Integer approximation (same style as chapter slides): 0.3R + 0.6G + 0.1B
    gray[idx] = (uint8_t)((rgb[base + 0] * 3u + rgb[base + 1] * 6u + rgb[base + 2] * 1u) / 10u);
  }
}

#ifndef BLK_X
#define BLK_X 16
#endif
#ifndef BLK_Y
#define BLK_Y 16
#endif

// 3x3 box blur using shared memory tiling + 1-pixel halo
__global__ void blur3x3_tiled_kernel(const uint8_t* in, uint8_t* out,
                                     unsigned int width, unsigned int height) {
  // Shared tile dimensions include 1-pixel halo on each side
  __shared__ uint8_t tile[(BLK_Y + 2) * (BLK_X + 2)];
  const int sW = BLK_X + 2;

  int tx = (int)threadIdx.x;
  int ty = (int)threadIdx.y;

  int outCol = (int)blockIdx.x * BLK_X + tx;
  int outRow = (int)blockIdx.y * BLK_Y + ty;

  auto load_global = [&](int r, int c) -> uint8_t {
    if (r >= 0 && r < (int)height && c >= 0 && c < (int)width) return in[r * (int)width + c];
    return 0;
  };

  // center
  tile[(ty + 1) * sW + (tx + 1)] = load_global(outRow, outCol);

  // halo: left/right
  if (tx == 0) {
    tile[(ty + 1) * sW + 0] = load_global(outRow, outCol - 1);
  }
  if (tx == BLK_X - 1) {
    tile[(ty + 1) * sW + (BLK_X + 1)] = load_global(outRow, outCol + 1);
  }

  // halo: top/bottom
  if (ty == 0) {
    tile[0 * sW + (tx + 1)] = load_global(outRow - 1, outCol);
  }
  if (ty == BLK_Y - 1) {
    tile[(BLK_Y + 1) * sW + (tx + 1)] = load_global(outRow + 1, outCol);
  }

  // halo corners
  if (tx == 0 && ty == 0) {
    tile[0 * sW + 0] = load_global(outRow - 1, outCol - 1);
  }
  if (tx == BLK_X - 1 && ty == 0) {
    tile[0 * sW + (BLK_X + 1)] = load_global(outRow - 1, outCol + 1);
  }
  if (tx == 0 && ty == BLK_Y - 1) {
    tile[(BLK_Y + 1) * sW + 0] = load_global(outRow + 1, outCol - 1);
  }
  if (tx == BLK_X - 1 && ty == BLK_Y - 1) {
    tile[(BLK_Y + 1) * sW + (BLK_X + 1)] = load_global(outRow + 1, outCol + 1);
  }

  __syncthreads();

  // write output only for valid pixels
  if (outRow < (int)height && outCol < (int)width) {
    unsigned int sum = 0;
    // 3x3 neighborhood in shared memory around (ty+1, tx+1)
    int sRow = ty + 1;
    int sCol = tx + 1;
    sum += tile[(sRow - 1) * sW + (sCol - 1)];
    sum += tile[(sRow - 1) * sW + (sCol    )];
    sum += tile[(sRow - 1) * sW + (sCol + 1)];
    sum += tile[(sRow    ) * sW + (sCol - 1)];
    sum += tile[(sRow    ) * sW + (sCol    )];
    sum += tile[(sRow    ) * sW + (sCol + 1)];
    sum += tile[(sRow + 1) * sW + (sCol - 1)];
    sum += tile[(sRow + 1) * sW + (sCol    )];
    sum += tile[(sRow + 1) * sW + (sCol + 1)];

    out[outRow * (int)width + outCol] = (uint8_t)(sum / 9u);
  }
}

// ------------------------------------------------------------
// main
// ------------------------------------------------------------
int main(int argc, char** argv) {
  const char* in_path  = (argc > 1) ? argv[1] : "input.ppm";
  const char* out_path = (argc > 2) ? argv[2] : "output.pgm";

  uint8_t* h_rgb = nullptr;
  int width = 0, height = 0;

  if (!read_ppm_p6(in_path, &h_rgb, &width, &height)) {
    fprintf(stderr, "Failed to read %s\n", in_path);
    return 1;
  }

  printf("Loaded image: %d x %d\n", width, height);

  size_t rgb_bytes  = (size_t)width * (size_t)height * 3;
  size_t gray_bytes = (size_t)width * (size_t)height;

  // Host output
  uint8_t* h_out = (uint8_t*)malloc(gray_bytes);
  if (!h_out) { fprintf(stderr, "malloc failed\n"); free(h_rgb); return 1; }

  // Device buffers
  uint8_t *d_rgb = nullptr, *d_gray = nullptr, *d_out = nullptr;
  CHECK_CUDA(cudaMalloc((void**)&d_rgb,  rgb_bytes));
  CHECK_CUDA(cudaMalloc((void**)&d_gray, gray_bytes));
  CHECK_CUDA(cudaMalloc((void**)&d_out,  gray_bytes));

  CHECK_CUDA(cudaMemcpy(d_rgb, h_rgb, rgb_bytes, cudaMemcpyHostToDevice));

  dim3 block(BLK_X, BLK_Y);
  dim3 grid((width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

  printf("Block: (%u, %u, %u)\n", block.x, block.y, block.z);
  printf("Grid : (%u, %u, %u)\n", grid.x,  grid.y,  grid.z);

  // 1) RGB -> gray
  rgb_to_gray_interleaved_kernel<<<grid, block>>>(d_rgb, d_gray, (unsigned)width, (unsigned)height);
  CHECK_CUDA(cudaGetLastError());

  // 2) Tiled 3x3 blur on grayscale
  blur3x3_tiled_kernel<<<grid, block>>>(d_gray, d_out, (unsigned)width, (unsigned)height);
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaDeviceSynchronize());

  // Copy back
  CHECK_CUDA(cudaMemcpy(h_out, d_out, gray_bytes, cudaMemcpyDeviceToHost));

  // Write output
  if (!write_pgm_p5(out_path, h_out, width, height)) {
    fprintf(stderr, "Failed to write %s\n", out_path);
  } else {
    printf("Wrote output: %s\n", out_path);
  }

  // Cleanup
  CHECK_CUDA(cudaFree(d_rgb));
  CHECK_CUDA(cudaFree(d_gray));
  CHECK_CUDA(cudaFree(d_out));
  free(h_rgb);
  free(h_out);

  return 0;
}
