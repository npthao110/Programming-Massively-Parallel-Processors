// Input:  input.ppm (P6 RGB)
// Output: output.pgm (P5 grayscale / black-white)
//
// You are GIVEN: PPM/PGM I/O + grayscale kerne
// You must COMPLETE: threshold kernel + its launch + copy back

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
  if (fscanf(fp, "%2s", magic) != 1) { fclose(fp); return false; }
  if (strcmp(magic, "P6") != 0) {
    fprintf(stderr, "ERROR: %s is not P6 PPM (got %s)\n", path, magic);
    fclose(fp);
    return false;
  }

  skip_ws_and_comments(fp);
  int w = 0, h = 0, maxv = 0;
  if (fscanf(fp, "%d", &w) != 1) { fclose(fp); return false; }
  skip_ws_and_comments(fp);
  if (fscanf(fp, "%d", &h) != 1) { fclose(fp); return false; }
  skip_ws_and_comments(fp);
  if (fscanf(fp, "%d", &maxv) != 1) { fclose(fp); return false; }

  if (w <= 0 || h <= 0 || maxv != 255) {
    fprintf(stderr, "ERROR: unsupported PPM header w=%d h=%d maxv=%d (need maxv=255)\n", w, h, maxv);
    fclose(fp);
    return false;
  }

  fgetc(fp); // consume one whitespace char after maxv

  size_t nbytes = (size_t)w * (size_t)h * 3;
  uint8_t* rgb = (uint8_t*)malloc(nbytes);
  if (!rgb) { fclose(fp); return false; }

  size_t got = fread(rgb, 1, nbytes, fp);
  fclose(fp);

  if (got != nbytes) {
    fprintf(stderr, "ERROR: short read. expected %zu bytes, got %zu\n", nbytes, got);
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
  size_t nbytes = (size_t)w * (size_t)h;
  size_t wrote = fwrite(gray, 1, nbytes, fp);
  fclose(fp);
  return wrote == nbytes;
}

// ------------------------------------------------------------
// GIVEN grayscale kernel (same formula idea as class: rgb_to_grayscale_02052026)
// gray = 3/10 R + 6/10 G + 1/10 B
// Input is interleaved RGB (PPM): rgb[3*i + {0,1,2}]
// ------------------------------------------------------------
__global__ void rgb_to_gray_interleaved_kernel(const uint8_t* rgb, uint8_t* gr,
                                               unsigned int width, unsigned int height) {
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < height && col < width) {
    unsigned int i = row * width + col;
    unsigned int base = 3 * i;
    uint8_t r = rgb[base + 0];
    uint8_t g = rgb[base + 1];
    uint8_t b = rgb[base + 2];
    gr[i] = (uint8_t)((3 * r + 6 * g + 1 * b) / 10);
  }
}

// ------------------------------------------------------------
// TODO (STUDENT): threshold kernel
// Input:  grayscale [0..255]
// Output: black/white, using T=128
// Rule: if gray >= T => 255 else 0
// Must use 2D indexing + boundary checks.
// ------------------------------------------------------------
__global__ void threshold_kernel(const uint8_t* gray, uint8_t* out,
                                 unsigned int width, unsigned int height, uint8_t T) {
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < height && col < width) {
    unsigned int idx = row * width + col;
    out[idx] = (gray[idx] >= T) ? 255 : 0;
  }
}

static uint64_t checksum_u8(const uint8_t* data, size_t n) {
  uint64_t s = 0;
  for (size_t i = 0; i < n; i++) s += data[i];
  return s;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s input.ppm [output.pgm]\n", argv[0]);
    return 1;
  }
  const char* in_path  = argv[1];
  const char* out_path = (argc >= 3) ? argv[2] : "output.pgm";

  // Read PPM on CPU
  uint8_t* h_rgb = nullptr;
  int width = 0, height = 0;
  if (!read_ppm_p6(in_path, &h_rgb, &width, &height)) {
    fprintf(stderr, "ERROR: failed to read %s\n", in_path);
    return 1;
  }
  printf("Loaded %s: %dx%d (P6)\n", in_path, width, height);

  size_t n_pixels = (size_t)width * (size_t)height;
  size_t rgb_bytes  = n_pixels * 3;
  size_t gray_bytes = n_pixels;

  // Host output buffer (final image)
  uint8_t* h_out = (uint8_t*)malloc(gray_bytes);
  if (!h_out) { fprintf(stderr, "malloc failed\n"); free(h_rgb); return 1; }

  // Device buffers
  uint8_t *d_rgb=nullptr, *d_gray=nullptr, *d_out=nullptr;
  CHECK_CUDA(cudaMalloc((void**)&d_rgb,  rgb_bytes));
  CHECK_CUDA(cudaMalloc((void**)&d_gray, gray_bytes));
  CHECK_CUDA(cudaMalloc((void**)&d_out,  gray_bytes));

  CHECK_CUDA(cudaMemcpy(d_rgb, h_rgb, rgb_bytes, cudaMemcpyHostToDevice));

  // Standard 2D grid/block
  dim3 block(16, 16, 1);
  dim3 grid((width  + block.x - 1) / block.x,
            (height + block.y - 1) / block.y,
            1);

  printf("Block=(%d,%d) Grid=(%d,%d)\n", block.x, block.y, grid.x, grid.y);

  // 1) GPU grayscale (given)
  rgb_to_gray_interleaved_kernel<<<grid, block>>>(d_rgb, d_gray, (unsigned)width, (unsigned)height);
  CHECK_CUDA(cudaGetLastError());

  // 2) GPU threshold (student)
  const uint8_t T = 128;

  // TODO: launch your threshold kernel here:
  threshold_kernel<<<grid, block>>>(d_gray, d_out, (unsigned)width, (unsigned)height, T);
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaDeviceSynchronize());

  // TODO: copy d_out back to h_out
  CHECK_CUDA(cudaMemcpy(h_out, d_out, gray_bytes, cudaMemcpyDeviceToHost));

  // Write output PGM
  if (!write_pgm_p5(out_path, h_out, width, height)) {
    fprintf(stderr, "ERROR: failed to write %s\n", out_path);
    return 1;
  }

  // Stats for checking
  uint64_t cs = checksum_u8(h_out, n_pixels);
  size_t white = 0;
  for (size_t i = 0; i < n_pixels; i++) if (h_out[i] == 255) white++;

  printf("Wrote %s (P5)\n", out_path);
  printf("Checksum(sum bytes) = %llu\n", (unsigned long long)cs);
  printf("WHITE PIXELS = %zu out of %zu\n", white, n_pixels);

  // cleanup
  free(h_rgb);
  free(h_out);
  CHECK_CUDA(cudaFree(d_rgb));
  CHECK_CUDA(cudaFree(d_gray));
  CHECK_CUDA(cudaFree(d_out));

  return 0;
}
