#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <png.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>

#define PATCH_SIZE 8
#define MAX_ITER 5
#define DICT_SIZE 512
#define SPARSITY 10
#define WIDTH 800
#define HEIGHT 800
#define INPUT_FILE "input.raw"
#define NOISY_FILE "noisy_image.png"
#define DENOISED_FILE "denoised_image.png"

// Maximum patch dimension and sparsity for static array sizing inside kernels
#define PATCH_DIM (PATCH_SIZE*PATCH_SIZE)

#define CUDA_CHECK(call)                                                     \
do {                                                                         \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                cudaGetErrorString(err));                                    \
        exit(1);                                                             \
    }                                                                        \
} while(0)

// Save image as PNG (host code)
void save_png(const char *filename, float *image, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Failed to open file %s for writing\n", filename);
        return;
    }
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fclose(fp);
        return;
    }
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        fclose(fp);
        png_destroy_write_struct(&png_ptr, NULL);
        return;
    }
    if (setjmp(png_jmpbuf(png_ptr))) {
        fclose(fp);
        png_destroy_write_struct(&png_ptr, &info_ptr);
        return;
    }
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height,
                 8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
    png_write_info(png_ptr, info_ptr);
    png_bytep row = (png_bytep)malloc(width * sizeof(png_byte));
    if (row == NULL) {
        fprintf(stderr, "Failed to allocate memory for PNG row\n");
        fclose(fp);
        png_destroy_write_struct(&png_ptr, &info_ptr);
        return;
    }
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float val = image[y * width + x];
            if (val < 0.0f) val = 0.0f;
            if (val > 1.0f) val = 1.0f;
            row[x] = (png_byte)(val * 255.0f);
        }
        png_write_row(png_ptr, row);
    }
    png_write_end(png_ptr, NULL);
    fclose(fp);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    free(row);
}

// GPU Kernels

// Kernel: Extract patches from image
__global__ void kernel_extract_patches(const float* __restrict__ d_image, float* __restrict__ d_patches,
                                       int width, int height, int patch_size, int num_patches_x, int num_patches_y) {
    int patch_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_patches = num_patches_x * num_patches_y;
    if (patch_id >= total_patches) return;

    int patch_dim = patch_size * patch_size;
    int py = patch_id / num_patches_x;
    int px = patch_id % num_patches_x;

    float* patch = &d_patches[patch_id * patch_dim];
    for (int dy = 0; dy < patch_size; dy++) {
        for (int dx = 0; dx < patch_size; dx++) {
            patch[dy * patch_size + dx] = d_image[(py + dy)*width + (px + dx)];
        }
    }
}

// Solve small linear system using Gaussian elimination (in-kernel)
__device__ void solve_linear_system(float* ATA, float* ATy, float* x, int dim) {
    // Gaussian elimination without pivoting (for simplicity)
    for (int i = 0; i < dim; i++) {
        float pivot = ATA[i*dim + i];
        if (fabsf(pivot) < 1e-12f) pivot = 1e-6f;

        for (int k = i+1; k < dim; k++) {
            float factor = ATA[k*dim + i]/pivot;
            for (int j = i; j < dim; j++) {
                ATA[k*dim + j] -= factor * ATA[i*dim + j];
            }
            ATy[k] -= factor * ATy[i];
        }
    }

    // Back substitution
    for (int i = dim-1; i >= 0; i--) {
        float val = ATy[i];
        for (int j = i+1; j < dim; j++) {
            val -= ATA[i*dim + j]*x[j];
        }
        x[i] = val / (ATA[i*dim + i] + 1e-12f);
    }
}

// Kernel: OMP sparse coding for each patch
__global__ void kernel_omp_coding(const float* __restrict__ D, const float* __restrict__ patches, float* __restrict__ codes,
                                  int dict_size, int patch_dim, int sparsity, int num_patches)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= num_patches) return;

    const float* y = &patches[p * patch_dim];

    float residual[PATCH_DIM];
    for (int i = 0; i < patch_dim; i++) {
        residual[i] = y[i];
    }

    int selected_atoms[SPARSITY];
    float x[SPARSITY];
    for (int s = 0; s < sparsity; s++) {
        selected_atoms[s] = -1;
        x[s] = 0.0f;
    }

    for (int s = 0; s < sparsity; s++) {
        float max_corr = 0.0f;
        int best_atom = -1;
        // Find best atom
        for (int k = 0; k < dict_size; k++) {
            bool already_selected = false;
            for (int j = 0; j < s; j++) {
                if (selected_atoms[j] == k) {
                    already_selected = true;
                    break;
                }
            }
            if (already_selected) continue;

            float dot = 0.0f;
            const float* atom = &D[k*patch_dim];
            for (int i = 0; i < patch_dim; i++) {
                dot += residual[i]*atom[i];
            }
            if (fabsf(dot) > fabsf(max_corr)) {
                max_corr = dot;
                best_atom = k;
            }
        }

        if (best_atom == -1) break;

        selected_atoms[s] = best_atom;

        // Build A
        int dim = s+1;
        float A[SPARSITY * PATCH_DIM];
        for (int col = 0; col < dim; col++) {
            const float* atom = &D[selected_atoms[col]*patch_dim];
            for (int i = 0; i < patch_dim; i++) {
                A[col*patch_dim + i] = atom[i];
            }
        }

        // Compute ATA
        float ATA[SPARSITY * SPARSITY];
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                float sum = 0.0f;
                for (int k = 0; k < patch_dim; k++) {
                    sum += A[i*patch_dim + k]*A[j*patch_dim + k];
                }
                ATA[i*dim + j] = sum;
            }
        }

        // Compute ATy
        float ATy[SPARSITY];
        for (int i = 0; i < dim; i++) {
            float sum = 0.0f;
            for (int k = 0; k < patch_dim; k++) {
                sum += A[i*patch_dim + k]*y[k];
            }
            ATy[i] = sum;
        }

        for (int i = 0; i < dim; i++) {
            ATA[i*dim + i] += 1e-6f;
        }

        solve_linear_system(ATA, ATy, x, dim);

        // Update residual
        for (int i = 0; i < patch_dim; i++) {
            float Ax_i = 0.0f;
            for (int j = 0; j < dim; j++) {
                Ax_i += A[j*patch_dim + i]*x[j];
            }
            residual[i] = y[i] - Ax_i;
        }
    }

    // Store codes
    for (int s = 0; s < sparsity; s++) {
        int atom = selected_atoms[s];
        if (atom >= 0) {
            codes[p*dict_size + atom] = x[s];
        } else {
            break;
        }
    }
}

// Dictionary update kernels
__global__ void kernel_count_atom_usage(const float* __restrict__ d_codes,
                                        int num_patches, int dict_size, int k,
                                        int* __restrict__ d_patch_indices,
                                        int* __restrict__ d_usage_count)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < num_patches) {
        float val = d_codes[p*dict_size + k];
        if (fabsf(val) > 1e-12f) {
            int idx = atomicAdd(d_usage_count, 1);
            d_patch_indices[idx] = p;
        }
    }
}

__global__ void kernel_build_error_matrix(const float* __restrict__ d_patches,
                                          const float* __restrict__ d_D,
                                          const float* __restrict__ d_codes,
                                          const int* __restrict__ d_patch_indices,
                                          int usage_count,
                                          int patch_dim, int dict_size, int k,
                                          float* __restrict__ d_E)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < usage_count) {
        int p = d_patch_indices[idx];
        const float* patch = &d_patches[p*patch_dim];
        float* E_vec = &d_E[idx*patch_dim];

        // Start E_vec = original patch
        for (int i = 0; i < patch_dim; i++) {
            E_vec[i] = patch[i];
        }

        // subtract sum of D_j*code_j for j != k
        for (int j = 0; j < dict_size; j++) {
            if (j == k) continue;
            float coeff = d_codes[p*dict_size + j];
            if (fabsf(coeff) > 1e-12f) {
                const float* atom = &d_D[j*patch_dim];
                for (int i = 0; i < patch_dim; i++) {
                    E_vec[i] -= atom[i]*coeff;
                }
            }
        }
    }
}

__global__ void kernel_compute_new_atom(const float* __restrict__ d_E,
                                        int usage_count,
                                        int patch_dim,
                                        float* __restrict__ d_new_atom)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < patch_dim) {
        float sum_val = 0.0f;
        for (int u = 0; u < usage_count; u++) {
            sum_val += d_E[u*patch_dim + i];
        }
        if (usage_count > 0) sum_val /= (float)usage_count;
        d_new_atom[i] = sum_val;
    }
}

__global__ void kernel_update_atom_and_codes(float* __restrict__ d_new_atom,
                                             float* __restrict__ d_D,
                                             float* __restrict__ d_codes,
                                             const int* __restrict__ d_patch_indices,
                                             const float* __restrict__ d_E,
                                             int usage_count,
                                             int patch_dim,
                                             int dict_size,
                                             int k)
{
    // Normalize the atom
    __shared__ float norm;
    if (threadIdx.x == 0) {
        float n = 0.0f;
        for (int i = 0; i < patch_dim; i++) {
            float val = d_new_atom[i];
            n += val*val;
        }
        norm = sqrtf(n) + 1e-6f;
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < patch_dim) {
        d_new_atom[i] /= norm;
    }
    __syncthreads();

    // One thread updates dictionary and codes
    if (i == 0) {
        for (int idx = 0; idx < patch_dim; idx++) {
            d_D[k*patch_dim + idx] = d_new_atom[idx];
        }

        // Update codes: x_k = <E_i, d_k_new>
        for (int u = 0; u < usage_count; u++) {
            int p = d_patch_indices[u];
            float dot = 0.0f;
            for (int n = 0; n < patch_dim; n++) {
                dot += d_E[u*patch_dim + n]*d_new_atom[n];
            }
            d_codes[p*dict_size + k] = dot;
        }
    }
}

// Kernel: Reconstruct patches from codes and dictionary
__global__ void kernel_reconstruct_patches(const float* __restrict__ d_codes,
                                           const float* __restrict__ d_D,
                                           float* __restrict__ d_reconstructed_patches,
                                           int num_patches, int patch_dim, int dict_size)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < num_patches) {
        float* recon_patch = &d_reconstructed_patches[p*patch_dim];
        for (int i = 0; i < patch_dim; i++) {
            recon_patch[i] = 0.0f;
        }
        for (int k = 0; k < dict_size; k++) {
            float coeff = d_codes[p*dict_size + k];
            if (fabsf(coeff) > 1e-12f) {
                const float* atom = &d_D[k*patch_dim];
                for (int i = 0; i < patch_dim; i++) {
                    recon_patch[i] += atom[i]*coeff;
                }
            }
        }
    }
}

// Kernel: Reconstruct the image from reconstructed patches
__global__ void kernel_reconstruct_image(const float* __restrict__ d_reconstructed_patches,
                                         float* __restrict__ d_image,
                                         float* __restrict__ d_weight_image,
                                         int width, int height, int patch_size,
                                         int num_patches_x, int num_patches_y)
{
    int patch_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_patches = num_patches_x * num_patches_y;
    if (patch_id >= total_patches) return;

    int patch_dim = patch_size * patch_size;
    int py = patch_id / num_patches_x;
    int px = patch_id % num_patches_x;

    const float* recon_patch = &d_reconstructed_patches[patch_id * patch_dim];
    for (int dy = 0; dy < patch_size; dy++) {
        for (int dx = 0; dx < patch_size; dx++) {
            int idx = (py + dy)*width + (px + dx);
            atomicAdd(&d_image[idx], recon_patch[dy*patch_size + dx]);
            atomicAdd(&d_weight_image[idx], 1.0f);
        }
    }
}

// Normalize the reconstructed image
__global__ void kernel_normalize_image(float* __restrict__ d_image,
                                       const float* __restrict__ d_weight_image,
                                       int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        if (d_weight_image[i] > 0.0f) {
            d_image[i] = d_image[i]/d_weight_image[i];
        }
        if (d_image[i] < 0.0f) d_image[i] = 0.0f;
        if (d_image[i] > 1.0f) d_image[i] = 1.0f;
    }
}

// Main K-SVD function
void ksvd_denoise(float *image, int width, int height, int patch_size, int dict_size, int sparsity, int max_iter) {
    int num_patches_x = width - patch_size + 1;
    int num_patches_y = height - patch_size + 1;
    int num_patches = num_patches_x * num_patches_y;
    int patch_dim = patch_size * patch_size;

    float *d_image, *d_patches, *d_D, *d_codes, *d_reconstructed_patches, *d_accum_image, *d_weight_image;

    CUDA_CHECK(cudaMalloc(&d_image, sizeof(float)*width*height));
    CUDA_CHECK(cudaMemcpy(d_image, image, sizeof(float)*width*height, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_patches, sizeof(float)*num_patches*patch_dim));

    int threads = 256;
    int blocks_patches = (num_patches + threads - 1) / threads;

    // Extract patches
    kernel_extract_patches<<<blocks_patches, threads>>>(d_image, d_patches, width, height, patch_size, num_patches_x, num_patches_y);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Initialize Dictionary on CPU
    float *h_patches = (float*)malloc(sizeof(float)*num_patches*patch_dim);
    CUDA_CHECK(cudaMemcpy(h_patches, d_patches, sizeof(float)*num_patches*patch_dim, cudaMemcpyDeviceToHost));

    float *h_D = (float*)malloc(sizeof(float)*dict_size*patch_dim);
    srand(0);
    for (int k = 0; k < dict_size; k++) {
        int idx = rand() % num_patches;
        for (int i = 0; i < patch_dim; i++) {
            h_D[k*patch_dim + i] = h_patches[idx*patch_dim + i];
        }
        float norm = 0.0f;
        for (int i = 0; i < patch_dim; i++) {
            norm += h_D[k*patch_dim + i]*h_D[k*patch_dim + i];
        }
        norm = sqrtf(norm) + 1e-6f;
        for (int i = 0; i < patch_dim; i++) {
            h_D[k*patch_dim + i] /= norm;
        }
    }

    free(h_patches);

    CUDA_CHECK(cudaMalloc(&d_D, sizeof(float)*dict_size*patch_dim));
    CUDA_CHECK(cudaMemcpy(d_D, h_D, sizeof(float)*dict_size*patch_dim, cudaMemcpyHostToDevice));
    free(h_D);

    CUDA_CHECK(cudaMalloc(&d_codes, sizeof(float)*num_patches*dict_size));
    CUDA_CHECK(cudaMemset(d_codes, 0, sizeof(float)*num_patches*dict_size));

    // Buffers for dictionary update
    int *d_patch_indices, *d_usage_count;
    float *d_E, *d_new_atom;
    CUDA_CHECK(cudaMalloc(&d_patch_indices, sizeof(int)*num_patches));
    CUDA_CHECK(cudaMalloc(&d_usage_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_E, sizeof(float)*num_patches*patch_dim));
    CUDA_CHECK(cudaMalloc(&d_new_atom, sizeof(float)*patch_dim));

    for (int iter = 0; iter < max_iter; iter++) {
        printf("Iteration %d\n", iter+1);

        // Sparse coding step
        CUDA_CHECK(cudaMemset(d_codes, 0, sizeof(float)*num_patches*dict_size));
        kernel_omp_coding<<<blocks_patches, threads>>>(d_D, d_patches, d_codes, dict_size, patch_dim, sparsity, num_patches);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Dictionary update step
        for (int k = 0; k < dict_size; k++) {
            CUDA_CHECK(cudaMemset(d_usage_count, 0, sizeof(int)));
            kernel_count_atom_usage<<<blocks_patches, threads>>>(d_codes, num_patches, dict_size, k, d_patch_indices, d_usage_count);
            CUDA_CHECK(cudaDeviceSynchronize());

            int h_usage_count;
            CUDA_CHECK(cudaMemcpy(&h_usage_count, d_usage_count, sizeof(int), cudaMemcpyDeviceToHost));

            if (h_usage_count == 0) {
                // Atom not used, skip
                continue;
            }

            int blocks_usage = (h_usage_count + threads - 1)/threads;
            kernel_build_error_matrix<<<blocks_usage, threads>>>(d_patches, d_D, d_codes, d_patch_indices,
                                                                 h_usage_count, patch_dim, dict_size, k, d_E);
            CUDA_CHECK(cudaDeviceSynchronize());

            int blocks_patch_dim = (patch_dim + threads - 1)/threads;
            kernel_compute_new_atom<<<blocks_patch_dim, threads>>>(d_E, h_usage_count, patch_dim, d_new_atom);
            CUDA_CHECK(cudaDeviceSynchronize());

            kernel_update_atom_and_codes<<<1, threads>>>(d_new_atom, d_D, d_codes, d_patch_indices, d_E,
                                                         h_usage_count, patch_dim, dict_size, k);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // Reconstruct patches
    CUDA_CHECK(cudaMalloc(&d_reconstructed_patches, sizeof(float)*num_patches*patch_dim));
    kernel_reconstruct_patches<<<blocks_patches, threads>>>(d_codes, d_D, d_reconstructed_patches, num_patches, patch_dim, dict_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Reconstruct image
    CUDA_CHECK(cudaMalloc(&d_accum_image, sizeof(float)*width*height));
    CUDA_CHECK(cudaMalloc(&d_weight_image, sizeof(float)*width*height));
    CUDA_CHECK(cudaMemset(d_accum_image, 0, sizeof(float)*width*height));
    CUDA_CHECK(cudaMemset(d_weight_image, 0, sizeof(float)*width*height));

    kernel_reconstruct_image<<<blocks_patches, threads>>>(d_reconstructed_patches, d_accum_image, d_weight_image,
                                                          width, height, patch_size, num_patches_x, num_patches_y);
    CUDA_CHECK(cudaDeviceSynchronize());

    int total_size = width*height;
    int blocks_pix = (total_size + threads - 1)/threads;
    kernel_normalize_image<<<blocks_pix, threads>>>(d_accum_image, d_weight_image, total_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(image, d_accum_image, sizeof(float)*width*height, cudaMemcpyDeviceToHost));

    // Free GPU memory
    cudaFree(d_image);
    cudaFree(d_patches);
    cudaFree(d_D);
    cudaFree(d_codes);
    cudaFree(d_patch_indices);
    cudaFree(d_usage_count);
    cudaFree(d_E);
    cudaFree(d_new_atom);
    cudaFree(d_reconstructed_patches);
    cudaFree(d_accum_image);
    cudaFree(d_weight_image);
}

int main(int argc, char *argv[])
{
    struct timespec start, stop;
    double time;

    int width = WIDTH;
    int height = HEIGHT;
    float noise_level = 0.1f; // Adjust noise level if desired

    // loading image
    FILE* fp;
    unsigned char *image_u8 = (unsigned char*)malloc(sizeof(unsigned char)*width*height);
    if(!(fp = fopen(INPUT_FILE, "rb")))
    {
        printf("Cannot open input file\n");
        return 1;
    }
    printf("Image loaded: %dx%d\n", width, height);
    fread(image_u8, sizeof(unsigned char), width*height, fp);
    fclose(fp);

    // Convert image to float in range [0,1]
    float *image = (float*)malloc(sizeof(float)*width*height);
    for(int i = 0; i < width * height; i++) {
        image[i] = image_u8[i] / 255.0f;
    }
    free(image_u8);

    // adding noise
    srand(0);
    for (int i = 0; i < width * height; i++)
    {
        float noise = noise_level * ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        image[i] = fmaxf(0.0f, fminf(1.0f, image[i] + noise));
    }

    // saving noisy image
    save_png(NOISY_FILE, image, width, height);
    printf("Noisy image saved to %s\n", NOISY_FILE);

    if(clock_gettime(CLOCK_REALTIME, &start) == -1){perror("clock gettime");}

    // Run K-SVD on GPU
    ksvd_denoise(image, width, height, PATCH_SIZE, DICT_SIZE, SPARSITY, MAX_ITER);

    if(clock_gettime(CLOCK_REALTIME, &stop) == -1){perror("clock gettime");}
    time = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec)/1e9;
    printf("Execution Time = %f sec \n", time);

    // saving denoised image
    save_png(DENOISED_FILE, image, width, height);
    printf("Denoised image saved to %s\n", DENOISED_FILE);

    free(image);

    return 0;
}
