#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <png.h>
#include <time.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define PATCH_SIZE 8     // Reduced for better performance
#define MAX_ITER 10
#define DICT_SIZE 512
#define SPARSITY 20
#define WIDTH 800
#define HEIGHT 800
#define INPUT_FILE "input.raw"
#define NOISY_FILE "noisy_image.png"
#define DENOISED_FILE "denoised_image.png"

#define BLOCK_SIZE 256   // Number of threads per block

/* Function Prototypes */
int load_raw(const char *filename, float **image, int width, int height);
void add_noise(float *image, int width, int height, float noise_level);
void k_svd_denoise(float *image, int width, int height);
void save_png(const char *filename, float *image, int width, int height);
void extract_patches(float *image, int width, int height, float **patches, int *num_patches);
void reconstruct_image(float *image, float *patches, int width, int height, int num_patches);

/* CUDA Kernels */
__global__ void extract_patches_kernel(float *image, float *patches, int width, int height, int num_patches);
__global__ void reconstruct_image_kernel(float *reconstructed_image, float *patches, float *weights, int width, int height, int num_patches);
__global__ void omp_kernel(float *D, int dict_size, float *X, int num_patches, int patch_vector_size, int sparsity, float *Gamma);
__global__ void update_dictionary_kernel(float *X, float *D, float *Gamma, float *omega, float *E, int num_patches, int patch_vector_size, int dict_size);

int main(int argc, char *argv[]) {
    float *image;
    if (load_raw(INPUT_FILE, &image, WIDTH, HEIGHT) != 0) {
        fprintf(stderr, "Failed to load raw image %s\n", INPUT_FILE);
        return 1;
    }
    printf("Image loaded: %dx%d\n", WIDTH, HEIGHT);

    /* Add synthetic noise */
    add_noise(image, WIDTH, HEIGHT, 0.1f);

    /* Save the noisy image as a PNG */
    save_png(NOISY_FILE, image, WIDTH, HEIGHT);
    printf("Noisy image saved to %s\n", NOISY_FILE);

    /* Apply K-SVD denoising */
    k_svd_denoise(image, WIDTH, HEIGHT);

    /* Save the denoised image as a PNG */
    save_png(DENOISED_FILE, image, WIDTH, HEIGHT);
    printf("Denoised image saved to %s\n", DENOISED_FILE);

    /* Cleanup */
    free(image);

    return 0;
}

/* Load RAW image and convert to float with fixed dimensions */
int load_raw(const char *filename, float **image, int width, int height) {
    FILE *fp;
    if (!(fp = fopen(filename, "rb"))) {
        printf("Cannot open RAW file %s\n", filename);
        return -1;
    }
    unsigned char *raw_data = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    if (raw_data == NULL) {
        fprintf(stderr, "Failed to allocate memory for raw_data\n");
        fclose(fp);
        return -1;
    }
    size_t read_count = fread(raw_data, sizeof(unsigned char), width * height, fp);
    fclose(fp);
    if (read_count != width * height) {
        fprintf(stderr, "Failed to read raw image data\n");
        free(raw_data);
        return -1;
    }

    *image = (float *)malloc(width * height * sizeof(float));
    if (*image == NULL) {
        fprintf(stderr, "Failed to allocate memory for image\n");
        free(raw_data);
        return -1;
    }
    for (int i = 0; i < width * height; i++) {
        (*image)[i] = raw_data[i] / 255.0f;
    }
    free(raw_data);
    return 0;
}

/* Add synthetic noise to an image */
void add_noise(float *image, int width, int height, float noise_level) {
    srand(0); // Use a fixed seed for reproducibility
    for (int i = 0; i < width * height; i++) {
        float noise = noise_level * ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        image[i] = fmaxf(0.0f, fminf(1.0f, image[i] + noise));
    }
}

/* Apply K-SVD denoising to an image */
void k_svd_denoise(float *image, int width, int height) {
    int patch_vector_size = PATCH_SIZE * PATCH_SIZE;
    int num_patches;
    float *patches;

    extract_patches(image, width, height, &patches, &num_patches);

    /* Initialize Dictionary with random patches */
    float *D = (float *)malloc(DICT_SIZE * patch_vector_size * sizeof(float));
    if (D == NULL) {
        fprintf(stderr, "Failed to allocate memory for dictionary D\n");
        free(patches);
        return;
    }
    srand(0); // Fixed seed for reproducibility
    for (int i = 0; i < DICT_SIZE * patch_vector_size; i++) {
        D[i] = ((float)rand() / RAND_MAX) - 0.5f;
    }
    /* Normalize dictionary atoms */
    for (int k = 0; k < DICT_SIZE; k++) {
        float norm = 0.0f;
        for (int j = 0; j < patch_vector_size; j++) {
            norm += D[k * patch_vector_size + j] * D[k * patch_vector_size + j];
        }
        norm = sqrtf(norm);
        if (norm > 0) {
            for (int j = 0; j < patch_vector_size; j++) {
                D[k * patch_vector_size + j] /= norm;
            }
        }
    }

    float *Gamma = (float *)calloc(DICT_SIZE * num_patches, sizeof(float));
    if (Gamma == NULL) {
        fprintf(stderr, "Failed to allocate memory for sparse codes Gamma\n");
        free(patches);
        free(D);
        return;
    }

    /* Copy data to GPU */
    float *d_patches, *d_D, *d_Gamma, *d_omega, *d_E;
    cudaMalloc((void **)&d_patches, num_patches * patch_vector_size * sizeof(float));
    cudaMalloc((void **)&d_D, DICT_SIZE * patch_vector_size * sizeof(float));
    cudaMalloc((void **)&d_Gamma, DICT_SIZE * num_patches * sizeof(float));
    cudaMalloc((void **)&d_omega, num_patches * sizeof(float));
    cudaMalloc((void **)&d_E, patch_vector_size * sizeof(float));

    cudaMemcpy(d_patches, patches, num_patches * patch_vector_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, D, DICT_SIZE * patch_vector_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_Gamma, 0, DICT_SIZE * num_patches * sizeof(float));

    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (num_patches + threadsPerBlock - 1) / threadsPerBlock;

    for (int iter = 0; iter < MAX_ITER; iter++) {
        printf("Iteration %d/%d\n", iter + 1, MAX_ITER);

        /* OMP */
        omp_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_D, DICT_SIZE, d_patches, num_patches, patch_vector_size, SPARSITY, d_Gamma);
        cudaDeviceSynchronize();

        /* Dictionary Update */
        update_dictionary_kernel<<<DICT_SIZE, threadsPerBlock>>>(d_patches, d_D, d_Gamma, d_omega, d_E, num_patches, patch_vector_size, DICT_SIZE);
        cudaDeviceSynchronize();
    }

    /* Reconstruct patches using the learned dictionary and sparse codes */
    cudaMemcpy(patches, d_patches, num_patches * patch_vector_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(D, d_D, DICT_SIZE * patch_vector_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Gamma, d_Gamma, DICT_SIZE * num_patches * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_patches; i++) {
        float *reconstructed_patch = &patches[i * patch_vector_size];
        memset(reconstructed_patch, 0, patch_vector_size * sizeof(float));
        for (int k = 0; k < DICT_SIZE; k++) {
            float coeff = Gamma[k * num_patches + i];
            if (coeff != 0.0f) {
                for (int j = 0; j < patch_vector_size; j++) {
                    reconstructed_patch[j] += coeff * D[k * patch_vector_size + j];
                }
            }
        }
    }

    /* Reconstruct the full image from patches */
    reconstruct_image(image, patches, width, height, num_patches);

    /* Free GPU memory */
    cudaFree(d_patches);
    cudaFree(d_D);
    cudaFree(d_Gamma);
    cudaFree(d_omega);
    cudaFree(d_E);

    free(patches);
    free(D);
    free(Gamma);
}

/* Extract overlapping patches from the image */
void extract_patches(float *image, int width, int height, float **patches, int *num_patches) {
    int patches_per_row = width - PATCH_SIZE + 1;
    int patches_per_col = height - PATCH_SIZE + 1;
    *num_patches = patches_per_row * patches_per_col;
    int patch_vector_size = PATCH_SIZE * PATCH_SIZE;
    *patches = (float *)malloc((*num_patches) * patch_vector_size * sizeof(float));
    if (*patches == NULL) {
        fprintf(stderr, "Failed to allocate memory for patches\n");
        exit(1);
    }

    /* Copy image to GPU */
    float *d_image, *d_patches;
    cudaMalloc((void **)&d_image, width * height * sizeof(float));
    cudaMalloc((void **)&d_patches, (*num_patches) * patch_vector_size * sizeof(float));
    cudaMemcpy(d_image, image, width * height * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (*num_patches + threadsPerBlock - 1) / threadsPerBlock;

    extract_patches_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_patches, width, height, *num_patches);
    cudaDeviceSynchronize();

    /* Copy patches back to host */
    cudaMemcpy(*patches, d_patches, (*num_patches) * patch_vector_size * sizeof(float), cudaMemcpyDeviceToHost);

    /* Free GPU memory */
    cudaFree(d_image);
    cudaFree(d_patches);
}

/* CUDA Kernel for extracting patches */
__global__ void extract_patches_kernel(float *image, float *patches, int width, int height, int num_patches) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_patches) return;

    int patches_per_row = width - PATCH_SIZE + 1;
    int i = idx / patches_per_row;
    int j = idx % patches_per_row;

    int patch_vector_size = PATCH_SIZE * PATCH_SIZE;
    for (int m = 0; m < PATCH_SIZE; m++) {
        for (int n = 0; n < PATCH_SIZE; n++) {
            int image_idx = (i + m) * width + (j + n);
            int patch_idx = idx * patch_vector_size + m * PATCH_SIZE + n;
            patches[patch_idx] = image[image_idx];
        }
    }
}

/* Reconstruct the image from overlapping patches */
void reconstruct_image(float *image, float *patches, int width, int height, int num_patches) {
    float *reconstructed_image = (float *)calloc(width * height, sizeof(float));
    float *weights = (float *)calloc(width * height, sizeof(float));
    if (reconstructed_image == NULL || weights == NULL) {
        fprintf(stderr, "Failed to allocate memory for image reconstruction\n");
        exit(1);
    }

    /* Copy data to GPU */
    float *d_reconstructed_image, *d_weights, *d_patches;
    cudaMalloc((void **)&d_reconstructed_image, width * height * sizeof(float));
    cudaMalloc((void **)&d_weights, width * height * sizeof(float));
    cudaMalloc((void **)&d_patches, num_patches * PATCH_SIZE * PATCH_SIZE * sizeof(float));

    cudaMemset(d_reconstructed_image, 0, width * height * sizeof(float));
    cudaMemset(d_weights, 0, width * height * sizeof(float));
    cudaMemcpy(d_patches, patches, num_patches * PATCH_SIZE * PATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = num_patches;

    reconstruct_image_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_reconstructed_image, d_patches, d_weights, width, height, num_patches);
    cudaDeviceSynchronize();

    /* Copy data back to host */
    cudaMemcpy(reconstructed_image, d_reconstructed_image, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(weights, d_weights, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // Normalize the reconstructed image
    for (int i = 0; i < width * height; i++) {
        if (weights[i] > 0) {
            image[i] = reconstructed_image[i] / weights[i];
        }
    }

    /* Free GPU memory */
    cudaFree(d_reconstructed_image);
    cudaFree(d_weights);
    cudaFree(d_patches);

    free(reconstructed_image);
    free(weights);
}

/* CUDA Kernel for reconstructing image */
__global__ void reconstruct_image_kernel(float *reconstructed_image, float *patches, float *weights, int width, int height, int num_patches) {
    int idx = blockIdx.x; // One block per patch
    int thread_id = threadIdx.x;

    int patches_per_row = width - PATCH_SIZE + 1;
    int i = idx / patches_per_row;
    int j = idx % patches_per_row;

    int patch_vector_size = PATCH_SIZE * PATCH_SIZE;

    for (int k = thread_id; k < patch_vector_size; k += blockDim.x) {
        int m = k / PATCH_SIZE;
        int n = k % PATCH_SIZE;
        int image_idx = (i + m) * width + (j + n);
        int patch_idx = idx * patch_vector_size + m * PATCH_SIZE + n;

        atomicAdd(&reconstructed_image[image_idx], patches[patch_idx]);
        atomicAdd(&weights[image_idx], 1.0f);
    }
}

/* CUDA Kernel for OMP */
__global__ void omp_kernel(float *D, int dict_size, float *X, int num_patches, int patch_vector_size, int sparsity, float *Gamma) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_patches) return;

    float *x = &X[idx * patch_vector_size];
    float *residual = (float *)malloc(patch_vector_size * sizeof(float));
    memcpy(residual, x, patch_vector_size * sizeof(float));
    int *indices = (int *)malloc(sparsity * sizeof(int));
    float *coeffs = (float *)malloc(sparsity * sizeof(float));
    int nnz = 0;

    while (nnz < sparsity) {
        // Compute correlations
        float max_corr = 0.0f;
        int max_idx = -1;
        for (int k = 0; k < dict_size; k++) {
            // Skip already selected atoms
            int already_selected = 0;
            for (int s = 0; s < nnz; s++) {
                if (indices[s] == k) {
                    already_selected = 1;
                    break;
                }
            }
            if (already_selected) continue;

            float dot = 0.0f;
            for (int j = 0; j < patch_vector_size; j++) {
                dot += D[k * patch_vector_size + j] * residual[j];
            }
            float corr = fabsf(dot);
            if (corr > max_corr) {
                max_corr = corr;
                max_idx = k;
            }
        }
        if (max_idx == -1) {
            break;
        }
        indices[nnz] = max_idx;
        // Compute the coefficient for this atom
        float numerator = 0.0f;
        float denominator = 0.0f;
        for (int j = 0; j < patch_vector_size; j++) {
            numerator += D[max_idx * patch_vector_size + j] * residual[j];
            denominator += D[max_idx * patch_vector_size + j] * D[max_idx * patch_vector_size + j];
        }
        float coeff = numerator / denominator;
        coeffs[nnz] = coeff;
        nnz++;

        // Update residual
        for (int j = 0; j < patch_vector_size; j++) {
            residual[j] -= coeff * D[max_idx * patch_vector_size + j];
        }
    }

    // Store the sparse codes
    for (int s = 0; s < nnz; s++) {
        Gamma[indices[s] * num_patches + idx] = coeffs[s];
    }

    free(residual);
    free(indices);
    free(coeffs);
}

/* CUDA Kernel for Dictionary Update */
__global__ void update_dictionary_kernel(float *X, float *D, float *Gamma, float *omega, float *E, int num_patches, int patch_vector_size, int dict_size) {
    int k = blockIdx.x; // One block per atom
    int thread_id = threadIdx.x;

    // Clear omega and E in shared or global memory
    if (thread_id == 0) {
        for (int i = 0; i < num_patches; i++) {
            omega[i] = 0.0f;
        }
        for (int i = 0; i < patch_vector_size; i++) {
            E[i] = 0.0f;
        }
    }
    __syncthreads();

    // Find all patches that use atom k
    int usage = 0;
    for (int n = thread_id; n < num_patches; n += blockDim.x) {
        omega[n] = Gamma[k * num_patches + n];
        if (omega[n] != 0.0f) {
            atomicAdd(&usage, 1);
        }
    }
    __syncthreads();
    if (usage == 0) {
        return;
    }

    // Compute the representation error for these patches excluding atom k
    for (int n = thread_id; n < num_patches; n += blockDim.x) {
        if (omega[n] != 0.0f) {
            float *x_n = &X[n * patch_vector_size];
            for (int l = 0; l < patch_vector_size; l++) {
                atomicAdd(&E[l], (x_n[l]) * omega[n]);  // Update representation error
            }
        }
    }
    __syncthreads();

    // Update atom k
    float norm = 0.0f;
    for (int l = thread_id; l < patch_vector_size; l += blockDim.x) {
        D[k * patch_vector_size + l] = E[l];
        atomicAdd(&norm, E[l] * E[l]);
    }
    __syncthreads();

    norm = sqrtf(norm);
    if (norm > 0) {
        for (int l = thread_id; l < patch_vector_size; l += blockDim.x) {
            D[k * patch_vector_size + l] /= norm;
        }
    }
}

/* Save the image as a PNG file */
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
