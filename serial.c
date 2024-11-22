#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <png.h>
#include <time.h>
#include <string.h>

#define PATCH_SIZE 8
#define MAX_ITER 5   // Reduced for testing purposes
#define DICT_SIZE 512 // Reduced for testing purposes
#define SPARSITY 10    // Sparsity level for OMP
#define WIDTH 800     // Width of the image
#define HEIGHT 800    // Height of the image
#define INPUT_FILE "input.raw"        // Input RAW file
#define NOISY_FILE "noisy_image.png"  // Output noisy image
#define DENOISED_FILE "denoised_image.png" // Output denoised image

/* Function Prototypes */
int load_raw(const char *filename, float **image, int width, int height);
void add_noise(float *image, int width, int height, float noise_level);
void k_svd_denoise(float *image, int width, int height);
void save_png(const char *filename, float *image, int width, int height);
void extract_patches(float *image, int width, int height, float **patches, int *num_patches);
void reconstruct_image(float *image, float *patches, int width, int height, int num_patches);
void omp(float *D, int dict_size, float *X, int num_patches, int patch_vector_size, int sparsity, float *Gamma);
void update_dictionary(float *X, float *D, float *Gamma, int num_patches, int patch_vector_size, int dict_size);

int main(int argc, char *argv[]) {
    float *image;
    struct timespec start, stop;
    double time;
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

    if(clock_gettime(CLOCK_REALTIME, &start) == -1){perror("clock gettime");}
    /* Apply K-SVD denoising */
    k_svd_denoise(image, WIDTH, HEIGHT);

    if(clock_gettime(CLOCK_REALTIME, &stop) == -1){perror("clock gettime");}
    time = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec)/1e9;
    printf("Execution Time = %f sec \n", time);

    /* Save the denoised image as a PNG */
    save_png(DENOISED_FILE, image, WIDTH, HEIGHT);
    printf("Denoised image saved to %s\n", DENOISED_FILE);

    /* Cleanup */
    free(image);

    return 0;
}

/* Load RAW image and convert to float with fixed dimensions (800x800) */
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

    for (int iter = 0; iter < MAX_ITER; iter++) {
        printf("Iteration %d/%d\n", iter + 1, MAX_ITER);
        omp(D, DICT_SIZE, patches, num_patches, patch_vector_size, SPARSITY, Gamma);
        update_dictionary(patches, D, Gamma, num_patches, patch_vector_size, DICT_SIZE);
    }

    /* Reconstruct patches using the learned dictionary and sparse codes */
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

    free(patches);
    free(D);
    free(Gamma);
}

/* Orthogonal Matching Pursuit (OMP) */
void omp(float *D, int dict_size, float *X, int num_patches, int patch_vector_size, int sparsity, float *Gamma) {
    for (int i = 0; i < num_patches; i++) {
        float *x = &X[i * patch_vector_size];
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
            Gamma[indices[s] * num_patches + i] = coeffs[s];
        }

        free(residual);
        free(indices);
        free(coeffs);
    }
}

/* Update Dictionary using a simplified method */
void update_dictionary(float *X, float *D, float *Gamma, int num_patches, int patch_vector_size, int dict_size) {
    for (int k = 0; k < dict_size; k++) {
        // Find all patches that use atom k
        float *omega = (float *)malloc(num_patches * sizeof(float));
        int usage = 0;
        for (int n = 0; n < num_patches; n++) {
            omega[n] = Gamma[k * num_patches + n];
            if (omega[n] != 0.0f) {
                usage++;
            }
        }
        if (usage == 0) {
            free(omega);
            continue;
        }

        // Compute the representation error for these patches excluding atom k
        float *E = (float *)calloc(patch_vector_size, sizeof(float));
        for (int n = 0; n < num_patches; n++) {
            if (omega[n] != 0.0f) {
                float *x_n = &X[n * patch_vector_size];
                float *x_hat = (float *)calloc(patch_vector_size, sizeof(float));
                for (int j = 0; j < dict_size; j++) {
                    if (j != k) {
                        float gamma_jn = Gamma[j * num_patches + n];
                        if (gamma_jn != 0.0f) {
                            for (int l = 0; l < patch_vector_size; l++) {
                                x_hat[l] += D[j * patch_vector_size + l] * gamma_jn;
                            }
                        }
                    }
                }
                for (int l = 0; l < patch_vector_size; l++) {
                    E[l] += (x_n[l] - x_hat[l]) * omega[n];
                }
                free(x_hat);
            }
        }

        // Update atom k
        float norm = 0.0f;
        for (int l = 0; l < patch_vector_size; l++) {
            D[k * patch_vector_size + l] = E[l];
            norm += E[l] * E[l];
        }
        norm = sqrtf(norm);
        if (norm > 0) {
            for (int l = 0; l < patch_vector_size; l++) {
                D[k * patch_vector_size + l] /= norm;
            }
        }

        free(E);
        free(omega);
    }
}

/* Extract overlapping patches from the image */
void extract_patches(float *image, int width, int height, float **patches, int *num_patches) {
    int stride = 1; // Overlapping patches with stride of 1
    int patches_per_row = width - PATCH_SIZE + 1;
    int patches_per_col = height - PATCH_SIZE + 1;
    *num_patches = patches_per_row * patches_per_col;
    int patch_vector_size = PATCH_SIZE * PATCH_SIZE;
    *patches = (float *)malloc((*num_patches) * patch_vector_size * sizeof(float));
    if (*patches == NULL) {
        fprintf(stderr, "Failed to allocate memory for patches\n");
        exit(1);
    }
    int patch_idx = 0;
    for (int i = 0; i <= height - PATCH_SIZE; i += stride) {
        for (int j = 0; j <= width - PATCH_SIZE; j += stride) {
            for (int m = 0; m < PATCH_SIZE; m++) {
                for (int n = 0; n < PATCH_SIZE; n++) {
                    (*patches)[patch_idx * patch_vector_size + m * PATCH_SIZE + n] = image[(i + m) * width + (j + n)];
                }
            }
            patch_idx++;
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
    int patch_vector_size = PATCH_SIZE * PATCH_SIZE;
    int stride = 1; // Use the same stride as in patch extraction
    int patch_idx = 0;
    for (int y = 0; y <= height - PATCH_SIZE; y += stride) {
        for (int x = 0; x <= width - PATCH_SIZE; x += stride) {
            for (int m = 0; m < PATCH_SIZE; m++) {
                for (int n = 0; n < PATCH_SIZE; n++) {
                    int idx = (y + m) * width + (x + n);
                    reconstructed_image[idx] += patches[patch_idx * patch_vector_size + m * PATCH_SIZE + n];
                    weights[idx] += 1.0f;
                }
            }
            patch_idx++;
        }
    }
    // Normalize the reconstructed image
    for (int i = 0; i < width * height; i++) {
        if (weights[i] > 0) {
            image[i] = reconstructed_image[i] / weights[i];
        }
    }
    free(reconstructed_image);
    free(weights);
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
