#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <png.h>
#include <time.h>
#include <string.h>
#include <omp.h> // Include OpenMP header

#define PATCH_SIZE 8
#define MAX_ITER 20
#define DICT_SIZE 512
#define SPARSITY 10
#define WIDTH 800
#define HEIGHT 800
#define INPUT_FILE "input.raw"
#define NOISY_FILE "noisy_image.png"
#define DENOISED_FILE "denoised_image.png"

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

// K-SVD Denoising Function
void ksvd_denoise(float *image, int width, int height, int patch_size, int dict_size, int sparsity, int max_iter) {
    int num_patches_x = width - patch_size + 1;
    int num_patches_y = height - patch_size + 1;
    int num_patches = num_patches_x * num_patches_y;
    int patch_dim = patch_size * patch_size;

    // Extract patches
    float *patches = (float*)malloc(sizeof(float)*num_patches*patch_dim);
    int patch_idx = 0;
    for(int y = 0; y < num_patches_y; y++) {
        for(int x = 0; x < num_patches_x; x++) {
            for(int dy = 0; dy < patch_size; dy++) {
                for(int dx = 0; dx < patch_size; dx++) {
                    patches[patch_idx*patch_dim + dy*patch_size + dx] =
                        image[(y + dy)*width + (x + dx)];
                }
            }
            patch_idx++;
        }
    }

    // Initialize dictionary D with random patches
    float *D = (float*)malloc(sizeof(float)*dict_size*patch_dim);
    srand(0);
    for(int k = 0; k < dict_size; k++) {
        int idx = rand() % num_patches;
        for(int i = 0; i < patch_dim; i++) {
            D[k*patch_dim + i] = patches[idx*patch_dim + i];
        }
        // Normalize the atom
        float norm = 0.0f;
        for(int i = 0; i < patch_dim; i++) {
            norm += D[k*patch_dim + i]*D[k*patch_dim + i];
        }
        norm = sqrtf(norm);
        for(int i = 0; i < patch_dim; i++) {
            D[k*patch_dim + i] /= norm + 1e-6f;
        }
    }

    // Initialize sparse codes
    float *codes = (float*)malloc(sizeof(float)*num_patches*dict_size);

    // K-SVD iterations
    for(int iter = 0; iter < max_iter; iter++) {
        printf("Iteration %d\n", iter + 1);

        // Reset codes array at the beginning of each iteration
        memset(codes, 0, sizeof(float)*num_patches*dict_size);

        // Sparse coding step using OMP
        #pragma omp parallel for schedule(static)
        for(int p = 0; p < num_patches; p++) {
            // Variables private to each thread
            float residual[patch_dim];
            int selected_atoms[sparsity];
            float x[sparsity];

            // Get the current patch
            float *y = &patches[p*patch_dim];

            // Initialize residual
            memcpy(residual, y, sizeof(float)*patch_dim);

            // Initialize indices and coefficients
            memset(selected_atoms, -1, sizeof(int)*sparsity);
            memset(x, 0, sizeof(float)*sparsity);

            // OMP algorithm
            for(int s = 0; s < sparsity; s++) {
                // Compute correlations
                float max_corr = 0.0f;
                int best_atom = -1;
                for(int k = 0; k < dict_size; k++) {
                    // Skip if atom already selected
                    int already_selected = 0;
                    for(int j = 0; j < s; j++) {
                        if(selected_atoms[j] == k) {
                            already_selected = 1;
                            break;
                        }
                    }
                    if(already_selected)
                        continue;

                    // Compute inner product between residual and atom
                    float dot = 0.0f;
                    for(int i = 0; i < patch_dim; i++) {
                        dot += residual[i]*D[k*patch_dim + i];
                    }
                    if(fabsf(dot) > fabsf(max_corr)) {
                        max_corr = dot;
                        best_atom = k;
                    }
                }

                if(best_atom == -1)
                    break; // No more atoms

                selected_atoms[s] = best_atom;

                // Build matrix of selected atoms
                float *A = (float*)malloc(sizeof(float)*patch_dim*(s+1));
                for(int j = 0; j <= s; j++) {
                    for(int i = 0; i < patch_dim; i++) {
                        A[j*patch_dim + i] = D[selected_atoms[j]*patch_dim + i];
                    }
                }

                // Solve least squares problem: min_x ||y - A*x||_2^2
                // Compute A^T*A
                float ATA[(s+1)*(s+1)];
                for(int i = 0; i <= s; i++) {
                    for(int j = 0; j <= s; j++) {
                        float sum = 0.0f;
                        for(int k = 0; k < patch_dim; k++) {
                            sum += A[i*patch_dim + k]*A[j*patch_dim + k];
                        }
                        ATA[i*(s+1) + j] = sum;
                    }
                }

                // Compute A^T*y
                float ATy[s+1];
                for(int i = 0; i <= s; i++) {
                    float sum = 0.0f;
                    for(int k = 0; k < patch_dim; k++) {
                        sum += A[i*patch_dim + k]*y[k];
                    }
                    ATy[i] = sum;
                }

                // Solve ATA * x = ATy
                // Add small value to diagonal to ensure invertibility
                for(int i = 0; i <= s; i++) {
                    ATA[i*(s+1)+i] += 1e-6f;
                }

                // Solve the linear system using Gaussian elimination
                float ATA_copy[(s+1)*(s+1)];
                float ATy_copy[s+1];
                memcpy(ATA_copy, ATA, sizeof(float)*(s+1)*(s+1));
                memcpy(ATy_copy, ATy, sizeof(float)*(s+1));

                // Perform Gaussian elimination
                for(int i = 0; i <= s; i++) {
                    // Find pivot
                    int max_row = i;
                    float max_val = fabsf(ATA_copy[i*(s+1)+i]);
                    for(int k = i+1; k <= s; k++) {
                        if(fabsf(ATA_copy[k*(s+1)+i]) > max_val) {
                            max_val = fabsf(ATA_copy[k*(s+1)+i]);
                            max_row = k;
                        }
                    }
                    // Swap rows
                    if(max_row != i) {
                        for(int j = i; j <= s; j++) {
                            float temp = ATA_copy[i*(s+1)+j];
                            ATA_copy[i*(s+1)+j] = ATA_copy[max_row*(s+1)+j];
                            ATA_copy[max_row*(s+1)+j] = temp;
                        }
                        float temp = ATy_copy[i];
                        ATy_copy[i] = ATy_copy[max_row];
                        ATy_copy[max_row] = temp;
                    }
                    // Eliminate
                    for(int k = i+1; k <= s; k++) {
                        float factor = ATA_copy[k*(s+1)+i] / ATA_copy[i*(s+1)+i];
                        for(int j = i; j <= s; j++) {
                            ATA_copy[k*(s+1)+j] -= factor * ATA_copy[i*(s+1)+j];
                        }
                        ATy_copy[k] -= factor * ATy_copy[i];
                    }
                }
                // Back substitution
                for(int i = s; i >= 0; i--) {
                    x[i] = ATy_copy[i];
                    for(int j = i+1; j <= s; j++) {
                        x[i] -= ATA_copy[i*(s+1)+j]*x[j];
                    }
                    x[i] /= ATA_copy[i*(s+1)+i];
                }

                // Update residual: residual = y - A*x
                for(int i = 0; i < patch_dim; i++) {
                    float Ax_i = 0.0f;
                    for(int j = 0; j <= s; j++) {
                        Ax_i += A[j*patch_dim + i]*x[j];
                    }
                    residual[i] = y[i] - Ax_i;
                }

                free(A);
            }

            // Store the sparse code
            for(int s = 0; s < sparsity; s++) {
                if(selected_atoms[s] >= 0) {
                    codes[p*dict_size + selected_atoms[s]] = x[s];
                } else {
                    break;
                }
            }
        } // End of sparse coding step

        // Synchronization barrier to ensure all threads have completed sparse coding
        #pragma omp barrier

        // Dictionary update step (K-SVD)
        #pragma omp parallel for schedule(static)
        for(int k = 0; k < dict_size; k++) {
            // Variables private to each thread
            int usage_count = 0;
            int *patch_indices = NULL;

            // First, count the usage of atom k
            for(int p = 0; p < num_patches; p++) {
                if(codes[p*dict_size + k] != 0.0f) {
                    usage_count++;
                }
            }
            if(usage_count == 0)
                continue; // Skip unused atom

            patch_indices = (int*)malloc(sizeof(int)*usage_count);
            int idx = 0;
            for(int p = 0; p < num_patches; p++) {
                if(codes[p*dict_size + k] != 0.0f) {
                    patch_indices[idx++] = p;
                }
            }

            // Build error matrix
            float *E = (float*)malloc(sizeof(float)*patch_dim*usage_count);
            for(int i = 0; i < usage_count; i++) {
                int p = patch_indices[i];
                float *y = &patches[p*patch_dim];

                // Compute approximation without atom k
                float *approx = (float*)calloc(patch_dim, sizeof(float)); // Use calloc for zero initialization
                for(int j = 0; j < dict_size; j++) {
                    if(j == k)
                        continue;
                    float coeff = codes[p*dict_size + j];
                    if(coeff != 0.0f) {
                        for(int n = 0; n < patch_dim; n++) {
                            approx[n] += D[j*patch_dim + n]*coeff;
                        }
                    }
                }
                // Compute error
                for(int n = 0; n < patch_dim; n++) {
                    E[i*patch_dim + n] = y[n] - approx[n];
                }
                free(approx);
            }

            // Update atom using the mean of errors
            float new_atom[patch_dim];
            memset(new_atom, 0, sizeof(float)*patch_dim);
            for(int i = 0; i < usage_count; i++) {
                for(int n = 0; n < patch_dim; n++) {
                    new_atom[n] += E[i*patch_dim + n];
                }
            }
            // Normalize new_atom
            float norm = 0.0f;
            for(int n = 0; n < patch_dim; n++) {
                norm += new_atom[n]*new_atom[n];
            }
            norm = sqrtf(norm);
            for(int n = 0; n < patch_dim; n++) {
                new_atom[n] /= norm + 1e-6f;
            }

            // Update D
            for(int n = 0; n < patch_dim; n++) {
                D[k*patch_dim + n] = new_atom[n];
            }

            // Update codes
            for(int i = 0; i < usage_count; i++) {
                int p = patch_indices[i];
                // Update code: x_k = <E_k_i, d_k>
                float dot = 0.0f;
                for(int n = 0; n < patch_dim; n++) {
                    dot += E[i*patch_dim + n]*new_atom[n];
                }

                // Use critical section to prevent race conditions
                #pragma omp critical
                {
                    codes[p*dict_size + k] = dot;
                }
            }

            free(E);
            free(patch_indices);
        } // End of dictionary update step

        // Synchronization barrier to ensure all threads have completed dictionary update
        #pragma omp barrier
    }

    // Reconstruct patches
    float *reconstructed_patches = (float*)malloc(sizeof(float)*num_patches*patch_dim);

    #pragma omp parallel for schedule(static)
    for(int p = 0; p < num_patches; p++) {
        float *recon_patch = &reconstructed_patches[p*patch_dim];
        memset(recon_patch, 0, sizeof(float)*patch_dim);
        for(int k = 0; k < dict_size; k++) {
            float coeff = codes[p*dict_size + k];
            if(coeff != 0.0f) {
                for(int n = 0; n < patch_dim; n++) {
                    recon_patch[n] += D[k*patch_dim + n]*coeff;
                }
            }
        }
    }

    // Reconstruct the image from patches
    float *accum_image = (float*)calloc(width*height, sizeof(float));
    float *weight_image = (float*)calloc(width*height, sizeof(float));

    // Accumulate reconstructed patches into the image sequentially
    for(int idx = 0; idx < num_patches; idx++) {
        int p = idx;
        int x = p % num_patches_x;
        int y = p / num_patches_x;
        float *recon_patch = &reconstructed_patches[p*patch_dim];

        for(int dy = 0; dy < patch_size; dy++) {
            for(int dx = 0; dx < patch_size; dx++) {
                int img_x = x + dx;
                int img_y = y + dy;
                int img_idx = img_y * width + img_x;

                accum_image[img_idx] += recon_patch[dy*patch_size + dx];
                weight_image[img_idx] += 1.0f;
            }
        }
    }

    // Normalize the accumulator image
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < width * height; i++) {
        if(weight_image[i] > 0.0f) {
            image[i] = accum_image[i] / weight_image[i];
        }
        // Ensure the pixel value is within [0,1]
        image[i] = fmaxf(0.0f, fminf(1.0f, image[i]));
    }

    // Free allocated memory
    free(patches);
    free(D);
    free(codes);
    free(reconstructed_patches);
    free(accum_image);
    free(weight_image);
}

int main(int argc, char *argv[])
{
    struct timespec start, stop;
    double time;

    unsigned char *image_u8;
    float *image;
    int width = WIDTH;
    int height = HEIGHT;
    float noise_level = 0.1f;

    // loading image
    FILE* fp;
    image_u8 = (unsigned char*)malloc(sizeof(unsigned char)*width*height);
    if(!(fp = fopen(INPUT_FILE, "rb")))
    {
        printf("Cannot open input file\n");
        return 1;
    }
    printf("Image loaded: %dx%d\n", width, height);
    fread(image_u8, sizeof(unsigned char), width*height, fp);
    fclose(fp);

    image = (float*)malloc(sizeof(float)*width*height);
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

    // starting K-SVD
    if(clock_gettime(CLOCK_REALTIME, &start) == -1){perror("clock gettime");}

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
