#include "matrix_utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

float* alloc_matrix(int rows, int cols) {
    float* mat = NULL;
    // Alinhamento de 64 bytes ideal para SIMD (AVX-512 / AVX2) e OpenBLAS
    if (posix_memalign((void**)&mat, 64, (size_t)rows * cols * sizeof(float)) != 0) {
        fprintf(stderr, "Erro ao alocar memória para matriz %dx%d\n", rows, cols);
        exit(EXIT_FAILURE);
    }
    return mat;
}

void free_matrix(float* mat) {
    if (mat) free(mat);
}

void init_matrices(float* A, float* B, int N, unsigned int seed) {
    srand(seed);
    size_t total = (size_t)N * N;
    for (size_t i = 0; i < total; ++i) {
        // Valores no intervalo [-1.0, 1.0] para evitar overflow e underflow
        A[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        B[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
}

void zero_matrix(float* mat, int rows, int cols) {
    memset(mat, 0, (size_t)rows * cols * sizeof(float));
}

ValidationResult validate_results(const float* C_ref, const float* C_test, int N, float rtol, float atol) {
    ValidationResult res = {0.0, 0.0, 0.0, 1};
    double sum_sq_diff = 0.0;
    double sum_sq_ref = 0.0;
    size_t total = (size_t)N * N;

    for (size_t i = 0; i < total; ++i) {
        double ref = (double)C_ref[i];
        double test = (double)C_test[i];
        double abs_err = fabs(ref - test);
        double rel_err = abs_err / (fabs(ref) + 1e-12);

        if (abs_err > res.max_abs_err) res.max_abs_err = abs_err;
        if (rel_err > res.max_rel_err) res.max_rel_err = rel_err;

        sum_sq_diff += abs_err * abs_err;
        sum_sq_ref  += ref * ref;

        // Critério clássico: |C_ref - C_test| <= atol + rtol * |C_ref|
        if (abs_err > (atol + rtol * fabs(ref))) {
            res.passed = 0;
        }
    }

    res.frobenius_rel_err = sqrt(sum_sq_diff) / (sqrt(sum_sq_ref) + 1e-12);
    return res;
}