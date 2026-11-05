#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <stddef.h>

typedef struct {
    double max_abs_err;
    double max_rel_err;
    double frobenius_rel_err;
    int passed;
} ValidationResult;

float* alloc_matrix(int rows, int cols);
void free_matrix(float* mat);
void init_matrices(float* A, float* B, int N, unsigned int seed);
void zero_matrix(float* mat, int rows, int cols);
ValidationResult validate_results(const float* C_ref, const float* C_test, int N, float rtol, float atol);

#endif // MATRIX_UTILS_H