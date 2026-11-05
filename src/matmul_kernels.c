#include "matmul_kernels.h"
#include "matrix_utils.h"
#include <stdlib.h>
#include <string.h>
#include <cblas.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

// Calcula a divisão de linhas entre processos MPI (suporta N não divisível por size)
static void compute_partition(int N, int size, int* sendcounts, int* displs, int* row_counts) {
    int base_rows = N / size;
    int rem = N % size;
    int offset = 0;
    for (int p = 0; p < size; ++p) {
        row_counts[p] = base_rows + (p < rem ? 1 : 0);
        sendcounts[p] = row_counts[p] * N;
        displs[p] = offset;
        offset += sendcounts[p];
    }
}

void matmul_serial(const float* A, const float* B, float* C, int N, int bs) {
    zero_matrix(C, N, N);
    for (int ii = 0; ii < N; ii += bs) {
        int i_max = MIN(ii + bs, N);
        for (int kk = 0; kk < N; kk += bs) {
            int k_max = MIN(kk + bs, N);
            for (int jj = 0; jj < N; jj += bs) {
                int j_max = MIN(jj + bs, N);
                for (int i = ii; i < i_max; ++i) {
                    for (int k = kk; k < k_max; ++k) {
                        float a_ik = A[(size_t)i * N + k];
                        for (int j = jj; j < j_max; ++j) {
                            C[(size_t)i * N + j] += a_ik * B[(size_t)k * N + j];
                        }
                    }
                }
            }
        }
    }
}

void matmul_mpi_blocked(const float* A_global, const float* B_global, float* C_global,
                        int N, int bs, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int *sendcounts = malloc(size * sizeof(int));
    int *displs     = malloc(size * sizeof(int));
    int *row_counts = malloc(size * sizeof(int));
    compute_partition(N, size, sendcounts, displs, row_counts);

    int local_rows = row_counts[rank];
    float* A_local = alloc_matrix(local_rows > 0 ? local_rows : 1, N);
    float* C_local = alloc_matrix(local_rows > 0 ? local_rows : 1, N);
    float* B_local = (rank == 0) ? (float*)B_global : alloc_matrix(N, N);

    zero_matrix(C_local, local_rows, N);

    // 1. Distribuir faixas de A e transmitir B
    MPI_Scatterv(A_global, sendcounts, displs, MPI_FLOAT,
                 A_local, local_rows * N, MPI_FLOAT, 0, comm);
    MPI_Bcast(B_local, N * N, MPI_FLOAT, 0, comm);

    // 2. Multiplicação local em blocos (sem BLAS)
    for (int ii = 0; ii < local_rows; ii += bs) {
        int i_max = MIN(ii + bs, local_rows);
        for (int kk = 0; kk < N; kk += bs) {
            int k_max = MIN(kk + bs, N);
            for (int jj = 0; jj < N; jj += bs) {
                int j_max = MIN(jj + bs, N);
                for (int i = ii; i < i_max; ++i) {
                    for (int k = kk; k < k_max; ++k) {
                        float a_ik = A_local[(size_t)i * N + k];
                        for (int j = jj; j < j_max; ++j) {
                            C_local[(size_t)i * N + j] += a_ik * B_local[(size_t)k * N + j];
                        }
                    }
                }
            }
        }
    }

    // 3. Recolher resultados em C_global
    MPI_Gatherv(C_local, local_rows * N, MPI_FLOAT,
                C_global, sendcounts, displs, MPI_FLOAT, 0, comm);

    free_matrix(A_local);
    free_matrix(C_local);
    if (rank != 0) free_matrix(B_local);
    free(sendcounts);
    free(displs);
    free(row_counts);
}

void matmul_mpi_openblas(const float* A_global, const float* B_global, float* C_global,
                         int N, int bs, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int *sendcounts = malloc(size * sizeof(int));
    int *displs     = malloc(size * sizeof(int));
    int *row_counts = malloc(size * sizeof(int));
    compute_partition(N, size, sendcounts, displs, row_counts);

    int local_rows = row_counts[rank];
    float* A_local = alloc_matrix(local_rows > 0 ? local_rows : 1, N);
    float* C_local = alloc_matrix(local_rows > 0 ? local_rows : 1, N);
    float* B_local = (rank == 0) ? (float*)B_global : alloc_matrix(N, N);

    zero_matrix(C_local, local_rows, N);

    // 1. Distribuir faixas de A e transmitir B
    MPI_Scatterv(A_global, sendcounts, displs, MPI_FLOAT,
                 A_local, local_rows * N, MPI_FLOAT, 0, comm);
    MPI_Bcast(B_local, N * N, MPI_FLOAT, 0, comm);

    // 2. Multiplicação local em blocos usando cblas_sgemm
    for (int ii = 0; ii < local_rows; ii += bs) {
        int M_blk = MIN(bs, local_rows - ii);
        for (int kk = 0; kk < N; kk += bs) {
            int K_blk = MIN(bs, N - kk);
            for (int jj = 0; jj < N; jj += bs) {
                int N_blk = MIN(bs, N - jj);

                const float* A_sub = &A_local[(size_t)ii * N + kk];
                const float* B_sub = &B_local[(size_t)kk * N + jj];
                float*       C_sub = &C_local[(size_t)ii * N + jj];

                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            M_blk, N_blk, K_blk,
                            1.0f, A_sub, N,
                                  B_sub, N,
                            1.0f, C_sub, N);
            }
        }
    }

    // 3. Recolher resultados em C_global
    MPI_Gatherv(C_local, local_rows * N, MPI_FLOAT,
                C_global, sendcounts, displs, MPI_FLOAT, 0, comm);

    free_matrix(A_local);
    free_matrix(C_local);
    if (rank != 0) free_matrix(B_local);
    free(sendcounts);
    free(displs);
    free(row_counts);
}