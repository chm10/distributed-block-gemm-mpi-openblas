#ifndef MATMUL_KERNELS_H
#define MATMUL_KERNELS_H

#include <mpi.h>

// Versão 1: Serial em blocos (ou direta) sem BLAS
void matmul_serial(const float* A, const float* B, float* C, int N, int block_size);

// Versão 2: MPI em blocos sem BLAS
void matmul_mpi_blocked(const float* A_global, const float* B_global, float* C_global,
                        int N, int block_size, MPI_Comm comm);

// Versão 3: MPI em blocos com OpenBLAS (cblas_sgemm por bloco)
void matmul_mpi_openblas(const float* A_global, const float* B_global, float* C_global,
                         int N, int block_size, MPI_Comm comm);

#endif // MATMUL_KERNELS_H