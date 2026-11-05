#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <cblas.h>
#include "matrix_utils.h"
#include "matmul_kernels.h"

// Declaração fraca/padrão do OpenBLAS para controlar número de threads em tempo de execução
extern void openblas_set_num_threads(int num_threads);
extern int  openblas_get_num_threads(void);

static void print_usage(const char* prog) {
    fprintf(stderr, "Uso: %s -n <tam_matriz> -b <tam_bloco> -m <modo: serial|mpi|blas|all> "
                    "[-t <openblas_threads>] [-r <repeticoes>] [-v (validar)]\n", prog);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 2000;
    int block_size = 256;
    char mode[16] = "all";
    int blas_threads = 1;
    int reps = 3;
    int do_validate = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) N = atoi(argv[++i]);
        else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) block_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) strncpy(mode, argv[++i], 15);
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) blas_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) reps = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) do_validate = 1;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) print_usage(argv[0]);
            MPI_Finalize();
            return 0;
        }
    }

    // Configura explicitamente o número de threads do OpenBLAS
    openblas_set_num_threads(blas_threads);

    float *A = NULL, *B = NULL, *C_res = NULL, *C_ref = NULL;

    if (rank == 0) {
        A = alloc_matrix(N, N);
        B = alloc_matrix(N, N);
        C_res = alloc_matrix(N, N);
        init_matrices(A, B, N, 42);

        if (do_validate) {
            C_ref = alloc_matrix(N, N);
        }
    }

    // Se validação estiver ativa, calcula a referência serial uma vez no rank 0
    // (tempo medido separadamente do benchmark principal)
    double val_ref_time = 0.0;
    if (do_validate && rank == 0) {
        double t0 = MPI_Wtime();
        matmul_serial(A, B, C_ref, N, block_size);
        val_ref_time = MPI_Wtime() - t0;
        printf("[VALIDACAO] Matriz de referencia (Serial) calculada em %.6f s\n", val_ref_time);
    }

    // Execução do(s) modo(s) solicitado(s)
    const char* modes_to_run[3];
    int num_modes = 0;
    if (strcmp(mode, "all") == 0) {
        modes_to_run[0] = "serial";
        modes_to_run[1] = "mpi";
        modes_to_run[2] = "blas";
        num_modes = 3;
    } else {
        modes_to_run[0] = mode;
        num_modes = 1;
    }

    // Tolerâncias adequadas para float (precisão simples ~ 1e-7 por operação, acumulada em N operações)
    float rtol = 1e-4f;
    float atol = 1e-3f;

    for (int m = 0; m < num_modes; ++m) {
        const char* cur_mode = modes_to_run[m];

        for (int r = 0; r < reps; ++r) {
            MPI_Barrier(MPI_COMM_WORLD);
            double t_start = MPI_Wtime();

            if (strcmp(cur_mode, "serial") == 0) {
                if (rank == 0) {
                    matmul_serial(A, B, C_res, N, block_size);
                }
            } else if (strcmp(cur_mode, "mpi") == 0) {
                matmul_mpi_blocked(A, B, C_res, N, block_size, MPI_COMM_WORLD);
            } else if (strcmp(cur_mode, "blas") == 0) {
                matmul_mpi_openblas(A, B, C_res, N, block_size, MPI_COMM_WORLD);
            }

            MPI_Barrier(MPI_COMM_WORLD);
            double t_exec = MPI_Wtime() - t_start;

            // Validação separada da tomada de tempo de execução
            double t_val = 0.0;
            ValidationResult vres = {0.0, 0.0, 0.0, 1};
            if (do_validate && rank == 0) {
                double tv0 = MPI_Wtime();
                vres = validate_results(C_ref, C_res, N, rtol, atol);
                t_val = MPI_Wtime() - tv0;
            }

            if (rank == 0) {
                double gflops = (2.0 * (double)N * (double)N * (double)N) / (t_exec * 1e9);
                printf("CSV,%s,%d,%d,%d,%d,%d,%.6f,%.4f,%s,%.3e,%.3e,%.3e,%.6f\n",
                       cur_mode, N, block_size, size, blas_threads, r + 1,
                       t_exec, gflops,
                       do_validate ? (vres.passed ? "PASS" : "FAIL") : "SKIP",
                       vres.max_abs_err, vres.max_rel_err, vres.frobenius_rel_err, t_val);
                fflush(stdout);
            }
        }
    }

    if (rank == 0) {
        free_matrix(A);
        free_matrix(B);
        free_matrix(C_res);
        if (C_ref) free_matrix(C_ref);
    }

    MPI_Finalize();
    return 0;
}