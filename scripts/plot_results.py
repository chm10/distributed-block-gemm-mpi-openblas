#!/bin/bash
#PBS -N bench_matmul
#PBS -l select=2:ncpus=16:mpiprocs=16
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o results_benchmark.log

cd $PBS_O_WORKDIR

module purge
module load gcc
module load openmpi
module load openblas

EXEC="./build/matmul_bench"
OUT_CSV="benchmark_results.csv"

echo "Tag,Mode,N,BlockSize,MPI_Procs,OpenBLAS_Threads,Rep,Time_s,GFLOPS,Validation,MaxAbsErr,MaxRelErr,FrobRelErr,ValTime_s" > $OUT_CSV

# Garantir que o OpenBLAS não crie threads implícitas além das configuradas via código
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

REPS=5

# ==============================================================================
# 1. VERIFICAÇÃO DE CORRETUDE (Tamanhos variados e tamanhos não múltiplos)
# ==============================================================================
echo "=== Executando Verificação de Corretude ==="
for N in 1000 2000; do
    for BS in 64 128 256; do
        mpirun -np 4 $EXEC -n $N -b $BS -m all -t 1 -r 1 -v | grep "^CSV" >> $OUT_CSV
    done
done

# ==============================================================================
# 2. LINHA DE BASE SERIAL (N = 2000, 4000, 6000)
# ==============================================================================
echo "=== Executando Versão Serial ==="
for N in 2000 4000 6000; do
    mpirun -np 1 $EXEC -n $N -b 256 -m serial -t 1 -r $REPS | grep "^CSV" >> $OUT_CSV
done

# ==============================================================================
# 3. IMPACTO DO TAMANHO DO BLOCO E MATRIZ (Escalabilidade Forte)
# ==============================================================================
echo "=== Executando MPI e MPI+OpenBLAS (Escalabilidade Forte) ==="
for N in 2000 4000 6000; do
    for BS in 64 128 256 512; do
        for NP in 1 2 4 8 16 32; do
            # Versão MPI pura (1 thread por processo)
            mpirun --bind-to core -np $NP $EXEC -n $N -b $BS -m mpi -t 1 -r $REPS | grep "^CSV" >> $OUT_CSV
            
            # Versão MPI + OpenBLAS (1 thread por processo MPI)
            mpirun --bind-to core -np $NP $EXEC -n $N -b $BS -m blas -t 1 -r $REPS | grep "^CSV" >> $OUT_CSV
        done
    done
done

# ==============================================================================
# 4. HÍBRIDO: MPI PROCESSOS vs OPENBLAS THREADS (Fixando 16 núcleos totais)
# ==============================================================================
echo "=== Executando Estudo Híbrido (Processos MPI x Threads OpenBLAS) ==="
N=4000
BS=256
for cfg in "16:1" "8:2" "4:4" "2:8" "1:16"; do
    NP="${cfg%%:*}"
    TH="${cfg##*:}"
    export OPENBLAS_NUM_THREADS=$TH
    mpirun --bind-to none -np $NP $EXEC -n $N -b $BS -m blas -t $TH -r $REPS | grep "^CSV" >> $OUT_CSV
done
export OPENBLAS_NUM_THREADS=1

# ==============================================================================
# 5. ESCALABILIDADE FRACA (Trabalho O(N^3 / P) constante por processo)
# Se P dobra, N_novo = N_base * 2^(1/3)
# P=1 -> N=2000 | P=2 -> N=2520 | P=4 -> N=3175 | P=8 -> N=4000 | P=16 -> N=5040 | P=32 -> N=6350
# ==============================================================================
echo "=== Executando Escalabilidade Fraca ==="
BS=256
for pair in "1:2000" "2:2520" "4:3175" "8:4000" "16:5040" "32:6350"; do
    NP="${pair%%:*}"
    N_WEAK="${pair##*:}"
    mpirun --bind-to core -np $NP $EXEC -n $N_WEAK -b $BS -m mpi  -t 1 -r $REPS | grep "^CSV" >> $OUT_CSV
    mpirun --bind-to core -np $NP $EXEC -n $N_WEAK -b $BS -m blas -t 1 -r $REPS | grep "^CSV" >> $OUT_CSV
done

echo "Experimentos finalizados!"