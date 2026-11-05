# Configuração mínima do CMake para o projeto matmul_mpi_blas
cmake_minimum_required(VERSION 3.18)

# Nome do projeto e linguagem principal (C)
project(matmul_mpi_blas C)

# Define o padrão da linguagem C para C11
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Busca o pacote MPI no sistema (essencial para comunicação entre processos)
find_package(MPI REQUIRED)

# Busca o pacote BLAS (essencial para operações de álgebra linear otimizadas)
find_package(BLAS REQUIRED)

# Localiza o cabeçalho cblas.h. 
# Como usamos o Spack, o caminho pode variar, então buscamos em diretórios comuns de bibliotecas.
find_path(CBLAS_INCLUDE_DIR NAMES cblas.h
          HINTS ${BLAS_LIBRARIES}
          PATH_SUFFIXES include ../include)

# Define o executável final e seus arquivos fonte
add_executable(matmul_bench
    src/main.c
    src/matmul_kernels.c
    src/matrix_utils.c
)

# Configura os diretórios de inclusão (headers) para o compilador
target_include_directories(matmul_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include      # Nossos headers locais
    ${MPI_C_INCLUDE_DIRS}                    # Headers do MPI
    ${CBLAS_INCLUDE_DIR}                     # Headers do BLAS/OpenBLAS
)

# Define as bibliotecas que serão linkadas ao executável
target_link_libraries(matmul_bench PRIVATE
    MPI::MPI_C    # Biblioteca MPI
    ${BLAS_LIBRARIES} # Biblioteca BLAS (ex: OpenBLAS)
    m             # Biblioteca matemática padrão do C
)
