# Implementações de Multiplicação de Matrizes

Este documento explica de forma detalhada as três abordagens implementadas no projeto, discutindo sua lógica, limitações e soluções aplicadas.

## Ambiente de Desenvolvimento (Local)

Você pode rodar as versões **Serial** e **MPI** localmente, desde que tenha spack instaldo com OpenMPI ou MPICH. A versão **BLAS** também funciona localmente se a biblioteca OpenBLAS estiver instalada.

## Compatibilidade com projeto anterior

Carregue o ambiente do 

```shell
# Baixe o spack se não tiver
git clone --depth=2 --branch=releases/v1.2 https://github.com/spack/spack.git ~/spack
# Carregue o ambiente com Spack
source ~/spack/share/spack/setup-env.sh
# Crie um ambiente novo
spack env create cenapad_challenge2
# Ative o novo ambiente
spack env activate cenapad_challenge2

# Adicionar dependências
spack add cmake
spack add openmpi
spack add openblas

# Instalar e compilar no ambiente Spack
spack concretize -f
spack install

spack spec -l > spack_spec_concretized2.txt
```

## Estrutura do Projeto

Para manter o código organizado e facilitar a manutenção, o projeto segue a seguinte estrutura de diretórios:

```text
.
├── CMakeLists.txt          # Configuração de compilação (CMake)
├── include/                # Arquivos de cabeçalho (.h)
│   ├── matmul_kernels.h    # Protótipos das implementações (Serial, MPI, BLAS)
│   └── matrix_utils.h      # Utilitários de matriz (alocação, inicialização, validação)
├── src/                    # Implementação dos algoritmos (.c)
│   ├── main.c              # Ponto de entrada e lógica de benchmark/argumentos
│   ├── matmul_kernels.c    # Implementação das funções de multiplicação
│   └── matrix_utils.c      # Implementação dos utilitários de matriz
├── scripts/                # Scripts de automação
│   ├── run_quick_test.pbs # Script para teste de submissão de experimentos no HPC (PBS)
├── plots and results
│   └── generate_plots.py.py     # Script para visualização de dados (Python/Matplotlib) e resultados
└── build/                  # Diretório de compilação (gerado pelo CMake)
```

### `src/`
Contém os arquivos de implementação em C.
- `main.c`: Ponto de entrada do programa, responsável pelo parsing de argumentos e controle do fluxo.
- `matmul_kernels.c`: Implementação dos kernels de multiplicação de matrizes (Serial, MPI, MPI+OpenBLAS).
- `matrix_utils.c`: Funções utilitárias para manipulação de matrizes (alocação, inicialização, etc.).

### `include/`
Contém os arquivos de cabeçalho (`.h`) utilizados no projeto.
- `matmul_kernels.h`: Definições das funções de multiplicação.
- `matrix_utils.h`: Definições das funções utilitárias.

### `scripts/`
Contém scripts de automação para o cluster e execução de experimentos.
- `compile.pbs`: Script PBS para compilação.
- `run_*.pbs`: Diversos scripts PBS para submissão de experimentos (single node, multi node, etc.).
- `plot_results.py`: Script para geração de gráficos a partir dos resultados.

### `build/`
Diretório gerado pelo CMake contendo os arquivos de compilação e o executável.
- `matmul_bench`: O executável final.

### Build e Dependências
- `CMakeLists.txt`: Arquivo de configuração do sistema de build CMake.
- `spack_spec_concretized2.txt`: Especificação de dependências para o gerenciador de pacotes Spack.

### Resultados e Logs

### Arquivos de Dados (`.csv`)
Armazenam os resultados numéricos dos benchmarks para análise posterior.
- `1node.csv`
- `2node.csv`
- `corretude.log`
- `local.log`

## Como Rodar o Projeto

Ambiente de Desenvolvimento (Local)

**Passos para compilação local:**

1. Certifique-se de ter `cmake`, `gcc` e um ambiente `MPI` instalados.
2. No terminal, dentro da pasta do projeto:
```bash
mkdir build && cd build
cmake ..
make
```

### Testar a compilação local (serial, MPI, MPI + BLAS)

**Execução local:**
* **Serial:** `./matmul_bench -n 1000 -m serial`
* **MPI (ex: 4 processos):** `mpirun -n 4 ./matmul_bench -n 1000 -m mpi`
* **MPI + BLAS (ex: 4 processos, 2 threads BLAS cada):** `mpirun -n 4 ./matmul_bench -n 1000 -m blas -t 2`

Ambiente de Computação de Alto Desempenho (HPC)

No cluster, utilizamos o gerenciador de tarefas **PBS** e o gerenciador de pacotes **Spack** para garantir reprodutibilidade e isolamento de dependências.

### Teste de Corretude
Para validar que as operações são correspondentes foi implementado a flag -v. É possível de testar todas operações locais por meio para matriz de tamanho -n e para todos os modos por meio do -m all e validar por meio da flag -v.

```shell
mpirun -n 4 ./matmul_bench -n 1000 -m all -v -b 128
[VALIDACAO] Matriz de referencia (Serial) calculada em 18.832099 s
CSV,serial,2048,256,4,1,1,19.200231,0.8948,PASS,0.000e+00,0.000e+00,0.000e+00,0.019971
CSV,serial,2048,256,4,1,2,19.085910,0.9001,PASS,0.000e+00,0.000e+00,0.000e+00,0.019854
CSV,serial,2048,256,4,1,3,19.180085,0.8957,PASS,0.000e+00,0.000e+00,0.000e+00,0.020090
CSV,mpi,2048,256,4,1,1,4.862792,3.5329,PASS,0.000e+00,0.000e+00,0.000e+00,0.020015
CSV,mpi,2048,256,4,1,2,4.860770,3.5344,PASS,0.000e+00,0.000e+00,0.000e+00,0.019950
CSV,mpi,2048,256,4,1,3,4.833147,3.5546,PASS,0.000e+00,0.000e+00,0.000e+00,0.019739
CSV,blas,2048,256,4,1,1,0.049046,350.2787,PASS,1.450e-04,1.901e+00,8.510e-07,0.020413
CSV,blas,2048,256,4,1,2,0.049257,348.7818,PASS,1.450e-04,1.901e+00,8.510e-07,0.021046
CSV,blas,2048,256,4,1,3,0.051169,335.7460,PASS,1.450e-04,1.901e+00,8.510e-07,0.019551

```

### Grupo de Identificação (O que foi rodado)
| Coluna | Nome | Descrição | Exemplo (`blas, 2048...`) |
| :--- | :--- | :--- | :--- |
| **1** | `Prefix` | Apenas um identificador literal. | `CSV` |
| **2** | `Modo` | O algoritmo utilizado. | `blas` |
| **3** | `N` | Tamanho da matriz ($N \times N$). | `2048` |
| **4** | `Block Size` | Tamanho do bloco/tile usado nos loops. | `256` |
| **5** | `Size` | Número de processos MPI rodando. | `4` |
| **6** | `Threads` | Número de threads do OpenBLAS. | `1` |
| **7** | `Iteração` | O número da repetição (para médias). | `1` |

### Grupo de Performance (O quão rápido foi)
| Coluna | Nome | Descrição | Por que é importante? |
| :--- | :--- | :--- | :--- |
| **8** | `Tempo (s)` | Tempo total de execução do kernel. | Mede a eficiência bruta. |
| **9** | `GFLOPS` | Bilhões de operações de ponto flutuante por segundo. | É a métrica padrão para comparar hardware/algoritmos. |

### Grupo de Validação (O quão correto foi)
Este é o grupo mais importante para o seu relatório de engenharia.

| Coluna | Nome | Descrição | Explicação Técnica |
| :--- | :--- | :--- | :--- |
| **10** | `Status` | `PASS`, `FAIL` ou `SKIP`. | Indica se o erro passou no seu critério de tolerância. |
| **11** | `Max Abs Error` | Erro Absoluto Máximo. | $\max |C_{ref} - C_{test}|$. Mede a maior diferença "bruta". |
| **12** | `Max Rel Error` | Erro Relativo Máximo. | $\max \frac{|C_{ref} - C_{test}|}{|C_{ref}|}$. Pode explodir se $C_{ref}$ for quase zero. |
| **13** | `Frobenius Err` | Erro de Frobenius Relativo. |  É a norma da matriz inteira: $\frac{\|C_{ref} - C_{test}\|_F}{\|C_{ref}\|_F}$. |
| **14** | `Tempo Val` | Tempo da função de validação. | Tempo que o CPU gastou apenas comparando as matrizes. |


## **Instalação via Spack:**
O uso do Spack permite instalar versões específicas de compiladores e bibliotecas sem interferir no sistema operacional.
```bash
source ~/spack/share/spack/setup-env.sh
spack env create cenapad_challenge2
# Adicionar dependências
spack add cmake
spack add openmpi
spack add openblas

# Instalar e compilar no ambiente Spack
spack concretize -f
spack install

spack spec -l > spack_spec_concretized2.txt
```

**Submissão de Experimentos:**
Para rodar experimentos em larga escala (múltiplos nós e tamanhos de matriz), utilizamos scripts `.pbs`.
```bash
qsub scripts/run_quick_test.pbs
```
*Nota: O script `run_quick_test.pbs` já está configurado para usar o ambiente Spack e realizar uma bateria de testes variando tamanho de matriz, blocos e número de processos.*

Videos para melhor endimento dos códigos

[Multiplicação em Tiling](https://youtu.be/jrfE-SMCHHY)
[Operação com Blas](https://youtu.be/CEdBBi3bgqE)
[MPI com Tiling e Blas](https://www.youtube.com/watch?v=sKnIJjS1VEk)
