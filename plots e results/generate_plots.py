import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os

# Configuração de estilo visual para relatórios e artigos acadêmicos
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'font.family': 'sans-serif'
})

COLS = [
    "Tag", "Mode", "N", "BlockSize", "MPI_Procs", "OpenBLAS_Threads", 
    "Rep", "Time_s", "GFLOPS", "Validation", "MaxAbsErr", "MaxRelErr", "FrobRelErr", "ValTime_s"
]

def load_csv_or_log(filepath):
    """Carrega dinamicamente qualquer ficheiro de log/CSV filtrando apenas as linhas válidas."""
    if not os.path.exists(filepath):
        print(f"[AVISO] Ficheiro '{filepath}' não encontrado.")
        return pd.DataFrame()
    
    valid_lines = []
    with open(filepath, 'r') as f:
        for line in f:
            line_str = line.strip()
            # Aceita apenas linhas CSV com dados válidos
            if line_str.startswith("CSV,") or (line_str and not line_str.startswith("=") and not line_str.startswith("[") and "Mode" not in line_str):
                valid_lines.append(line_str)
                
    if not valid_lines:
        return pd.DataFrame()
        
    df = pd.read_csv(io.StringIO("\n".join(valid_lines)), names=COLS)
    
    # Conversão de colunas numéricas
    num_cols = ['N', 'BlockSize', 'MPI_Procs', 'OpenBLAS_Threads', 'Rep', 'Time_s', 'GFLOPS', 'ValTime_s']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df.dropna(subset=['GFLOPS', 'Time_s'])


# ==============================================================================
# 1. ESCALABILIDADE FORTE (1 NÓ - par128) -> Lido de `1node.csv`
# ==============================================================================
def plot_strong_scaling(csv_path="1node.csv", output_file="fig1_strong_scaling.png"):
    df = load_csv_or_log(csv_path)
    if df.empty:
        return

    grp = df.groupby(['Mode', 'MPI_Procs'])['GFLOPS'].mean().reset_index()
    
    plt.figure(figsize=(8, 5))
    colors = {'blas': '#1f77b4', 'mpi': '#ff7f0e'}
    labels = {'blas': 'MPI + OpenBLAS (Híbrido)', 'mpi': 'MPI Puro'}
    markers = {'blas': 'o', 'mpi': 's'}

    for mode in ['blas', 'mpi']:
        sub = grp[grp['Mode'] == mode].sort_values('MPI_Procs')
        if not sub.empty:
            plt.plot(sub['MPI_Procs'], sub['GFLOPS'], marker=markers[mode], linewidth=2.5, 
                     markersize=8, color=colors[mode], label=labels[mode])
            for _, row in sub.iterrows():
                plt.annotate(f"{row['GFLOPS']:.1f}", (row['MPI_Procs'], row['GFLOPS']), 
                             textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9, fontweight='bold')

    plt.title('Escalabilidade Forte em 1 Nó (par128: 128 cores) - N=4000, B=256', pad=15)
    plt.xlabel('Número de Processos MPI (P)')
    plt.ylabel('Desempenho (GFLOPS)')
    plt.xticks(sorted(grp['MPI_Procs'].unique()))
    plt.legend(frameon=True, loc='upper right')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"-> Gráfico salvo: {output_file}")


# ==============================================================================
# 2. IMPACTO DO TAMANHO DO BLOCO -> Lido de `2node.csv`
# ==============================================================================
def plot_block_size_impact(csv_path="2node.csv", output_file="fig2_block_size.png"):
    df = load_csv_or_log(csv_path)
    if df.empty:
        return

    filtered = df[(df['N'] == 6000) & (df['Mode'] == 'blas')]
    if filtered.empty:
        filtered = df[df['Mode'] == 'blas']
        
    grp = filtered.groupby(['MPI_Procs', 'BlockSize'])['GFLOPS'].mean().reset_index()
    
    procs = sorted(grp['MPI_Procs'].unique())
    x = np.arange(len(procs))
    bs_values = sorted(grp['BlockSize'].unique())
    width = 0.25
    colors = ['#aec7e8', '#1f77b4', '#08519c']

    plt.figure(figsize=(9, 5))

    for i, bs in enumerate(bs_values):
        sub = grp[grp['BlockSize'] == bs].set_index('MPI_Procs').reindex(procs).reset_index()
        sub['GFLOPS'] = sub['GFLOPS'].fillna(0)
        
        rects = plt.bar(x + i*width, sub['GFLOPS'], width, label=f'Bloco B={int(bs)}', color=colors[i % len(colors)])
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                plt.annotate(f'{height:.0f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    plt.xlabel('Número de Processos MPI (P)')
    plt.ylabel('Desempenho (GFLOPS)')
    plt.title('Impacto do Tamanho do Bloco (B) no Modo Híbrido (N=6000)', pad=15)
    plt.xticks(x + width, [f'P={int(p)}' for p in procs])
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"-> Gráfico salvo: {output_file}")


# ==============================================================================
# 3. ALOCAÇÃO HÍBRIDA (P x T = 256) -> Lido de `2node.csv`
# ==============================================================================
def plot_hybrid_tradeoff(csv_path="2node.csv", output_file="fig3_hybrid_tradeoff.png"):
    df = load_csv_or_log(csv_path)
    if df.empty:
        return

    # Filtra execuções com N=4000, B=256 e modo BLAS onde P * Threads == 256
    hy = df[(df['N'] == 4000) & (df['BlockSize'] == 256) & (df['Mode'] == 'blas')].copy()
    hy['Total_Cores'] = hy['MPI_Procs'] * hy['OpenBLAS_Threads']
    hy = hy[hy['Total_Cores'] == 256]

    if hy.empty:
        print("[AVISO] Nenhuma configuração P*T=256 encontrada em 2node.csv.")
        return

    grp = hy.groupby(['MPI_Procs', 'OpenBLAS_Threads'])['GFLOPS'].mean().reset_index()
    grp = grp.sort_values('MPI_Procs', ascending=False)
    grp['label'] = grp.apply(lambda r: f"{int(r['MPI_Procs'])}x{int(r['OpenBLAS_Threads'])}", axis=1)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(grp['label'], grp['GFLOPS'], color='#2ca02c', width=0.45, edgecolor='black')

    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f} GFLOPS',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.xlabel('Configuração (Processos MPI × Threads OpenBLAS)')
    plt.ylabel('Desempenho Total (GFLOPS)')
    plt.title('Alocação Híbrida Inter/Intra-Nó (Total de 256 Cores - N=4000, B=256)', pad=15)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"-> Gráfico salvo: {output_file}")


# ==============================================================================
# 4. COMPARATIVO LOCAL vs CLUSTER -> Lido de `local.log` e `corretude.log`
# ==============================================================================
def plot_local_vs_cluster(local_log="local.log", cluster_log="corretude.log", output_file="fig4_local_vs_cluster.png"):
    df_loc = load_csv_or_log(local_log)
    df_clus = load_csv_or_log(cluster_log)

    if df_loc.empty or df_clus.empty:
        return

    loc_sub = df_loc[df_loc['N'] == 1000].groupby('Mode')['GFLOPS'].mean()
    clus_sub = df_clus[df_clus['N'] == 1000].groupby('Mode')['GFLOPS'].mean()

    comp_df = pd.DataFrame({'Local': loc_sub, 'Cluster': clus_sub}).reindex(['serial', 'mpi', 'blas']).fillna(0)
    comp_df.index = ['Serial (1 core)', 'MPI (4 cores)', 'MPI+OpenBLAS (4 cores)']

    plt.figure(figsize=(8, 5))
    x = np.arange(len(comp_df))
    width = 0.35

    rects1 = plt.bar(x - width/2, comp_df['Local'], width, label='Local (Ryzen 7 - Zen 4)', color='#d62728')
    rects2 = plt.bar(x + width/2, comp_df['Cluster'], width, label='Cluster Lovelace (Zen 2)', color='#1f77b4')

    for rect in list(rects1) + list(rects2):
        h = rect.get_height()
        if h > 0:
            plt.annotate(f'{h:.1f}', xy=(rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.xlabel('Modo de Execução')
    plt.ylabel('Desempenho (GFLOPS)')
    plt.title('Comparativo de Throughput: Máquina Local vs Cluster HPC (N=1000, B=128)', pad=15)
    plt.xticks(x, comp_df.index)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"-> Gráfico salvo: {output_file}")


# ==============================================================================
# 5. ESCALABILIDADE FRACA (WEAK SCALING) -> Lido de `2node.csv`
# ==============================================================================
def plot_weak_scaling(csv_path="2node.csv", output_file="fig5_weak_scaling.png"):
    df = load_csv_or_log(csv_path)
    if df.empty:
        return

    # Pares de Escalabilidade Fraca: (P, N)
    weak_pairs = [(16, 2000), (64, 2520), (128, 3175), (256, 4000)]
    weak_rows = []

    for p, n in weak_pairs:
        sub = df[(df['MPI_Procs'] == p) & (df['N'] == n)]
        if not sub.empty:
            t_mpi = sub[sub['Mode'] == 'mpi']['Time_s'].mean()
            t_blas = sub[sub['Mode'] == 'blas']['Time_s'].mean()
            g_blas = sub[sub['Mode'] == 'blas']['GFLOPS'].mean()
            weak_rows.append({'P': p, 'N': n, 'Time_MPI': t_mpi, 'Time_BLAS': t_blas, 'GFLOPS_BLAS': g_blas})

    df_weak = pd.DataFrame(weak_rows).dropna()
    if df_weak.empty:
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df_weak))
    x_labels = [f"P={int(row['P'])}\nN={int(row['N'])}" for _, row in df_weak.iterrows()]

    line1 = ax1.plot(x, df_weak['Time_MPI'], marker='s', color='#ff7f0e', linewidth=2.5, label='Tempo MPI Puro')
    line2 = ax1.plot(x, df_weak['Time_BLAS'], marker='o', color='#1f77b4', linewidth=2.5, label='Tempo MPI+OpenBLAS')
    ax1.set_xlabel('Processos MPI (P) / Dimensão Matriz (N)')
    ax1.set_ylabel('Tempo de Execução (s)', color='#ff7f0e')

    ax2 = ax1.twinx()
    line3 = ax2.plot(x, df_weak['GFLOPS_BLAS'], marker='^', color='#2ca02c', linestyle='--', linewidth=2, label='GFLOPS Híbrido')
    ax2.set_ylabel('Desempenho (GFLOPS)', color='#2ca02c')

    plt.xticks(x, x_labels)
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', frameon=True)

    plt.title('Escalabilidade Fraca (Carga Proporcional por Processo)', pad=15)
    fig.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"-> Gráfico salvo: {output_file}")


# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    print("Iniciando a leitura dos ficheiros de log e geração dos gráficos...")
    plot_strong_scaling("1node.csv")
    plot_block_size_impact("2node.csv")
    plot_hybrid_tradeoff("2node.csv")
    plot_local_vs_cluster("local.log", "corretude.log")
    plot_weak_scaling("2node.csv")
    print("Todos os 5 gráficos foram processados e salvos diretamente a partir dos logs!")
